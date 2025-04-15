import asyncio
import itertools
import re
from dataclasses import replace
import random
from typing import TypedDict, List, Any, Coroutine

import numpy as np
from cachesaver.typedefs import Response

from ..typedefs import Algorithm, Model, Agent, Environment, DecodingParameters, State, Benchmark, MAX_SEED, Request


class AgentDictRAFA_tot(TypedDict):
    step: Agent
    evaluate: Agent
    step_params: DecodingParameters
    eval_params: DecodingParameters


class AlgorithmRAFA_tot(Algorithm):
    def __init__(self,
                 model: Model,
                 agents: AgentDictRAFA_tot,
                 env: Environment,

                 prompt_sample=None,
                 naive_run=None,
                 method_generate=None,
                 method_evaluate=None,
                 method_select=None,
                 n_generate_sample=None,
                 n_evaluate_sample=None,
                 n_select_sample=None):
        super().__init__(model, agents, env)

        self.step_agent = agents["step"]
        self.eval_agent = agents["evaluate"]

        self.step_params = agents["step_params"]
        self.eval_params = agents["eval_params"]

        self.prompt_sample = prompt_sample
        self.naive_run = naive_run
        self.method_generate = method_generate
        self.method_evaluate = method_evaluate
        self.method_select = method_select
        self.n_generate_sample = n_generate_sample
        self.n_evaluate_sample = n_evaluate_sample
        self.n_select_sample = n_select_sample

    ##RAFA BELOW-------------------------------------

    async def gpt_with_history(self, prompt, state, n, config, namespace, request_id) -> Response:
        messages = []
        for h in state.history:
            if 'answer' in h:
                messages.extend([{"role": "assistant", "content": h["answer"]}])
            if 'feedback' in h:
                messages.extend([{"role": "user", "content": h["feedback"]}])
        messages.append({"role": "user", "content": prompt})

        response = await self.chatgpt(messages=messages, prompt=prompt, n=n, config=config, namespace=namespace,
                                      request_id=request_id)
        return response

    async def gpt(self, prompt, n, config, namespace, request_id) -> Response:
        messages = [{"role": "user", "content": prompt}]
        return await self.chatgpt(messages=messages, prompt=prompt, n=n, config=config, namespace=namespace,
                                  request_id=request_id)

    async def chatgpt(self, messages, prompt, config, n, namespace, request_id) -> Response:
        response = await self.model.request(
            Request(model="as",
                    prompt=prompt,
                    messages=messages,
                    n=n,
                    request_id=request_id,
                    namespace=namespace,
                    )
        )
        # response = await self.request(
        #     prompt=prompt,
        #     messages=messages,
        #     n=n,
        #     request_id=request_id,
        #     namespace=namespace,
        #     config=config.api.parameters
        # )
        return response  # todo fix typing

    async def get_value(self, env, state, y, config, environment, namespace, request_id):
        value_prompt = environment.Prompter.value_prompt_wrap(state.puzzle, y)

        if config.framework.cache_value and value_prompt in env.value_cache:
            return environment.Prompter.value_cache[value_prompt]  # todo cache values in future
        value_outputs = await self.gpt_with_history(prompt=value_prompt, state=state,
                                                    n=config.framework.n_evaluate_sample, config=config,
                                                    namespace=namespace,
                                                    request_id=request_id)  # todo could simply use the state.puzzle_index as the namespace as that is what it is...

        value = environment.Prompter.value_outputs_unwrap(state.puzzle, y, value_outputs)
        if config.framework.cache_value:
            environment.Prompter.value_cache[value_prompt] = value
        return value

    async def get_values(self, env, state, ys, config, environment, request_id, namespace):
        values = []
        local_value_cache = {}
        for y in ys:  # each partial output
            if y in local_value_cache and config.framework.cache_value:  # avoid duplicate candidates
                value = local_value_cache[y]
            else:
                # value = get_value(env, history, x, y, n_evaluate_sample, cache_value=cache_value)
                value = await self.get_value(env=env, state=state, y=y, config=config, environment=environment,
                                             namespace=namespace, request_id=request_id)
                if config.framework.cache_value:
                    local_value_cache[y] = value
            values.append(value)
        return values

    async def get_proposals(self, environment, state, y, config, namespace, request_id):
        propose_prompt = environment.Prompter.propose_prompt_wrap(state.puzzle, y)
        result = await self.gpt_with_history(prompt=propose_prompt, state=state, n=1, config=config,
                                             namespace=namespace, request_id=request_id)

        proposal_list = [x.split('\n') for x in result]  # todo the stop token
        proposals = []
        for p in proposal_list:
            proposals.extend(p)
        proposals = proposals[:min(len(proposals), config.framework.n_propose_sample)]
        return [y + _ + '\n' for _ in proposals]

    async def get_samples(self, environment, x, y, config, request_id, namespace):
        if config.framework.prompt_sample == 'standard':
            prompt = environment.Prompter.standard_prompt_wrap(x, y)
        elif config.framework.prompt_sample == 'cot':
            prompt = environment.Prompter.cot_prompt_wrap(x, y)
        else:
            raise ValueError(f'prompt_sample {config.framework.prompt_sample} not recognized')
        samples = await self.gpt(prompt=prompt, n=config.framework.n_generate_sample, config=config,
                                 namespace=namespace, request_id=request_id)
        return [y + _ for _ in samples]

    async def plan_rafa(self, state, environment, config):

        history = state.history
        ys = ["\n".join(history) + "\n"] if len(history) else [""]  # current output candidates
        infos = []
        prompt = "Now we would like to play a game of 24. That is, given 4 numbers, try to use "
        "them with arithmetic operations (+ - * /) to get 24. "
        obs = [{"feedback": prompt},
               {"feedback": "What you have learned about the puzzle are summarized below.\n" + "\n".join(
                   state.reflects)}]
        value_obs = [prompt,
                     dict(feedback="What you have learned about the puzzle are summarized below.\n" + "\n".join(
                         state.value_reflects))]
        for step in range(4 - len(history)):
            # generation

            if config.framework.method_generate == 'sample':
                coroutines = [
                    self.get_samples(environment=environment, x=state.puzzle, y=y, config=config,
                                     namespace=str(state.index),
                                     request_id=f"step-{str(state.index)}-{step}-{y}-{hash(state)}")
                    for y in ys]
                new_ys = await asyncio.gather(*coroutines)

            elif config.framework.method_generate == 'propose':
                coroutines = [
                    self.get_proposals(environment=environment, state=state, y=y, config=config,
                                       namespace=str(state.index),
                                       request_id=f"step-{str(state.index)}-{step}-{y}-{hash(state)}")
                    for y in ys]
                new_ys = await asyncio.gather(*coroutines)

            new_ys = list(itertools.chain(*new_ys))
            ids = list(range(len(new_ys)))
            # evaluation
            if config.framework.method_evaluate == 'vote':
                print("vote not impl")
            elif config.framework.method_evaluate == 'value':
                values = await self.get_values(env=value_obs, state=state, ys=new_ys, config=config,
                                               environment=environment, namespace=str(state.index),
                                               request_id=f"step-{str(state.index)}-{step}-{step}-{hash(state)}")  # todo step twice, not sure if safe

            # selection
            if config.framework.method_select == 'sample':
                array_values = np.array(values) + 1e-10
                ps = array_values / sum(array_values)
                select_ids = np.random.choice(ids, size=config.framework.n_select_sample, p=ps).tolist()
            elif config.framework.method_select == 'greedy':
                select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:config.framework.n_select_sample]
            select_new_ys = [new_ys[select_id] for select_id in select_ids]

            # log
            if config.run.to_print:
                sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
                print(
                    f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')

            infos.append(
                {'step': step, 'x': state.puzzle, 'ys': ys, 'new_ys': new_ys, 'values': values,
                 'select_new_ys': select_new_ys})
            ys = select_new_ys

        if config.run.to_print:
            print(ys)

        ys_list = [y.split('\n')[len(history):] for y in ys]
        res_ys = ["\n".join(ys) for ys in ys_list][0]
        return state, res_ys, {'steps': infos}

    async def reflect_rafa(self, state, environment, config):
        y = state.answer
        feedback = config.framework.feedback
        reflect_prompt, value_reflect_prompt = environment.prompter.reflect_prompt_wrap(state.puzzle, y, feedback)
        reflects = await self.gpt(prompt=reflect_prompt, n=config.framework.n_generate_sample, config=config)
        value_reflects = self.gpt(prompt=value_reflect_prompt, n=config.framework.n_generate_sample, config=config)
        state = replace(state, reflects=reflects, value_reflects=value_reflects)

        return state

    async def act_rafa(self, state, environment, config):
        if len(state.feedback) >= 1:
            state = await self.reflect_rafa(state=state, environment=environment, config=config)
        state, action, info = await self.plan_rafa(state=state, environment=environment, config=config)
        return state, action, info

    @staticmethod
    def update_rafa(state, done):
        if done:
            state = replace(state, reflects=[], value_reflects=[])
        return state

    def verification_helper(self, input_text):
        match = re.search(r'Answer:\s*(.*)=', input_text)
        if match:
            expression = match.group(1).strip()

            try:
                # Step 2: Evaluate the expression
                result = eval(expression)

                # Step 3: Check result
                if result == 24:
                    return True
                else:
                    return False
            except Exception as e:
                print(f"Error evaluating expression: {e}")
                return False
        else:
            print("No valid expression found.")
            return False

    ##RAFA below:
    def reset_rafa(self, idx: int):
        puzzle = self.data.read(idx=idx)
        state = StateGame24(
            puzzle=puzzle,
            history=[],
            feedbacks=[],
            cur_step=0,
            current_state=puzzle,
            action=""
        )

        return state

    def check_step_rafa(self, environment, state: StateGame24, action): #todo this should be the AgentEvaluateGame24 .act method for feedback
        try:
            if "answer" in action.lower():
                correct, feedback = self.Parser.check_answer(state.puzzle, action)
                if not correct:
                    return f"Step {state.index} tries to give an answer but it is incorrect. {feedback}", 0
                return f"Step {state.index} is correct. {feedback}", 10
            else:
                # Check if the step is valid
                # correct, feedback = self.Parser.check_valid_move(idx, last_step, cur_step)
                correct, feedback = environment.Parser.check_valid_move_rafa(state, action)
                if not correct:
                    return f"Step {state.index} is illegal. {feedback}", 0

                formula = action.split('left:')[0].strip("()")
                correct, feedback = environment.Parser.check_equation(formula)
                if not correct:
                    return f"Step {state.index} is not correctly calculated. {feedback}", 0

                correct, feedback = environment.Parser.check_twentyfour(action)
                if not correct:
                    return f"Step {state.index} is impossible to lead to 24. {feedback}", 0

                return f"Step {state.index} is correct and can lead to 24.", 1

        except Exception as e:
            print(e)
            return f"Step {state.index} is invalid.", 0

    def generate_feedback_rafa(self, action, state: StateGame24, environment, config):
        feedbacks = ["Evaluation:"]  # feedbacks for each step
        rewards = 0
        if isinstance(action, list):
            action = action[0]
        actions = action.strip(" \n").split('\n')
        idx = len(state.history)

        #

        for action in actions:
            if idx == 0:
                last_step = state.puzzle
            else:
                last_step = state.history[-1]
            print(last_step)
            # print(action)
            if config.framework.feedback_print:
                idx += 1
            feedback, reward = self.check_step_rafa(state=state, action=action, environment=environment)
            if config.framework.feedback_print:
                state = replace(state, feedbacks=state.feedbacks.append(feedback))
                feedbacks.append(feedback)
            if reward > 0:
                if config.framework.feedback_print:
                    state = replace(state, history=state.history.append(action))
                    # state.history.append(action)
                rewards += reward
            else:
                break

        total_feedback = " ".join(feedbacks) if config.framework.feedback_print else None
        return state, total_feedback, rewards

    def step_rafa(self, config, action, state: StateGame24, environment):
        # state.cur_step += 1 #todo frozen dataclass
        state = replace(state, cur_step=state.cur_step + 1)
        prev_len = len(state.history)
        generated_state, feedback, reward = self.generate_feedback_rafa(action, state, environment=environment,
                                                                        config=config)
        new_len = len(state.history)
        delta = new_len - prev_len + 1 if new_len < 4 else new_len - prev_len
        assert delta > 0
        done = (reward >= 10) or (generated_state.cur_step > config.framework.max_step)
        answer = [f"Step {i + 1}: {x}" for i, x in enumerate(action.split('\n')[:delta]) if x != ""]
        answer = "Attempt answer: " + "\n".join(answer)
        if generated_state.feedback:
            info = {'action': action, 'history': generated_state.history}
            obs = {'answer': answer, 'feedback': feedback}
        else:
            info = {'action': action, 'history': []}
            obs = {'answer': answer, 'feedback': []}
        return generated_state, obs, reward, done, info

    async def solve(self, idx: int, state: State, namespace: str, value_cache: dict = None):
        # Initial state
        initial_state = self.reset_rafa(idx)
        puzzle = initial_state.puzzle

        # Randomness initial seed
        randomness = idx
        random.seed(randomness)

        # Set up log
        logs = []

        state = self.reset_rafa(idx)
        log = {'idx': idx,
               'state_act': [],
               'action_act': [],
               'agent_info_act': [],
               'state_step': [],
               'obs_step': [],
               'reward_step': [],
               'done_step': [],
               'env_info_step': [],
               'state_update': []}

        done = False
        while not done:
            state, action, agent_info = await self.act_rafa(state=state, environment=self.environment,
                                                            config=self.config)

            ##rafa 2.0
            action_coroutines = [
                self.step_agent.act(
                    model=self.model,
                    state=state,
                    namespace=namespace,
                    request_id=f"idx{idx}-step{1}-{hash(state)}-agent{1}",
                    params=self.step_params, )
            ]

            ##
            log['state_act'].append(state)
            log['action_act'].append(action)
            log['agent_info_act'].append(agent_info)
            state, obs, reward, done, env_info = self.step_rafa(config=self.config, action=action,
                                                                state=state, environment=self.environment)

            log['state_step'].append(state)
            log['obs_step'].append(obs)
            log['reward_step'].append(reward)
            log['done_step'].append(done)
            log['env_info_step'].append(env_info)
            state = self.update_rafa(state=state, done=done)

            log['state_update'].append(state)
            print(obs)
            print(reward, done, env_info)

            logs = logs + [log]
        # return logs

        correct = 0
        for i in range(len(logs)):
            is_correct = self.verification_helper(logs[i]['obs_step'][-1]['answer'])
            if is_correct:
                correct += 1
        # verifications = [self.environment.verify(state) for state in states]
        return logs, correct

    async def benchmark(self, benchmark: Benchmark, share_ns: bool = False, cache: bool = True):
        cache = {} if cache else None
        solve_coroutines = [
            self.solve(
                idx=index,
                state=state,
                namespace="benchmark" if share_ns else f"benchmark-{index}",
                value_cache=cache
            )
            for index, state in benchmark
        ]
        results = await asyncio.gather(*solve_coroutines)
        return results
