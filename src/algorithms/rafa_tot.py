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
        return response  # todo fix typing
    ## ---------------------------------------------UTILS------------------------------------------------------------------------------------------

    def check_answer(problem: str, answer: str):
        expression = answer.strip().split('\n')[-1].lower().replace('answer: ', '').split('=')[0]
        numbers = re.findall(r'\d+', expression)
        problem_numbers = re.findall(r'\d+', problem)
        if sorted(numbers) != sorted(problem_numbers):
            return False, "The numbers you use are not the original numbers from the problem."
        try:
            # print(sympy.simplify(expression))
            if sympy.simplify(expression) == 24:
                return True, "The formula is correct."
            else:
                return False, "The formula does not lead to 24."

        except Exception as e:
            # print(e)
            return False, "The formula is invalid."

    def check_valid_move(idx, last_step, cur_step):
        if idx == 1:
            original_nums = [float(num) for num in last_step.split(" ")]
        else:
            original_nums = [float(num) for num in last_step.split('left:')[-1].strip("()").split(" ") if
                             num != '']
        formula = [op for op in cur_step.split('left:')[0].strip("()").split(" ") if op != '']
        new_nums = [float(num) for num in cur_step.split('left:')[-1].strip("()").split(" ") if num != '']

        try:
            print(original_nums, new_nums, formula)
            original_nums.remove(float(eval(formula[0])))
            original_nums.remove(float(eval(formula[2])))
            for num in original_nums:
                new_nums.remove(num)
            new_nums.remove(float(formula[4]))
            assert len(new_nums) == 0
        except ValueError:
            return False, "You use value that does not exists in last step or you use them repeatedly; or you drop numbers from the last step."
        except AssertionError:
            return False, "You have more numbers left than expected."

        return True, "The move the valid and correct."

    def check_equation(equation):
        try:
            left, right = equation.split("=")
            err = abs(eval(left) - float(right))
            if err < 1e-10:
                return True, "The Equation is correct."
            else:
                return False, f"The Equation is incorrect, the result should be {eval(left)}"
        except Exception as e:
            print(e)
            return False, "The Equation is not valid."

    def check_twentyfour(cur_step):
        cards = [float(num) for num in cur_step.split('left:')[-1].strip("()").split(" ") if num != '']

        try:
            for nums in itertools.permutations(cards):  # 四个数
                for ops in itertools.product('+-*/', repeat=len(cards) - 1):  # 三个运算符（可重复！）
                    # 构造三种中缀表达式 (bsd)
                    if len(cards) == 4:
                        bds1 = '({0}{4}{1}){5}({2}{6}{3})'.format(*nums, *ops)  # (a+b)*(c-d)
                        bds2 = '(({0}{4}{1}){5}{2}){6}{3}'.format(*nums, *ops)  # (a+b)*c-d
                        bds3 = '{0}{4}({1}{5}({2}{6}{3}))'.format(*nums, *ops)  # a/(b-(c/d))
                        bdss = [bds1, bds2, bds3]
                    elif len(cards) == 3:
                        bds1 = '({0}{3}{1}){4}{2}'.format(*nums, *ops)  # (a+b)*c
                        bds2 = '{0}{3}({1}{4}{2})'.format(*nums, *ops)  # a+(b*c)
                        bdss = [bds1, bds2]
                    elif len(cards) == 2:
                        bds1 = '({0}{2}{1})'.format(*nums, *ops)  # a+b
                        bdss = [bds1]
                    else:
                        if len(nums) == 1 and abs(nums[0] - 24) < 1e-5:
                            return True, ""
                        return False, ""
                    for bds in bdss:  # 遍历
                        try:
                            if abs(eval(bds) - 24.0) < 1e-10:  # eval函数
                                return True, ""
                        except ZeroDivisionError:  # 零除错误！
                            continue
        except Exception as e:
            print(e)
            return False, "It is not a valid formula."
        return False, ""

    ## ---------------------------------------Some more utils in top of agent-----------------------------------------------------------------------------------------------------
    def get_value(self,env, history, x, y, n_evaluate_sample, cache_value=True):
        # validation_prompt = env.validation_prompt_wrap(x, y)
        # if validation_prompt:
        #     validation_outputs = gpt_with_history(validation_prompt, history, n=1, stop=None)
        #     validation = env.validation_outputs_unwrap(x, y, validation_outputs)
        #     if validation == 0:
        #         return 0
        value_prompt = env.value_prompt_wrap(x, y)

        if cache_value and value_prompt in env.value_cache:
            return env.value_cache[value_prompt]
        value_outputs = self.gpt_with_history(value_prompt, history, temperature=0.3, n=n_evaluate_sample, stop=None)
        value = env.value_outputs_unwrap(x, y, value_outputs)
        if cache_value:
            env.value_cache[value_prompt] = value
        return value

    def get_values(self, env, history, x, ys, n_evaluate_sample, cache_value=True):
        values = []
        local_value_cache = {}
        for y in ys:  # each partial output
            if y in local_value_cache and cache_value:  # avoid duplicate candidates
                value = local_value_cache[y]
            else:
                value = self.get_value(env, history, x, y, n_evaluate_sample, cache_value=cache_value)
                if cache_value:
                    local_value_cache[y] = value
            values.append(value)
        return values

    def get_votes(self,env, history, x, ys, n_evaluate_sample):
        vote_prompt = env.vote_prompt_wrap(x, ys)
        vote_outputs = self.gpt_with_history(vote_prompt, history, n=n_evaluate_sample, stop=None)
        values = env.vote_outputs_unwrap(vote_outputs, len(ys))
        return values

    def get_proposals(env, history, x, y, n_propose_sample=10):
        propose_prompt = env.propose_prompt_wrap(x, y)
        proposal_list = [x.split('\n') for x in gpt_with_history(propose_prompt, history, n=1, stop=["\n\n"])]
        proposals = []
        for p in proposal_list:
            proposals.extend(p)
        proposals = proposals[:min(len(proposals), n_propose_sample)]
        return [y + _ + '\n' for _ in proposals]

    def get_samples(env, history, x, y, n_generate_sample, prompt_sample, stop):
        if prompt_sample == 'standard':
            prompt = env.standard_prompt_wrap(x, y)
        elif prompt_sample == 'cot':
            prompt = env.cot_prompt_wrap(x, y)
        else:
            raise ValueError(f'prompt_sample {prompt_sample} not recognized')
        samples = gpt(prompt, n=n_generate_sample, stop=stop)
        return [y + _ for _ in samples]

    ## ---------------------------------------------game24 env------------------------------------------------------------------------------------------
    def reset(self, idx: int):
        self.index = idx
        self.puzzle = self.data[idx]
        self.history = []
        self.feedbacks = []
        self.cur_step = 0
        return {"action": "", "feedback": []}

    def check_step(self, idx, last_step, cur_step):
        try:
            if "answer" in cur_step.lower():
                correct, feedback = self.check_answer(self.puzzle, cur_step)
                if not correct:
                    return f"Step {idx} tries to give an answer but it is incorrect. {feedback}", 0
                return f"Step {idx} is correct. {feedback}", 10
            else:
                # Check if the step is valid
                correct, feedback = self.check_valid_move(idx, last_step, cur_step)
                if not correct:
                    return f"Step {idx} is illegal. {feedback}", 0

                formula = cur_step.split('left:')[0].strip("()")
                correct, feedback = self.check_equation(formula)
                if not correct:
                    return f"Step {idx} is not correctly calculated. {feedback}", 0

                correct, feedback = self.check_twentyfour(cur_step)
                if not correct:
                    return f"Step {idx} is impossible to lead to 24. {feedback}", 0

                return f"Step {idx} is correct and can lead to 24.", 1

        except Exception as e:
            print(e)
            return f"Step {idx} is invalid.", 0

    def generate_feedback(self, action):
        feedbacks = ["Evaluation:"]   # feedbacks for each step
        rewards = 0
        if isinstance(action, list):
            action = action[0]
        actions = action.strip(" \n").split('\n')
        idx = len(self.history)

        for action in actions:
            if idx == 0:
                last_step = self.puzzle
            else:
                last_step = self.history[-1]
            print(last_step)
            # print(action)
            if self.feedback:
                idx += 1
            feedback, reward = self.check_step(idx, last_step, action)
            if self.feedback:
                self.feedbacks.append(feedback)
                feedbacks.append(feedback)
            if reward > 0:
                if self.feedback:
                    self.history.append(action)
                rewards += reward
            else:
                break
        # if 'answer' not in steps[-1].lower():
        #     feedbacks.append("The answer is not complete.")
        total_feedback = " ".join(feedbacks) if self.feedback else None
        return total_feedback, rewards

    def step(self, action):
        self.cur_step += 1
        prev_len = len(self.history)
        feedback, reward = self.generate_feedback(action)
        new_len = len(self.history)
        delta = new_len - prev_len + 1 if new_len < 4 else new_len - prev_len
        assert delta > 0
        done = (reward >= 10) or (self.cur_step > self.max_steps)
        answer = [f"Step {i + 1}: {x}" for i, x in enumerate(action.split('\n')[:delta]) if x != ""]
        answer = "Attempt answer: " + "\n".join(answer)
        if self.feedback:
            info = {'action': action, 'history': self.history}
            obs = {'answer': answer, 'feedback': feedback}
        else:
            info = {'action': action, 'history': []}
            obs = {'answer': answer, 'feedback': []}
        return obs, reward, done, info

    @staticmethod
    def standard_prompt_wrap(x: str, y: str = '') -> str:
        return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y: str = '') -> str:
        return cot_prompt.format(input=x) + y

    @staticmethod
    def propose_prompt_wrap(x: str, y: str = '') -> str:
        current_numbers = get_current_numbers(y if y else x)
        if current_numbers == '24':
            prompt = cot_prompt.format(input=x) + 'Steps:\n' + y
            # print([prompt])
        else:
            prompt = propose_prompt.format(input=current_numbers)
        return prompt

    @staticmethod
    def validation_prompt_wrap(x: str, y: str) -> str or None:
        last_line = y.strip().split('\n')[-1]
        if 'left: ' not in last_line:  # last step
            return
        if len(y.strip().split('\n')) > 1:
            prev_line = get_current_numbers(y.strip().split('\n')[-2])
        else:
            prev_line = x
        return validation_prompt.format(input=prev_line, formula=last_line)

    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        last_line = y.strip().split('\n')[-1]
        if 'left: ' not in last_line:  # last step
            ans = last_line.lower().replace('answer: ', '')
            # print([value_last_step_prompt.format(input=x, answer=ans)])
            return value_last_step_prompt.format(input=x, answer=ans)
        current_numbers = get_current_numbers(y)
        return value_prompt.format(input=current_numbers)

    @staticmethod
    def validation_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        validations = [_.split('\n')[-1] for _ in value_outputs]
        if "invalid" in validations:
            return 0
        return 1

    @staticmethod
    def reflect_prompt_wrap(x: str, y: str, feedback: str) -> str:
        return reflect_prompt.format(input=x, answer=y, feedback=feedback), value_reflect_prompt.format(input=x,
                                                                                                        answer=y,
                                                                                                        feedback=feedback)

    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        if len(y.strip().split('\n')) == 4 and 'answer' not in y.lower():
            return 0
        value_names = [_.split('\n')[-1].lower() for _ in value_outputs]
        value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}  # TODO: ad hoc
        value = sum(value * sum(name in value_name for value_name in value_names) for name, value in value_map.items())
        return value

    ## ---------------------------------------------The actual run------------------------------------------------------------------------------------------
    def plan(self, env, to_print=True):
        x = env.puzzle  # input
        prompt = "Now we would like to play a game of 24. That is, given 4 numbers, try to use them with arithmetic operations (+ - * /) to get 24. "
        obs = [prompt,
               {"feedback": "What you have learned about the game of 24 puzzle are summarized below.\n" + "\n".join(
                   self.reflects)}]

        # ys = get_samples(env, x, obs, '', self.n_generate_sample, self.prompt_sample, stop=None)
        ys = get_samples(env, obs, x, '', self.n_generate_sample, self.prompt_sample, stop=None)

        if to_print:
            print(ys)

        result = ys[0] if len(ys) else ys
        return result, {'proposals': ys}

    def reflect(self, env, obs):
        y = obs['answer']
        feedback = obs['feedback']
        reflect_prompt, value_reflect_prompt = env.reflect_prompt_wrap(env.puzzle, y, feedback)
        # reflects = gpt_with_history(reflect_prompt, obs, stop=None)
        # value_reflects = gpt_with_history(value_reflect_prompt, obs, stop=None)
        reflects = gpt(reflect_prompt, stop=None)
        value_reflects = gpt(value_reflect_prompt, stop=None)
        self.reflects.extend(reflects)
        self.value_reflects.extend(value_reflects)
        return

    def act(self, env, obs):
        if len(obs['feedback']) >= 1:
            self.reflect(env, obs)
        action, info = self.plan(env)
        return action, info

    def update(self, obs, reward, done, info):
        if done:
            self.reflects = []
            self.value_reflects = []
    ## ---------------------------------------------UTILS------------------------------------------------------------------------------------------
    ## ---------------------------------------------UTILS------------------------------------------------------------------------------------------

    ## --------------------------------------------------------------------------------------------------------------------------------------------

    async def solve(self) -> List[State]:
        pass

    async def benchmark(self, benchmark: Benchmark) -> List[List[State]]:
        pass
    #
    # async def get_value(self, env, state, y, config, environment, namespace, request_id):
    #     value_prompt = environment.Prompter.value_prompt_wrap(state.puzzle, y)
    #
    #     if config.framework.cache_value and value_prompt in env.value_cache:
    #         return environment.Prompter.value_cache[value_prompt]  # todo cache values in future
    #     value_outputs = await self.gpt_with_history(prompt=value_prompt, state=state,
    #                                                 n=config.framework.n_evaluate_sample, config=config,
    #                                                 namespace=namespace,
    #                                                 request_id=request_id)  # todo could simply use the state.puzzle_index as the namespace as that is what it is...
    #
    #     value = environment.Prompter.value_outputs_unwrap(state.puzzle, y, value_outputs)
    #     if config.framework.cache_value:
    #         environment.Prompter.value_cache[value_prompt] = value
    #     return value
    #
    # async def get_values(self, env, state, ys, config, environment, request_id, namespace):
    #     values = []
    #     local_value_cache = {}
    #     for y in ys:  # each partial output
    #         if y in local_value_cache and config.framework.cache_value:  # avoid duplicate candidates
    #             value = local_value_cache[y]
    #         else:
    #             # value = get_value(env, history, x, y, n_evaluate_sample, cache_value=cache_value)
    #             value = await self.get_value(env=env, state=state, y=y, config=config, environment=environment,
    #                                          namespace=namespace, request_id=request_id)
    #             if config.framework.cache_value:
    #                 local_value_cache[y] = value
    #         values.append(value)
    #     return values
    #
    # async def get_proposals(self, environment, state, y, config, namespace, request_id):
    #     propose_prompt = environment.Prompter.propose_prompt_wrap(state.puzzle, y)
    #     result = await self.gpt_with_history(prompt=propose_prompt, state=state, n=1, config=config,
    #                                          namespace=namespace, request_id=request_id)
    #
    #     proposal_list = [x.split('\n') for x in result]  # todo the stop token
    #     proposals = []
    #     for p in proposal_list:
    #         proposals.extend(p)
    #     proposals = proposals[:min(len(proposals), config.framework.n_propose_sample)]
    #     return [y + _ + '\n' for _ in proposals]
    #
    # async def get_samples(self, environment, x, y, config, request_id, namespace):
    #     if config.framework.prompt_sample == 'standard':
    #         prompt = environment.Prompter.standard_prompt_wrap(x, y)
    #     elif config.framework.prompt_sample == 'cot':
    #         prompt = environment.Prompter.cot_prompt_wrap(x, y)
    #     else:
    #         raise ValueError(f'prompt_sample {config.framework.prompt_sample} not recognized')
    #     samples = await self.gpt(prompt=prompt, n=config.framework.n_generate_sample, config=config,
    #                              namespace=namespace, request_id=request_id)
    #     return [y + _ for _ in samples]
    #
    # async def plan_rafa(self, state, environment, config):
    #
    #     history = state.history
    #     ys = ["\n".join(history) + "\n"] if len(history) else [""]  # current output candidates
    #     infos = []
    #     prompt = "Now we would like to play a game of 24. That is, given 4 numbers, try to use "
    #     "them with arithmetic operations (+ - * /) to get 24. "
    #     obs = [{"feedback": prompt},
    #            {"feedback": "What you have learned about the puzzle are summarized below.\n" + "\n".join(
    #                state.reflects)}]
    #     value_obs = [prompt,
    #                  dict(feedback="What you have learned about the puzzle are summarized below.\n" + "\n".join(
    #                      state.value_reflects))]
    #     for step in range(4 - len(history)):
    #         # generation
    #
    #         if config.framework.method_generate == 'sample':
    #             coroutines = [
    #                 self.get_samples(environment=environment, x=state.puzzle, y=y, config=config,
    #                                  namespace=str(state.index),
    #                                  request_id=f"step-{str(state.index)}-{step}-{y}-{hash(state)}")
    #                 for y in ys]
    #             new_ys = await asyncio.gather(*coroutines)
    #
    #         elif config.framework.method_generate == 'propose':
    #             coroutines = [
    #                 self.get_proposals(environment=environment, state=state, y=y, config=config,
    #                                    namespace=str(state.index),
    #                                    request_id=f"step-{str(state.index)}-{step}-{y}-{hash(state)}")
    #                 for y in ys]
    #             new_ys = await asyncio.gather(*coroutines)
    #
    #         new_ys = list(itertools.chain(*new_ys))
    #         ids = list(range(len(new_ys)))
    #         # evaluation
    #         if config.framework.method_evaluate == 'vote':
    #             print("vote not impl")
    #         elif config.framework.method_evaluate == 'value':
    #             values = await self.get_values(env=value_obs, state=state, ys=new_ys, config=config,
    #                                            environment=environment, namespace=str(state.index),
    #                                            request_id=f"step-{str(state.index)}-{step}-{step}-{hash(state)}")  # todo step twice, not sure if safe
    #
    #         # selection
    #         if config.framework.method_select == 'sample':
    #             array_values = np.array(values) + 1e-10
    #             ps = array_values / sum(array_values)
    #             select_ids = np.random.choice(ids, size=config.framework.n_select_sample, p=ps).tolist()
    #         elif config.framework.method_select == 'greedy':
    #             select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:config.framework.n_select_sample]
    #         select_new_ys = [new_ys[select_id] for select_id in select_ids]
    #
    #         # log
    #         if config.run.to_print:
    #             sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
    #             print(
    #                 f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
    #
    #         infos.append(
    #             {'step': step, 'x': state.puzzle, 'ys': ys, 'new_ys': new_ys, 'values': values,
    #              'select_new_ys': select_new_ys})
    #         ys = select_new_ys
    #
    #     if config.run.to_print:
    #         print(ys)
    #
    #     ys_list = [y.split('\n')[len(history):] for y in ys]
    #     res_ys = ["\n".join(ys) for ys in ys_list][0]
    #     return state, res_ys, {'steps': infos}
    #
    # async def reflect_rafa(self, state, environment, config):
    #     y = state.answer
    #     feedback = config.framework.feedback
    #     reflect_prompt, value_reflect_prompt = environment.prompter.reflect_prompt_wrap(state.puzzle, y, feedback)
    #     reflects = await self.gpt(prompt=reflect_prompt, n=config.framework.n_generate_sample, config=config)
    #     value_reflects = self.gpt(prompt=value_reflect_prompt, n=config.framework.n_generate_sample, config=config)
    #     state = replace(state, reflects=reflects, value_reflects=value_reflects)
    #
    #     return state
    #
    # async def act_rafa(self, state, environment, config):
    #     if len(state.feedback) >= 1:
    #         state = await self.reflect_rafa(state=state, environment=environment, config=config)
    #     state, action, info = await self.plan_rafa(state=state, environment=environment, config=config)
    #     return state, action, info
    #
    # @staticmethod
    # def update_rafa(state, done):
    #     if done:
    #         state = replace(state, reflects=[], value_reflects=[])
    #     return state
    #
    # def verification_helper(self, input_text):
    #     match = re.search(r'Answer:\s*(.*)=', input_text)
    #     if match:
    #         expression = match.group(1).strip()
    #
    #         try:
    #             # Step 2: Evaluate the expression
    #             result = eval(expression)
    #
    #             # Step 3: Check result
    #             if result == 24:
    #                 return True
    #             else:
    #                 return False
    #         except Exception as e:
    #             print(f"Error evaluating expression: {e}")
    #             return False
    #     else:
    #         print("No valid expression found.")
    #         return False
    #
    # ##RAFA below:
    # def reset_rafa(self, idx: int):
    #     puzzle = self.data.read(idx=idx)
    #     state = StateGame24(
    #         puzzle=puzzle,
    #         history=[],
    #         feedbacks=[],
    #         cur_step=0,
    #         current_state=puzzle,
    #         action=""
    #     )
    #
    #     return state
    #
    # def check_step_rafa(self, environment, state: StateGame24, action): #todo this should be the AgentEvaluateGame24 .act method for feedback
    #     try:
    #         if "answer" in action.lower():
    #             correct, feedback = self.Parser.check_answer(state.puzzle, action)
    #             if not correct:
    #                 return f"Step {state.index} tries to give an answer but it is incorrect. {feedback}", 0
    #             return f"Step {state.index} is correct. {feedback}", 10
    #         else:
    #             # Check if the step is valid
    #             # correct, feedback = self.Parser.check_valid_move(idx, last_step, cur_step)
    #             correct, feedback = environment.Parser.check_valid_move_rafa(state, action)
    #             if not correct:
    #                 return f"Step {state.index} is illegal. {feedback}", 0
    #
    #             formula = action.split('left:')[0].strip("()")
    #             correct, feedback = environment.Parser.check_equation(formula)
    #             if not correct:
    #                 return f"Step {state.index} is not correctly calculated. {feedback}", 0
    #
    #             correct, feedback = environment.Parser.check_twentyfour(action)
    #             if not correct:
    #                 return f"Step {state.index} is impossible to lead to 24. {feedback}", 0
    #
    #             return f"Step {state.index} is correct and can lead to 24.", 1
    #
    #     except Exception as e:
    #         print(e)
    #         return f"Step {state.index} is invalid.", 0
    #
    # def generate_feedback_rafa(self, action, state: StateGame24, environment, config):
    #     feedbacks = ["Evaluation:"]  # feedbacks for each step
    #     rewards = 0
    #     if isinstance(action, list):
    #         action = action[0]
    #     actions = action.strip(" \n").split('\n')
    #     idx = len(state.history)
    #
    #     #
    #
    #     for action in actions:
    #         if idx == 0:
    #             last_step = state.puzzle
    #         else:
    #             last_step = state.history[-1]
    #         print(last_step)
    #         # print(action)
    #         if config.framework.feedback_print:
    #             idx += 1
    #         feedback, reward = self.check_step_rafa(state=state, action=action, environment=environment)
    #         if config.framework.feedback_print:
    #             state = replace(state, feedbacks=state.feedbacks.append(feedback))
    #             feedbacks.append(feedback)
    #         if reward > 0:
    #             if config.framework.feedback_print:
    #                 state = replace(state, history=state.history.append(action))
    #                 # state.history.append(action)
    #             rewards += reward
    #         else:
    #             break
    #
    #     total_feedback = " ".join(feedbacks) if config.framework.feedback_print else None
    #     return state, total_feedback, rewards
    #
    # def step_rafa(self, config, action, state: StateGame24, environment):
    #     # state.cur_step += 1 #todo frozen dataclass
    #     state = replace(state, cur_step=state.cur_step + 1)
    #     prev_len = len(state.history)
    #     generated_state, feedback, reward = self.generate_feedback_rafa(action, state, environment=environment,
    #                                                                     config=config)
    #     new_len = len(state.history)
    #     delta = new_len - prev_len + 1 if new_len < 4 else new_len - prev_len
    #     assert delta > 0
    #     done = (reward >= 10) or (generated_state.cur_step > config.framework.max_step)
    #     answer = [f"Step {i + 1}: {x}" for i, x in enumerate(action.split('\n')[:delta]) if x != ""]
    #     answer = "Attempt answer: " + "\n".join(answer)
    #     if generated_state.feedback:
    #         info = {'action': action, 'history': generated_state.history}
    #         obs = {'answer': answer, 'feedback': feedback}
    #     else:
    #         info = {'action': action, 'history': []}
    #         obs = {'answer': answer, 'feedback': []}
    #     return generated_state, obs, reward, done, info
    #
    # async def solve(self, idx: int, state: State, namespace: str, value_cache: dict = None):
    #     # Initial state
    #     initial_state = self.reset_rafa(idx)
    #     puzzle = initial_state.puzzle
    #
    #     # Randomness initial seed
    #     randomness = idx
    #     random.seed(randomness)
    #
    #     # Set up log
    #     logs = []
    #
    #     state = self.reset_rafa(idx)
    #     log = {'idx': idx,
    #            'state_act': [],
    #            'action_act': [],
    #            'agent_info_act': [],
    #            'state_step': [],
    #            'obs_step': [],
    #            'reward_step': [],
    #            'done_step': [],
    #            'env_info_step': [],
    #            'state_update': []}
    #
    #     done = False
    #     while not done:
    #         state, action, agent_info = await self.act_rafa(state=state, environment=self.environment,
    #                                                         config=self.config)
    #
    #         ##rafa 2.0
    #         action_coroutines = [
    #             self.step_agent.act(
    #                 model=self.model,
    #                 state=state,
    #                 namespace=namespace,
    #                 request_id=f"idx{idx}-step{1}-{hash(state)}-agent{1}",
    #                 params=self.step_params, )
    #         ]
    #
    #         ##
    #         log['state_act'].append(state)
    #         log['action_act'].append(action)
    #         log['agent_info_act'].append(agent_info)
    #         state, obs, reward, done, env_info = self.step_rafa(config=self.config, action=action,
    #                                                             state=state, environment=self.environment)
    #
    #         log['state_step'].append(state)
    #         log['obs_step'].append(obs)
    #         log['reward_step'].append(reward)
    #         log['done_step'].append(done)
    #         log['env_info_step'].append(env_info)
    #         state = self.update_rafa(state=state, done=done)
    #
    #         log['state_update'].append(state)
    #         print(obs)
    #         print(reward, done, env_info)
    #
    #         logs = logs + [log]
    #     # return logs
    #
    #     correct = 0
    #     for i in range(len(logs)):
    #         is_correct = self.verification_helper(logs[i]['obs_step'][-1]['answer'])
    #         if is_correct:
    #             correct += 1
    #     # verifications = [self.environment.verify(state) for state in states]
    #     return logs, correct
    #
    # async def benchmark(self, benchmark: Benchmark, share_ns: bool = False, cache: bool = True):
    #     cache = {} if cache else None
    #     solve_coroutines = [
    #         self.solve(
    #             idx=index,
    #             state=state,
    #             namespace="benchmark" if share_ns else f"benchmark-{index}",
    #             value_cache=cache
    #         )
    #         for index, state in benchmark
    #     ]
    #     results = await asyncio.gather(*solve_coroutines)
    #     return results
