import asyncio
import itertools
import random
from typing import List, Any
from dataclasses import dataclass, replace

import numpy as np
from omegaconf.dictconfig import DictConfig

from cachesaver.typedefs import BatchRequestModel
from cachesaver.typedefs import Request, Response

from .agent_basic import AgentBasic
from ..tasks.basic import StateBasic, EnvironmentBasic
from ..tasks.game24 import EnvironmentGame24


@dataclass(frozen=True)
class Request(Request):
    max_completion_tokens: int
    temperature: float
    top_p: float
    stop: str


class AgentLLM(AgentBasic):
    def __init__(self, api: BatchRequestModel):
        self.name = "LLM Agent"
        self.api = api
        self.calls = {"total": 0, "cached": 0, "duplicated": 0}
        self.tokens = {
            "total": {"in": 0, "out": 0},
            "cached": {"in": 0, "out": 0},
            "generated": {"in": 0, "out": 0}
        }

    async def request(self, prompt: str, n: int, request_id: str, namespace: str, config: DictConfig,
                      messages: List[Any]) -> List[Any]:
        """
        Makes a request to the api and tracks the number of calls.
        """
        request = Request(
            prompt=prompt,  # todo maybe just use this for storing in cache? idk
            messages=messages,
            n=n,
            request_id=request_id,
            namespace=namespace,
            max_completion_tokens=config.max_completion_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            stop=config.stop
        )
        response = await self.api.request(request)
        self.calls = {
            "total": self.calls["total"] + len(response.data),
            "cached": self.calls["cached"] + sum(response.cached),
            "duplicated": self.calls["duplicated"] + sum(response.duplicated)
        }

        messages, tokin, tokout = zip(*response.data)
        cached_tokin = [int(tokens * cached) for tokens, cached in zip(tokin, response.cached)]
        cached_tokout = [int(tokens * cached) for tokens, cached in zip(tokout, response.cached)]
        generated_tokin = [int(tokens * (not cached)) for tokens, cached in zip(tokin, response.cached)]
        generated_tokout = [int(tokens * (not cached)) for tokens, cached in zip(tokout, response.cached)]

        self.tokens["total"]["in"] += sum(tokin)
        self.tokens["total"]["out"] += sum(tokout)
        self.tokens["cached"]["in"] += sum(cached_tokin)
        self.tokens["cached"]["out"] += sum(cached_tokout)
        self.tokens["generated"]["in"] += sum(generated_tokin)
        self.tokens["generated"]["out"] += sum(generated_tokout)
        return messages

    async def act_step(self, state: StateBasic, environment: EnvironmentBasic, namespace: str, request_id: str,
                       config: DictConfig, cache: dict = None) -> StateBasic:
        """
        Returns the next state after performing 1 action.
        """
        if cache is not None and state in cache:
            inference = cache[state]
        else:
            prompt = environment.prompter.act(state)
            response = await self.request(
                prompt=prompt,
                n=1,
                request_id=request_id,
                namespace=namespace,
                config=config
            )
            inference = environment.parser.act(response[0])

            if cache is not None:
                cache[state] = inference

        random.seed(state.randomness)
        randomness = random.randint(0, 1000)
        new_state = environment.get_next_state(
            inference=inference,
            state=state,
            randomness=randomness
        )
        return new_state

    async def react_step(self, state: StateBasic, environment: EnvironmentBasic, namespace: str, request_id: str,
                         config: DictConfig, cache: dict = None) -> StateBasic:
        """
        Returns the next state after performing 1 thought and 1 action.
        """
        if cache is not None and state in cache:
            inference = cache[state]
        else:
            prompt = environment.prompter.react(state)
            response = await self.request(
                prompt=prompt,
                n=1,
                request_id=request_id,
                namespace=namespace,
                config=config
            )
            inference = environment.parser.react(response[0])

            if cache is not None:
                cache[state] = inference

        random.seed(state.randomness)
        randomness = random.randint(0, 1000)
        new_state = environment.get_next_state(
            inference=inference,
            state=state,
            randomness=randomness
        )
        return new_state

    async def foa_step(self, state: StateBasic, environment: EnvironmentBasic, namespace: str, request_id: str,
                       config: DictConfig, cache: dict = None) -> StateBasic:
        """
        Returns a list of proposals for the given state.
        """
        if cache is not None and state in cache:
            inferences = cache[state]
        else:
            prompt = environment.prompter.bfs(state)
            response = await self.request(
                prompt=prompt,
                n=1,
                request_id=request_id,
                namespace=namespace,
                config=config
            )
            inferences = environment.parser.bfs(response[0])

            if cache is not None:
                cache[state] = inferences

        random.seed(state.randomness)
        select_inference = random.choice(inferences)
        randomness = random.randint(0, 1000)
        new_state = environment.get_next_state(
            inference=select_inference,
            state=state,
            randomness=randomness
        )
        return new_state

    async def tot_step(self, state: StateBasic, environment: EnvironmentBasic, namespace: str, request_id: str,
                       config: DictConfig, cache: dict = None) -> List[StateBasic]:
        """
        Returns a list of proposals for the given state.
        """
        if cache is not None and state in cache:
            inferences = cache[state]
        else:
            prompt = environment.prompter.bfs(state)
            response = await self.request(
                prompt=prompt,
                n=1,
                request_id=request_id,
                namespace=namespace,
                config=config
            )
            inferences = environment.parser.bfs(response[0])

            if cache is not None:
                cache[state] = inferences
        randies = [random.randint(0, 1000) for _ in range(len(inferences))]
        new_states = [
            environment.get_next_state(
                inference=inference,
                state=state,
                randomness=randies[i]
            )
            for i, inference in enumerate(inferences)
        ]
        return new_states

    async def evaluate(self, state: StateBasic, environment: EnvironmentBasic, n: int, namespace: str, request_id: str,
                       config: DictConfig, cache: dict = None) -> int:
        """
        Some tasks use exlucively deterministc methods for evaluation (humaneval), others exlucively llm-based methods (mini crosswords) and others a mixture of the two (game24).

        For this, even in this LLM-based agent, we employ a 2 step methods. First we employ the deterministic method, if it fails to find a value, we employ the llm-based method.
        """

        if cache is not None and state in cache:
            inference = cache[state]

        # Deterministic method
        inference = environment.get_value(state)

        # LLM-based method
        if inference.value is None:
            prompt = environment.prompter.evaluate(state)
            response = await self.request(
                prompt=prompt,
                n=n,
                request_id=request_id,
                namespace=namespace,
                config=config
            )
            inference = environment.parser.evaluate(response)

        if cache is not None:
            cache[state] = inference

        value = inference.value
        return value

    ##RAFA BELOW-------------------------------------
    # def completions_with_backoff(**kwargs):
    #     if "prompt" in kwargs:
    #         return openai.Completion.create(**kwargs)
    #     else:
    #         assert "messages" in kwargs, "Either prompt or messages must be provided"
    #         return openai.ChatCompletion.create(**kwargs)

    async def gpt_with_history(self, prompt, state, n, config, namespace, request_id) -> list:
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

    async def gpt(self, prompt, n, config, namespace, request_id) -> list:
        messages = [{"role": "user", "content": prompt}]
        return await self.chatgpt(messages=messages, prompt=prompt, n=n, config=config, namespace=namespace,
                                  request_id=request_id)

    async def chatgpt(self, messages, prompt, config, n, namespace, request_id) -> list:
        response = await self.request(
            prompt=prompt,
            messages=messages,
            n=n,
            request_id=request_id,
            namespace=namespace,
            config=config.api.parameters
        )
        return response

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
        # propose_prompt = env.propose_prompt_wrap(state.puzzle, y)
        propose_prompt = environment.Prompter.propose_prompt_wrap(state.puzzle, y)
        # proposal_list = [x.split('\n') for x in self.gpt_with_history(propose_prompt, state.history, n=1, stop=["\n\n"])] #todo the stop token
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

                # new_ys = [
                #     self.get_samples( environment=environment,x=state.puzzle, y=y, config=config)
                #     for y in ys]

                coroutines = [
                    self.get_samples(environment=environment, x=state.puzzle, y=y, config=config,
                                     namespace=str(state.index),
                                     request_id=f"step-{str(state.index)}-{step}-{y}-{hash(state)}")
                    for y in ys]
                new_ys = await asyncio.gather(*coroutines)

            elif config.framework.method_generate == 'propose':
                # new_ys = [self.get_proposals(obs, state.puzzle, y, config.framework.n_generate_sample) for y in ys]
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
                # values = get_votes(env, value_obs, x, new_ys, self.n_evaluate_sample)
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
        # if len(ys):
        #     return ys[0], {'steps': infos}
        ys_list = [y.split('\n')[len(history):] for y in ys]
        res_ys = ["\n".join(ys) for ys in ys_list][0]
        return state, res_ys, {'steps': infos}

    async def reflect_rafa(self, state, environment, config):
        y = state.answer
        feedback = state.feedback
        reflect_prompt, value_reflect_prompt = environment.prompter.reflect_prompt_wrap(state.puzzle, y, feedback)
        reflects = await self.gpt(prompt=reflect_prompt, n=config.framework.n_generate_sample, config=config)
        value_reflects = self.gpt(prompt=value_reflect_prompt, n=config.framework.n_generate_sample, config=config)
        state.reflects.extend(reflects)  # todo cant extend frozen dataclass
        state.value_reflects.extend(value_reflects)
        return state

    async def act_rafa(self, state, environment, config):
        if len(state.feedback) >= 1:
            state = await self.reflect_rafa(state=state, environment=environment, config=config)
        state, action, info = await self.plan_rafa(state=state, environment=environment, config=config)
        return state, action, info

    def update_rafa(self, state, done):
        if done:
            state = replace(state, reflects=[], value_reflects=[])
            # state.value_reflects = []
        return state
