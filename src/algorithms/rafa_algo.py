import asyncio
from dataclasses import replace
from typing import TypedDict

from ..algorithm_options.rafa import RAFAOptions, RequestOptions
from ..typedefs import Algorithm, Model, Agent, Environment, Benchmark, DecodingParameters, State

class AgentDictRAFA_tot(TypedDict):
    agent_act: Agent
    agent_eval: Agent
    # model_params: DecodingParameters


class AlgorithmRAFA_tot(Algorithm):

    def __init__(self,
                 model: Model,
                 agents: AgentDictRAFA_tot,
                 env: Environment,
                 rafa_options:RAFAOptions,
                 value_cache:dict=None):

        super().__init__(model, agents, env)

        self.agent_act = agents['agent_act']
        self.agent_eval = agents['agent_eval']
        # self.model_params = agents['model_params']

        self.rafa_options = rafa_options


        # This value_cache should be used as a caching mechanism
        self.value_cache = value_cache  # todo utilize if it is None it means we dont want to cache values

    async def solve(self, idx: int, state: State, namespace: str, value_cache: dict = None):
        # Initial state
        # initial_state = self.reset_rafa(state.puzzle)  # todo verify puzzle is accessible



        request_options = RequestOptions(max_completion_tokens=200,
                                         temperature=1.0,
                                         top_p=1.0,
                                         logprobs=False,
                                         request_id=f"idx{idx}-step{0}-{hash(state)}-agent{0}",
                                         namespace=namespace)

        done = False
        i = 0
        while not done:
            request_options.request_id=f"idx{idx}-step{i}-{hash(state)}-agent{0}"
            i += 1
            #todo this should return the cache_value dict if it should be stored across puzzles...
            state, action, agent_info = await self.agent_act.act(state=state,
                                                                 model=self.model,
                                                                 request_options=request_options,
                                                                 cache_value=self.value_cache,
                                                                 )

            state, obs, reward, done, env_info = self.agent_eval.act(state=state,
                                                                     model=self.model,
                                                                     feedback_print=self.feedback_print,
                                                                     action=action,
                                                                     )

            # log['state_step'].append(state)
            # log['obs_step'].append(obs)
            # log['reward_step'].append(reward)
            # log['done_step'].append(done)
            # log['env_info_step'].append(env_info)
            # state = self.update_rafa(state=state, done=done)
            if done:
                state = replace(state, reflects=[], value_reflects=[])
                i = 0

            # log['state_update'].append(state)
            print(obs)
            print(reward, done, env_info)

            # logs = logs + [log]
        # return logs

        correct = 0
        # for i in range(len(logs)):
        #     # is_correct = self.verification_helper(logs[i]['obs_step'][-1]['answer'])
        #     # if is_correct:
        #     #     correct += 1
        # # verifications = [self.environment.verify(state) for state in states]
        return  correct

    # async def solve(self, idx:int, state: State, namespace: str, value_cache: dict = None):
    #     self.run()

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
