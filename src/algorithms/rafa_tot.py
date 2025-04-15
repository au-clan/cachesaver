import asyncio
from dataclasses import replace
from typing import TypedDict

from ..tasks.game24.state import GameState_rafa
from ..typedefs import Algorithm, Model, Agent, Environment, Benchmark


class AgentDictRAFA_tot(TypedDict):
    agent_act: Agent
    agent_eval: Agent

class framework_rafa_settings:
    max_step=1

class model_settings

class AlgorithmRAFA_tot(Algorithm):

    def __init__(self,
                 model: Model,
                 agents: AgentDictRAFA_tot,
                 env: Environment,
                 n_generate_sample: int,
                 n_evaluate_sample: int,
                 n_select_sample: int):
        super().__init__(model, agents, env)
        # self.agent_eval = None
        self.agent_act = agents['agent_act']
        self.agent_eval = agents['agent_eval']

        self.n_generate_sample = n_generate_sample
        self.n_evaluate_sample = n_evaluate_sample
        self.n_select_sample = n_select_sample
        self.value_cache = {}  # todo utilize

    async def solve(self, puzzle_idx: int, state: GameState_rafa, namespace: str, seed: int = 0,
                    value_cache: dict = None,
                    step_cache: dict = None):
        # Initial state
        initial_state = self.reset_rafa(state.puzzle)  # todo verify puzzle is accessible

        # Set up log
        logs = []
        log = {'idx': puzzle_idx,
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
            state, action, agent_info = await self.agent_act.act(state=state, environment=self.environment,
                                                                 config=self.config)
            log['state_act'].append(state)
            log['action_act'].append(action)
            log['agent_info_act'].append(agent_info)
            state, obs, reward, done, env_info = self.agent_eval.act(config=self.config, action=action,
                                                                     state=state, environment=self.environment)

            log['state_step'].append(state)
            log['obs_step'].append(obs)
            log['reward_step'].append(reward)
            log['done_step'].append(done)
            log['env_info_step'].append(env_info)
            # state = self.update_rafa(state=state, done=done)
            if done:
                state = replace(state, reflects=[], value_reflects=[])

            log['state_update'].append(state)
            print(obs)
            print(reward, done, env_info)

            logs = logs + [log]
        # return logs

        correct = 0
        # for i in range(len(logs)):
        #     # is_correct = self.verification_helper(logs[i]['obs_step'][-1]['answer'])
        #     # if is_correct:
        #     #     correct += 1
        # # verifications = [self.environment.verify(state) for state in states]
        return logs, correct

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
