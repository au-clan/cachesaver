import random
from typing import Any

from src.agents.agent_basic import AgentBasic
from src.frameworks.framework_basic import FrameworkBasic
from src.tasks import EnvironmentBasic
from src.tasks.game24 import EnvironmentGame24


class FrameworkRAFA(FrameworkBasic):
    def __init__(self, config: Any, agent: AgentBasic, environment: EnvironmentBasic):
        self.config = config
        self.agent = agent
        self.environment = environment

        super().__init__()
        # RAFA options
        self.prompt_sample = config.framework.prompt_sample
        self.naive_run = config.framework.naive_run
        self.method_generate = config.framework.method_generate
        self.method_evaluate = config.framework.method_evaluate
        self.method_select = config.framework.method_select
        self.n_generate_sample = config.framework.n_generate_sample
        self.n_evaluate_sample = config.framework.n_evaluate_sample
        self.n_select_sample = config.framework.n_select_sample

        self.value_cache = {}

    async def run(self, puzzle_idx: int, namespace: str, seed: int = 0, value_cache: dict = None,
                  step_cache: dict = None):
        # Initial state
        initial_state = self.environment.reset_rafa(puzzle_idx)
        puzzle = initial_state.puzzle

        # Randomness initial seed
        randomness = puzzle_idx + seed
        random.seed(randomness)


        # Set up log
        logs=[]
        log = {}
        log[puzzle_idx] = {"puzzle": puzzle}

        state = self.environment.reset_rafa(puzzle_idx)
        log = {'idx': puzzle_idx, 'agent_info': [], 'env_info': []}
        done = False
        while not done:
            state, action, agent_info = await self.agent.act_rafa(state=state, environment=self.environment,
                                                                  config=self.config)  # todo env is something similar to our env?

            state, obs, reward, done, env_info = self.environment.step_rafa(config=self.config, action=action,
                                                                             state=state, environment=self.environment)
            # agent.update(obs, reward, done, env_info)
            state = self.agent.update_rafa(state=state,done= done)
            log['agent_info'].append(agent_info)
            log['env_info'].append(env_info)
            print(obs)
            print(reward, done, env_info)
            # log['usage_so_far'] = gpt_usage(args.backend)
            logs = logs + [log]
            log+=logs

            # with open(file, 'w') as f:
            #     json.dump(tmp_logs, f, indent=4)
        return logs
        # logs.append(log)
        # with open(file, 'w') as f:
        #     json.dump(logs, f, indent=4)
