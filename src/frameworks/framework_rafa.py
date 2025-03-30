import random
import re
from typing import Any

from src.agents.agent_basic import AgentBasic
from src.frameworks.framework_basic import FrameworkBasic
from src.tasks import EnvironmentBasic


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

        self.value_cache = {} #todo use as they do in the rafa code
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
    async def run(self, puzzle_idx: int, namespace: str, seed: int = 0, value_cache: dict = None,
                  step_cache: dict = None):
        # Initial state
        initial_state = self.environment.reset_rafa(puzzle_idx)
        puzzle = initial_state.puzzle

        # Randomness initial seed
        randomness = puzzle_idx + seed
        random.seed(randomness)

        # Set up log
        logs = []

        state = self.environment.reset_rafa(puzzle_idx)
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
            state, action, agent_info = await self.agent.act_rafa(state=state, environment=self.environment,
                                                                  config=self.config)
            log['state_act'].append(state)
            log['action_act'].append(action)
            log['agent_info_act'].append(agent_info)
            state, obs, reward, done, env_info = self.environment.step_rafa(config=self.config, action=action,
                                                                            state=state, environment=self.environment)

            log['state_step'].append(state)
            log['obs_step'].append(obs)
            log['reward_step'].append(reward)
            log['done_step'].append(done)
            log['env_info_step'].append(env_info)
            state = self.agent.update_rafa(state=state, done=done)

            log['state_update'].append(state)
            print(obs)
            print(reward, done, env_info)

            logs = logs + [log]
        # return logs

        correct=0
        for i in range(len(logs)):
            is_correct = self.verification_helper(logs[i]['obs_step'][-1]['answer'])
            if is_correct:
                correct += 1
        # verifications = [self.environment.verify(state) for state in states]
        return logs, correct
        #todo return verifications but as they atm are all wrong i am not sure of prober impl
        # logs.append(log)
        # with open(file, 'w') as f:
        #     json.dump(logs, f, indent=4)
