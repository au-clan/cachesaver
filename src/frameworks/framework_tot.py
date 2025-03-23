import random
import asyncio

from typing import Any

from .framework_basic import FrameworkBasic
from ..agents.agent_basic import AgentBasic
from ..tasks.basic.environment_basic import EnvironmentBasic

class FrameworkToT(FrameworkBasic):
    def __init__(self, config:Any, agent:AgentBasic, environment: EnvironmentBasic):
        # Todo: Change the config type from Any to something
        self.config = config
        self.agent = agent
        self.environment = environment

        # ToT options
        self.n_select_sample = config.framework.n_select_sample


        # Task options
        self.task = config.task.name
        self.min_steps = config.task.min_steps if config.task.min_steps else 1

    async def run(self, puzzle_idx: int, namespace: str, seed: int=0, value_cache: dict=None, step_cache: dict=None):
        
         # Randomness initial seed
        randomness = puzzle_idx + seed
        random.seed(randomness)
        
        # Initial state
        initial_state = self.environment.reset(puzzle_idx)
        puzzle = initial_state.puzzle
        states = [self.environment.reset(puzzle_idx, random.randint(0, 1000))]
        
        # Set up log
        log = {}
        log[puzzle_idx] = {"puzzle": puzzle}

        for step in range(self.min_steps):
            print(f"Step {step}")

            
            # Step
            step_coroutines = [
                self.agent.tot_step(
                    state=state, 
                    environment=self.environment, 
                    namespace=namespace, 
                    request_id=f"step-{puzzle_idx}-{step}-{i}-{hash(state)}",
                    cache=step_cache,
                    config=self.config.api.parameters
                )
                for i, state in enumerate(states)
            ]
            branches = await asyncio.gather(*step_coroutines) # Suggested states for each branch
            states = [state for branch in branches for state in branch] # Flatten suggestions

            # Logging - Step
            log[puzzle_idx]["Step {step}"] = {}
            log[puzzle_idx]["Step {step}"]["Suggestions"] = [f"{' -> '.join(state.steps)}" for state in states]
            
            # Evaluation
            value_coroutines = [
                self.agent.evaluate(
                    state=state, 
                    environment=self.environment, 
                    n=self.config.framework.evaluations, 
                    namespace=namespace, 
                    request_id=f"evaluate-{puzzle_idx}-{step}-{i}-{hash(state)}",
                    cache=value_cache,
                    config=self.config.api.parameters
                )
                for i, state in enumerate(states)
            ]
            values = await asyncio.gather(*value_coroutines)

            # Logging - Evaluation
            log[puzzle_idx]["Step {step}"]["Evaluations"] = values

            # Selection
            agent_ids = [f"{step}.{i}" for i in range(len(states))]
            state_value_pairs = list(zip(states, values, agent_ids))
            sorted_pairs = sorted(state_value_pairs, key=lambda x: x[1], reverse=True)
            states, values, selected_ids = map(list, zip(*sorted_pairs[:self.n_select_sample])) 

            # Logging - Selection
            log[puzzle_idx]["Step {step}"]["Selected"] = [f"{id} : {' -> '.join(state.steps)}" for id, state in zip(selected_ids,states)]
            
        verifications = [self.environment.verify(state) for state in states]
        log[puzzle_idx]["Input"] = puzzle
        log[puzzle_idx]["Verifications"] = verifications

        return states, verifications, log