import random
import asyncio

from typing import Any

from .framework_basic import FrameworkBasic
from ..agents.agent_basic import AgentBasic
from ..tasks.basic.environment_basic import EnvironmentBasic
from ..utils import Resampler

class FrameworkFoA(FrameworkBasic):
    def __init__(self, config:Any, agent:AgentBasic, environment: EnvironmentBasic):
        # Todo: Change the config type from Any to something
        self.config = config
        self.agent = agent
        self.environment = environment

        # FoA options
        self.num_agents = config.framework.num_agents
        self.num_steps = config.framework.num_steps
        self.k = config.framework.k
        self.backtrack = config.framework.backtrack
        self.resampling = config.framework.resampling
        self.origin = config.framework.origin

        # Task options
        self.task = config.task.name
        self.min_steps = config.task.min_steps if config.task.min_steps else 1

        # Ablations
        self.caching = config.framework.caching
        self.batching = config.framework.batching
        self.pruning = config.framework.pruning

    async def run(self, puzzle_idx: int, namespace: str, seed: int=0, value_cache: dict=None, step_cache: dict=None):
        
        # Initial state
        initial_state = self.environment.reset(puzzle_idx)
        puzzle = initial_state.puzzle
        
        # Randomness initial seed
        randomness = puzzle_idx + seed
        random.seed(randomness)

        # Initialize the resampler
        resampler = Resampler(randomness)

        # Set up log
        log = {}
        log[puzzle_idx] = {"puzzle": puzzle}
        log[puzzle_idx].update({f"Agent {i}": {} for i in range(self.num_agents)})

        # State identifier shows the step and the agent where the state was visited eg. 0.1 means step 0, agent 1
        # List of states [(state_identifier, state_value, state)]
        records = [("INIT", self.origin, self.environment.reset(puzzle_idx, random.randint(0, 1000)))]

        # Initialize state for each agent
        randies = [random.randint(0, 1000) for _ in range(self.num_agents)]
        states = [self.environment.reset(puzzle_idx, randies[i]) for i in range(self.num_agents)]

        solved = False
        for step in range(self.num_steps):
            print(f"Step {step}")
            
            if solved:
                break

            # Logging
            for agent_id in range(self.num_agents):
                log[puzzle_idx][f"Agent {agent_id}"].update({f"Step {step}": {}})
            
            # Step
            step_coroutines = [
                self.agent.foa_step(
                    state=state, 
                    environment=self.environment, 
                    namespace=namespace, 
                    request_id=f"step-{puzzle_idx}-{step}-{i}-{hash(state)}",
                    cache=step_cache,
                    config=self.config.api.parameters
                )
                for i, state in enumerate(states)
            ]
            states = await asyncio.gather(*step_coroutines)

            # Logging
            for agent_id, state in enumerate(states):
                log[puzzle_idx][f"Agent {agent_id}"][f"Step {step}"].update({"Step": f"{' -> '.join(state.steps)}"})
            
            # Depreciation : old state values are decayed by the backtrack coefficient
            records = [(idx, value*self.config.framework.backtrack, state) for idx, value, state in records]
            
            # Verifications
            verifications = [self.environment.verify(state) for state in states]
            if any([v.correct for v in verifications]):
                solved = True
                break

            # Updating state records (discard states that can't be solved in time or states with value=0)
            remaining_steps = self.num_steps - (step + 1)
            records = [(idx, value, state) for idx, value, state in records if  remaining_steps >= self.min_steps - len(state.steps)]
            records = [(idx, value, state) for idx, value, state in records if value>0]

            # Pruning
            idxs_invalid = [idx for idx, v in enumerate(verifications) if (v.finished and not v.correct)]
            if records != []:
                replacements, _ = resampler.resample(records.copy(), len(idxs_invalid), self.config.framework.resampling)
            else:
                replacements, _ = resampler.resample([("", 1, state) for state in states], len(idxs_invalid), resampling_method="linear")
            states = [replacements.pop(0) if i in idxs_invalid else state for i, state in enumerate(states)]

            #Todo: Log - Pruning

            # Resampling
            if step < self.config.framework.num_steps-1 and self.config.framework.k > 0 and step % self.config.framework.k == 0:
                
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

                # Update records
                for idx, (state, value) in enumerate(zip(states, values)):
                    if idx not in idxs_invalid:
                        records.append((f"{idx}.{step}", value, state))
                
                # Logging - Evaluation
                for agent_id, value in enumerate(values):
                    log[puzzle_idx][f"Agent {agent_id}"][f"Step {step}"].update({"Evaluation": value})

                # Resampling
                states, resampled_idxs = resampler.resample(records, self.config.framework.num_agents, self.config.framework.resampling)

                # Logging - Resampling
                for agent_id, resampled_idx in enumerate(resampled_idxs):
                    log[puzzle_idx][f"Agent {agent_id}"][f"Step {step}"].update({"Resampling": {"Idx":records[resampled_idx][0], "Resampled state": records[resampled_idx][2].current_state, "Value": records[resampled_idx][1],}})
            
        verifications = [self.environment.verify(state) for state in states]
        log[puzzle_idx]["Input"] = puzzle
        log[puzzle_idx]["Verifications"] = verifications

        return states, verifications, log