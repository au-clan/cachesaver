import random
import logging
import asyncio
from typing import TypedDict
from ..typedefs import Algorithm, Model, Agent, Environment, DecodingParameters, State, Benchmark, MAX_SEED
from ..utils import Resampler
logger = logging.getLogger(__name__)

class AgentDictFOA(TypedDict):
    step: Agent # ActAgent
    evaluate: Agent # EvaluateAgent
    step_params: DecodingParameters
    eval_params: DecodingParameters

class AlgorithmFOA(Algorithm):
    def __init__(self, 
                 model: Model, 
                 agents: AgentDictFOA,
                 env: Environment, 
                 num_agents: int, 
                 num_steps: int, 
                 k: int, 
                 backtrack: float, 
                 resampling: str, 
                 origin: float, 
                 min_steps: int,
                 num_evaluations: int
                 ):
        super().__init__(model, agents, env)

        self.step_agent = agents["step"]
        self.eval_agent = agents["evaluate"]

        self.step_params = agents["step_params"]
        self.eval_params = agents["eval_params"]

        self.num_agents = num_agents
        self.num_steps = num_steps
        self.k = k
        self.backtrack = backtrack
        self.resampling = resampling
        self.origin = origin
        self.min_steps = min_steps
        self.num_evaluations = num_evaluations


    async def solve(self, idx: int, state: State, namespace: str, value_cache: dict = None):
        randomness = idx
        random.seed(randomness)
        resampler = Resampler(randomness)

        # Records of previously visited states (state_identifier, state_value, state)
        visited_states = [("INIT", self.origin, state)]

        # Initialize state for each agent
        states = [state.clone(randomness=random.randint(0, MAX_SEED)) for _ in range(self.num_agents)]

        solved = False
        for step in range(self.num_steps):
            print(f"Step {step} ({idx})")

            if solved:
                break
            
            # Generate actions for each state
            action_coroutines = [
                self.step_agent.act(
                    model=self.model,
                    state=state,
                    n=1,
                    namespace=namespace,
                    request_id=f"idx{idx}-step{step}-{hash(state)}-agent{i}",
                    params=self.step_params
                )
                for i, state in enumerate(states)
            ]
            actions = await asyncio.gather(*action_coroutines)

            # Execute actions
            states = [self.env.step(state, action[0]) for state, action in zip(states, actions)]

            # Early stop in case any state is solved
            if any(self.env.evaluate(state)[1] == 1 for state in states):
                solved = True
                break

            # Filter previously visited states records
            remaining_steps = self.num_steps - (step + 1)
            visited_states = [(identifier, value*self.backtrack, state) for identifier, value, state in visited_states]
            visited_states = [state for state in visited_states if  remaining_steps >= self.min_steps - len(state[2].steps)]

            # Pruning : Failed = Finished not correctly
            failed = [i for i, state in enumerate(states) if self.env.is_final(state)]
            if visited_states != []:
                replacements, _ = resampler.resample(visited_states.copy(), len(failed), self.resampling)
            else:
                replacements, _ = resampler.resample([("", 1, state) for state in states], len(failed), resampling_method="linear")
            states = [replacements.pop(0) if i in failed else state for i, state in enumerate(states)]

            # Evaluation phase
            if step < self.num_steps-1 and self.k and step % self.k == 0:
                
                # Evaluate the states
                value_coroutines = [
                    self.eval_agent.act(
                        model=self.model,
                        state=state,
                        n=self.num_evaluations,
                        namespace=namespace,
                        request_id=f"idx{idx}-evaluation{step}-{hash(state)}-agent{i}",
                        params=self.eval_params,
                        cache=value_cache
                    )
                    for i, state in enumerate(states)
                ]
                values = await asyncio.gather(*value_coroutines)

                # Update previously visited states records
                for i, (state, value) in enumerate(zip(states, values)):
                    if i not in failed:
                        visited_states.append((f"{i}.{step}", value, state))

                # Resampling
                states, resampled_idxs = resampler.resample(visited_states, self.num_agents, self.resampling)

        return states
    
    async def benchmark(self, benchmark: Benchmark, share_ns: bool=False, cache: bool=True):
        cache = {} if cache else None
        solve_coroutines = [
            self.solve(
                idx=index,
                state=state,
                namespace="benchmark" if share_ns else f"benchmark_{index}",
                value_cache=cache
            )
            for index, state in benchmark
        ]
        results = await asyncio.gather(*solve_coroutines)
        return results



