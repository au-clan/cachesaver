import random
import logging
import asyncio
from typing import TypedDict
from omegaconf import OmegaConf
from ..typedefs import Method, Model, Agent, Environment, DecodingParameters, State, Benchmark, MAX_SEED
from .. import MethodFactory, AgentDictFactory
from ..utils import Resampler
logger = logging.getLogger(__name__)

@AgentDictFactory.register
class AgentDictFOA(TypedDict):
    step: Agent # ActAgent
    evaluate: Agent # EvaluateAgent
    step_params: DecodingParameters
    evaluate_params: DecodingParameters

@MethodFactory.register
class MethodFOA(Method):
    def __init__(self, 
                 agents: AgentDictFOA,
                 model: Model, 
                 env: Environment, 
                 config: OmegaConf
                 ):
        super().__init__(model, agents, env, config)

        self.step_agent = agents["step"]
        self.eval_agent = agents["evaluate"]

        self.step_params = agents["step_params"]
        self.evaluate_params = agents["evaluate_params"]
        
        self.num_agents = config.num_agents
        self.num_steps = config.num_steps
        self.k = config.k
        self.backtrack = config.backtrack
        self.resampling = config.resampling
        self.origin = config.origin
        self.min_steps = config.min_steps
        self.num_evaluations = config.num_evaluations


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
                        params=self.evaluate_params,
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
    
    async def benchmark(self, benchmark: Benchmark, ns_ratio: bool=False, cache: bool=True):
        cache = {} if cache else None

        # Set up Namespace distibution
        n_shared = int(ns_ratio * len(benchmark))
        n_unique = len(benchmark) - n_shared
        namespaces = [f"benchmark_{0}" for _ in range(n_shared)] + [f"benchmark_{i+1}" for i in range(n_unique)]
        random.seed(42)
        random.shuffle(namespaces)

        solve_coroutines = [
            self.solve(
                idx=index,
                state=state,
                namespace=ns,
                value_cache=cache
            )
            for (index, state), ns in zip(benchmark, namespaces)
        ]
        results = await asyncio.gather(*solve_coroutines)
        return results