import random
import asyncio
from tqdm.asyncio import tqdm
from typing import TypedDict
from ..typedefs import Algorithm, Model, Agent, Environment, DecodingParameters, State, Benchmark, MAX_SEED

class AgentDictGOT(TypedDict):
    step: Agent
    aggregate: Agent
    evaluate: Agent
    step_params: DecodingParameters
    aggregate_params: DecodingParameters
    eval_params: DecodingParameters

class AlgorithmGOT(Algorithm):
    def __init__(self, 
                 model, 
                 agents, 
                 env, 
                 num_selections, 
                 num_steps,
                 num_best,
                 num_evaluations,
                 ):
        super().__init__(model, agents, env)

        self.step_agent = agents["step"]
        self.aggregate_agent = agents["aggregate"]
        self.eval_agent = agents["evaluate"]

        self.step_params = agents["step_params"]
        self.aggregate_params = agents["aggregate_params"]
        self.eval_params = agents["eval_params"]

        self.num_selections = num_selections
        self.num_steps = num_steps
        self.num_best = num_best
        self.num_evaluations = num_evaluations

    async def solve(self, idx:int, state: State, namespace: str, value_cache: dict = None):
        randomness = idx
        random.seed(randomness)
        states = [state.clone(randomness=random.randint(0, MAX_SEED))]

        for step in range(self.num_steps):

            # Generate actions for each state
            action_coroutines = [
                self.step_agent.act(
                    model=self.model,
                    state=state,
                    namespace=namespace,
                    request_id=f"idx{idx}-step{step}-{hash(state)}",
                    params=self.step_params,
                )
                for state in states
            ]
            actions = await asyncio.gather(*action_coroutines)

            # Aggregate actions
            aggregate_coroutines = [
                self.aggregate_agent.act(
                    model=self.model,
                    state=state,
                    actions=actions,
                    k=self.num_selections,
                    namespace=namespace,
                    request_id=f"idx{idx}-aggregate{step}-{hash(state)}",
                    params=self.aggregate_params,
                )
                for state, actions in zip(states, actions)
            ]
            actions = await asyncio.gather(*aggregate_coroutines)

            # Execute actions on environment
            state_proposals = []
            for state, actions in zip(states, actions):
                for action in actions:
                    state_proposals.append(self.env.step(state, action))

            # Evaluate all proposals
            value_coroutines = [
                self.eval_agent.act(
                    model=self.model,
                    state=state,
                    n=self.num_evaluations,
                    namespace=namespace,
                    request_id=f"idx{idx}-evaluation{step}-{hash(state)}",
                    params=self.eval_params,
                    cache=value_cache
                )
                for state in state_proposals
            ]
            values = await asyncio.gather(*value_coroutines)

            # Choose the best states based on their value
            state_value_pairs = list(zip(state_proposals, values))
            sorted_pairs = sorted(state_value_pairs, key=lambda x: x[1], reverse=True)
            states, values = map(list, zip(*sorted_pairs[:self.num_best]))
        
        return states

            
    
    async def benchmark(self, benchmark, share_ns: bool=False, cache: bool=True):
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
        results = await tqdm.gather(*solve_coroutines)
        return results

