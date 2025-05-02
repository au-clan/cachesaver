import random
import logging
import asyncio
from typing import TypedDict
from ..typedefs import Algorithm, Model, Agent, Environment, DecodingParameters, State, Benchmark, MAX_SEED
logger = logging.getLogger(__name__)

class AgentDictTOT(TypedDict):
    step: Agent
    reflect: Agent
    evaluate: Agent
    step_params: DecodingParameters
    reflect_params: DecodingParameters
    eval_params: DecodingParameters
# fill in all benchmarks for react
# adding algo for reflexion

class AlgorithmReflexion(Algorithm):
    def __init__(self,
                model: Model,
                agents: AgentDictTOT,
                env: Environment,
                num_selections: int,
                num_steps: int,
                num_evaluations: int
                ):
        super().__init__(model, agents, env)

        self.step_agent = agents["step"]
        self.reflect_agent = agents["reflect"]
        self.eval_agent = agents["evaluate"]

        self.step_params = agents["step_params"]
        self.reflect_params = agents["reflect_params"]
        self.eval_params = agents["eval_params"]

        self.num_selections = num_selections
        self.num_steps = num_steps
        self.num_evaluations = num_evaluations

    async def solve(self, idx:int, state: State, namespace: str, value_cache: dict = None):

        randomness = idx
        random.seed(randomness)
        states = [state.clone(randomness=random.randint(0, MAX_SEED))]
        print(f"idx={idx}, start")
        for step in range(self.num_steps):
            # Generate actions for each state
            action_coroutines = [
                self.step_agent.act(
                    model=self.model,
                    state=state,
                    namespace=namespace,
                    request_id=f"idx{idx}-step{step}-{hash(state)}-agent{i}",
                    params=self.step_params,
                )
                for i, state in enumerate(states)
            ]
            actions = await asyncio.gather(*action_coroutines)

            # Execute actions
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
                    request_id=f"idx{idx}-evaluation{step}-{hash(state)}-agent{i}",
                    params=self.eval_params,
                    cache=value_cache
                )
                for i, state in enumerate(state_proposals)
            ]
            values = await asyncio.gather(*value_coroutines)

            # reflect actions
            reflect_coroutines = [
                self.reflect_agent.act(
                    model=self.model,
                    state=state,
                    namespace=namespace,
                    request_id=f"idx{idx}-reflect{step}-{hash(state)}-agent{i}",
                    params=self.reflect_params,
                )
                for i, state in enumerate(states)
            ]

            actions = await asyncio.gather(*reflect_coroutines)

            # Choose the best states based on their value
            state_value_pairs = list(zip(state_proposals, values))
            sorted_pairs = sorted(state_value_pairs, key=lambda x: x[1], reverse=True)
            states, values = map(list, zip(*sorted_pairs[:self.num_selections]))

        print(f"idx={idx}, end")

        return states

    async def benchmark(self, benchmark: Benchmark, share_ns: bool=False, cache: bool=True):
        cache = {} if cache else None
        print(f"length of benchmark = {len(benchmark)}")
        for index, state in benchmark:
            print(f"index = {index}")

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


