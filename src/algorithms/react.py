import random
import logging
import asyncio
from typing import TypedDict
from ..typedefs import Algorithm, Model, Agent, Environment, DecodingParameters, State, Benchmark, MAX_SEED
logger = logging.getLogger(__name__)

class AgentDictReact(TypedDict):
    step: Agent # React Agent
    step_params: DecodingParameters

class AlgorithmReact(Algorithm):
    def __init__(self,
                 model: Model,
                 agents: AgentDictReact,
                 env: Environment,
                 num_steps: int
                 ):
        super().__init__(model, agents, env)
        self.step_agent = agents["step"]

        self.step_params = agents["step_params"]

        self.num_steps = num_steps

    async def solve(self, idx: int, state: State, namespace: str, value_cache: dict = None):
        randomness = idx
        random.seed(randomness)
        state = state.clone(randomness=random.randint(0, MAX_SEED))
        
        for step in range(self.num_steps):
            print(f"Step {step} ({idx})")

            # Generate action using the step agent
            action = await self.step_agent.act(
                model=self.model,
                state=state,
                n=1, 
                namespace=namespace,
                request_id=f"idx{idx}-step{step}-{hash(state)}",
                params = self.step_params)
            
            # Execute the action
            state = self.env.step(state, action[0])

            if self.env.evaluate(state)[1] == 1:
                break
        return [state]
    
    async def benchmark(self, benchmark: Benchmark, share_ns: bool=False, cache: bool=True):
        solve_coroutines = [
            self.solve(
                idx=index,
                state=state,
                namespace="benchmark" if share_ns else f"benchmark-{index}",
                value_cache=None
            )
            for index, state in benchmark
        ]
        results = await asyncio.gather(*solve_coroutines)
        return results
            
            