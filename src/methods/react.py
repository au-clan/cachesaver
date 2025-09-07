import random
import logging
import asyncio
from typing import TypedDict
from omegaconf import OmegaConf
from ..typedefs import Method, Model, Agent, Environment, DecodingParameters, State, Benchmark, MAX_SEED
from .. import MethodFactory, AgentDictFactory
logger = logging.getLogger(__name__)

@AgentDictFactory.register
class AgentDictReact(TypedDict):
    step: Agent # React Agent
    step_params: DecodingParameters

@MethodFactory.register
class MethodReact(Method):
    def __init__(self,
                 model: Model,
                 agents: AgentDictReact,
                 env: Environment,
                 config: OmegaConf,
                 ):
        super().__init__(model, agents, env, config)
        self.step_agent = agents["step"]

        self.step_params = agents["step_params"]

        self.num_steps = config.num_steps

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
    
    async def benchmark(self, benchmark: Benchmark, ns_ratio: bool=False):

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
                value_cache=None
            )
            for (index, state), ns in zip(benchmark, namespaces)
        ]
        results = await asyncio.gather(*solve_coroutines)
        return results
            
            