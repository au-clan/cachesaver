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
class AgentDictCOT(TypedDict):
    step: Agent # ActAgent
    step_params: DecodingParameters


@MethodFactory.register
class MethodCOT(Method):
    def __init__(self,
                 model: Model,
                 agents: AgentDictCOT,
                 env: Environment,
                 config: OmegaConf,
                 n: int = 1
                ):
        super().__init__(model, agents, env, config)
        
        self.step_agent = agents["step"]
        self.step_params = agents["step_params"]

        self.n = config.n
        assert self.n == 1, "CoT has only 1 output"

    async def solve(self, idx: int, state: State, namespace: str, value_cache: dict=None):
        randomness = idx
        random.seed(randomness)

        states = [state.clone(randomness=random.randint(0, MAX_SEED)) for _ in range(self.n)]

        action_coroutines = [
            self.step_agent.act(
                model=self.model,
                state=state,
                n=1,
                namespace=namespace,
                request_id=f"idx{idx}-{hash(state)}-agent{i}",
                params=self.step_params
            )
            for i, state in enumerate(states)
        ]
        actions = await asyncio.gather(*action_coroutines)

        # Execute the actions
        states = [self.env.step(state, action[0]) for state, action in zip(states, actions)]
        return states