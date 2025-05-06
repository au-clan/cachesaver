import logging
import asyncio

from typing import TypedDict, List

import numpy as np

from ..models.rap import RAPWorldModel, RAPSearchConfig
# Import core interfaces
from ..typedefs import Algorithm, Model, Agent, Environment, DecodingParameters, State, Benchmark

# Import LLM Reasoners components (ensure llm-reasoners is installed or vendored)
from ..third_party.llm_reasoners_mcts import MCTS
from ..third_party.llm_reasoners_mcts import MCTSResult

logger = logging.getLogger(__name__)


class AgentDictRAP(TypedDict):
    step: Agent
    evaluate: Agent
    step_params: DecodingParameters
    eval_params: DecodingParameters


class AlgorithmRAP(Algorithm):
    """
    RAP (Reasoning via Planning) implementation using Monte-Carlo Tree Search from llm-reasoners.
    """

    def __init__(
            self,
            model: Model,
            agents: AgentDictRAP,
            env: Environment,
            num_evaluations: int
    ):
        super().__init__(model, agents, env)

        # Core agents and parameters
        self.step_agent = agents['step']
        self.eval_agent = agents['evaluate']
        self.step_params = agents['step_params']
        self.eval_params = agents['eval_params']
        self.num_evaluations = num_evaluations

        # World model adapter
        self.world_model: RAPWorldModel[State, str] = RAPWorldModel(
            step_agent=self.step_agent,
            step_params=self.step_params,
            env=self.env
        )
        # Search config adapter
        self.search_config: RAPSearchConfig[State, str] = RAPSearchConfig(
            model=self.model,
            step_agent=self.step_agent,
            step_params=self.step_params,
            env=self.env,
            num_evaluations=self.num_evaluations,
        )
        # MCTS with trace output and mean cumulative reward
        self.search_algo = MCTS(
            output_trace_in_each_iter=True,
            cum_reward=np.mean,
        )

    async def solve(
            self,
            idx: int,
            state: State,
            namespace: str,
            value_cache: dict = None,
    ) -> List[State]:
        """
        Perform MCTS search from the given state and
        return the sequence of states along the best reasoning trace.
        """
        # Execute MCTS using the callable interface
        # __call__ signature: (world_model, search_config, log_file=None, **kwargs)
        self.world_model.update_init_state(state)
        self.search_config.update_namespace(namespace)
        mcts_result: MCTSResult = await self.search_algo(
            self.world_model,
            self.search_config
        )
        # If no trace, return empty plan
        if mcts_result.trace is None:
            return []
        # trace is tuple: (states: List[State], actions: List[str])
        states_seq, _ = mcts_result.trace
        # Build state sequence, skipping the initial state
        result_states: List[State] = []
        for st in states_seq[1:]:
            result_states.append(st)
            if self.env.is_final(st):
                break
        return result_states

    async def benchmark(
            self,
            benchmark: Benchmark,
            share_ns: bool = False,
            cache: bool = True
    ) -> List[List[State]]:
        cache = {} if cache else None
        tasks = [
            self.solve(
                idx=i,
                state=s,
                namespace="benchmark" if share_ns else f"benchmark_{i}",
                value_cache=cache
            )
            for i, s in benchmark
        ]
        return await asyncio.gather(*tasks)
