import logging
import asyncio
import math

from typing import TypedDict, List, Optional, Tuple

import numpy as np

from ..models import API
from ..tasks.game24 import prompts
# Import core interfaces
from ..typedefs import Algorithm, Agent, Environment, DecodingParameters, State, Benchmark

# Import LLM Reasoners components (ensure llm-reasoners is installed or vendored)
from ..third_party.llm_reasoners_mcts import MCTS, WorldModel, SearchConfig
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
            model: API,
            agents: AgentDictRAP,
            env: Environment,
            num_evaluations: int,
            w_exp: float = 1.,
            depth_limit: int = 5,
            n_iters: int = 5,
            logprobs_model: Optional[API] = None,
    ):
        super().__init__(model, agents, env)
        self.logprobs_model = logprobs_model
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
            eval_agent=self.eval_agent,
            eval_params=self.eval_params,
            env=self.env,
            num_evaluations=self.num_evaluations,
            logprobs_model=self.logprobs_model,
        )
        # MCTS with trace output and mean cumulative reward
        self.search_algo = MCTS(
            output_trace_in_each_iter=True,
            cum_reward=np.mean,
            w_exp=w_exp,
            depth_limit=depth_limit,
            n_iters=n_iters,
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


class RAPWorldModel(WorldModel):
    """
    Adapter: wraps the Game24 Environment for llm-reasoners' WorldModel.
    """
    def __init__(
        self,
        step_agent: Agent,
        step_params: DecodingParameters,
        env: Environment
    ):
        super().__init__()
        self.step_agent = step_agent
        self.step_params = step_params
        self.env = env
        self.init_state_value = None

    def init_state(self) -> State:
        # Directly use the provided state object
        return self.init_state_value

    async def step(self, state: State, action: str) -> Tuple[State, dict]:
        # Apply action using the environment's deterministic step
        next_state = self.env.step(state, action)
        return next_state, {}

    async def is_solved(self, state: State) -> bool:
        try:
            finished, _ = self.env.evaluate(state)
        except Exception:
            return False
        return finished

    def is_terminal(self, state: State) -> bool:
        # Terminal if environment recognizes final state
        return self.env.is_final(state)

    def update_init_state(self, state: State):
        self.init_state_value = state


class RAPSearchConfig(SearchConfig):
    """
    SearchConfig: defines action generation via LLM and reward via Environment.evaluate().
    """
    def __init__(
        self,
        model: API,
        step_agent: Agent,
        eval_agent: Agent,
        step_params: DecodingParameters,
        eval_params: DecodingParameters,
        env: Environment,
        num_evaluations: int,
        logprobs_model: Optional[API] = None,
    ):
        super().__init__()
        self.model = model
        self.step_agent = step_agent
        self.eval_agent = eval_agent
        self.step_params = step_params
        self.logprob_eval_params = eval_params

        self.no_logprob_eval_params = DecodingParameters(
            temperature=eval_params.temperature,
            max_completion_tokens=eval_params.max_completion_tokens,
            top_p=eval_params.top_p,
            stop=eval_params.stop,
            logprobs=False,
            self_eval=eval_params.self_eval
        )

        self.env = env
        self.namespace = None
        self.num_evaluations = num_evaluations
        self.logprobs_model = logprobs_model
        self.step_counter = 0
        self.eval_counter = 0

    async def get_actions(self, state: State) -> List[str]:
        self.step_counter += 1
        # Propose candidate actions via the step agent
        return await self.step_agent.act(
            model=self.model,
            state=state,
            namespace=self.namespace,
            request_id=f"step{self.step_counter}-{hash(state)}",
            params=self.step_params
        )

    async def reward(self, state: State, action: str, **kwargs) -> Tuple[float, dict]:
        # Execute action and evaluate via environment
        next_state = self.env.step(state, action) # the implementation of step in EnvironmentGame24 is not consistent with the base model and only returns state (not a tuple)
        self.eval_counter += 1

        # task specific reward [0.001, 20]
        task_specific_reward = await self.eval_agent.act(
            model=self.model,
            state=next_state,
            namespace=self.namespace,
            n=self.num_evaluations,
            request_id=f"task_specific_eval{self.eval_counter}-{hash(next_state)}",
            params=self.no_logprob_eval_params
        )
        task_specific_reward = task_specific_reward/self.num_evaluations

        # logprobs reward
        logprobs_reward = await self.eval_agent.logprobs_reward(
            logprobs_model=self.logprobs_model,
            state=state,
            action=action,
            next_state=next_state,
            namespace=self.namespace,
            num_evaluations=self.num_evaluations,
            request_id=f"self_eval{self.eval_counter}-{hash(next_state)}",
            eval_params=self.logprob_eval_params)

        # self eval reward
        self_eval_reward = await self.eval_agent.self_eval_reward(
            model=self.model,
            state=state,
            action=action,
            next_state=next_state,
            namespace=self.namespace,
            num_evaluations=self.num_evaluations,
            request_id=f"logprobs_eval{self.eval_counter}-{hash(next_state)}",
            eval_params=self.no_logprob_eval_params)

        return task_specific_reward + logprobs_reward + self_eval_reward, {"task_specific_reward": task_specific_reward, "logprobs_reward": logprobs_reward, "self_eval_reward": self_eval_reward}

    async def fast_reward(self, state: State, action: str) -> Tuple[float, dict]:
        return await self.reward(state, action)

    def update_namespace(self, namespace: str):
        self.namespace = namespace
