from typing import List, Tuple

# Import core interfaces
from ..typedefs import Model, Agent, Environment, DecodingParameters, State

# Import LLM Reasoners components (ensure llm-reasoners is installed or vendored)
from ..third_party.llm_reasoners.reasoners import WorldModel
from ..third_party.llm_reasoners.reasoners import SearchConfig


class RAPWorldModel(WorldModel[State, str]):
    """
    Adapter: wraps the Game24 Environment for llm-reasoners' WorldModel.
    """
    def __init__(
        self,
        model: Model,
        step_agent: Agent,
        step_params: DecodingParameters,
        env: Environment
    ):
        super().__init__(model=model)
        self.step_agent = step_agent
        self.step_params = step_params
        self.env = env

    def init_state(self, state: State) -> State:
        # Directly use the provided state object
        return state

    async def step(self, state: State, action: str) -> Tuple[State, dict]:
        # Apply action using the environment's deterministic step
        next_state, _ = self.env.step(state, action)
        return next_state, {}

    def is_terminal(self, state: State) -> bool:
        # Terminal if environment recognizes final state
        return self.env.is_final(state)


class RAPSearchConfig(SearchConfig[State, str]):
    """
    SearchConfig: defines action generation via LLM and reward via Environment.evaluate().
    """
    def __init__(
        self,
        model: Model,
        step_agent: Agent,
        step_params: DecodingParameters,
        env: Environment
    ):
        super().__init__()
        self.model = model
        self.step_agent = step_agent
        self.step_params = step_params
        self.env = env

    async def get_actions(self, state: State) -> List[str]:
        # Propose candidate actions via the step agent
        return await self.step_agent.act(
            model=self.model,
            state=state,
            n=self.step_params.n,
            namespace="rap",
            request_id="",
            params=self.step_params
        )

    async def reward(self, state: State, action: str, **kwargs) -> Tuple[float, dict]:
        # Execute action and evaluate via environment
        next_state, _ = self.env.step(state, action)
        finished, score = self.env.evaluate(next_state)
        return score, {"finished": finished}

    def fast_reward(self, state: State, action: str) -> Tuple[float, dict]:
        # the same as normal reward for now, Execute action and evaluate via environment
        next_state, _ = self.env.step(state, action)
        finished, score = self.env.evaluate(next_state)
        return score, {"finished": finished}
