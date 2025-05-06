from typing import List, Tuple

# Import core interfaces
from ..typedefs import Model, Agent, Environment, DecodingParameters, State

# Import LLM Reasoners components (ensure llm-reasoners is installed or vendored)
from ..third_party.llm_reasoners_mcts import WorldModel
from ..third_party.llm_reasoners_mcts import SearchConfig


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
        model: Model,
        step_agent: Agent,
        step_params: DecodingParameters,
        env: Environment,
        num_evaluations: int
    ):
        super().__init__()
        self.model = model
        self.step_agent = step_agent
        self.step_params = step_params
        self.env = env
        self.namespace = None
        self.num_evaluations = num_evaluations
        self.step_counter = 0

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
        finished, score = self.env.evaluate(next_state)
        return score, {"finished": finished}

    async def fast_reward(self, state: State, action: str) -> Tuple[float, dict]:
        # the same as normal reward for now, Execute action and evaluate via environment
        next_state = self.env.step(state, action) # the implementation of step in EnvironmentGame24 is not consistent with the base model and only returns state (not a tuple)
        finished, score = self.env.evaluate(next_state)
        return score, {"finished": finished}

    def update_namespace(self, namespace: str):
        self.namespace = namespace
