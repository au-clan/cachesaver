import random
import logging
import asyncio
from typing import TypedDict
from omegaconf import OmegaConf
from ..typedefs import Method, Model, Agent, Environment, DecodingParameters, State, Benchmark, MAX_SEED
from .. import MethodFactory, AgentDictFactory
logger = logging.getLogger(__name__)

@AgentDictFactory.register
class AgentDictTOT(TypedDict):
    step: Agent
    evaluate: Agent
    step_params: DecodingParameters
    evaluate_params: DecodingParameters

@MethodFactory.register
class MethodTOT_BFS(Method):
    def __init__(self,
                model: Model,
                agents: AgentDictTOT,
                env: Environment,
                config: OmegaConf,
                ):
        super().__init__(model, agents, env, config)

        self.step_agent = agents["step"]
        self.eval_agent = agents["evaluate"]

        self.step_params = agents["step_params"]
        self.evaluate_params = agents["evaluate_params"]

        self.num_selections = config.num_selections
        self.num_steps = config.num_steps
        self.num_evaluations = config.num_evaluations

    async def solve(self, idx:int, state: State, namespace: str, value_cache: dict = None):
        
        randomness = idx
        random.seed(randomness)
        states = [state.clone(randomness=random.randint(0, MAX_SEED))]

        for step in range(self.num_steps):

            print(f"Step {step} ({idx})")

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
            for state, actions in zip(states, actions): # Bad practice
                for action in actions:
                    state_proposals.append(self.env.step(state, action))

            if state_proposals == []:
                return states

            # Evaluate all proposals
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
                for i, state in enumerate(state_proposals)
            ]
            values = await asyncio.gather(*value_coroutines)

            # Choose the best states based on their value
            state_value_pairs = list(zip(state_proposals, values))
            sorted_pairs = sorted(state_value_pairs, key=lambda x: x[1], reverse=True)
            states, values = map(list, zip(*sorted_pairs[:self.num_selections]))
            
            # Early stopping condition
            for state in states:
                if self.env.evaluate(state)[1]==1:
                    return states

        return states