import random
import asyncio
from typing import TypedDict
from omegaconf import OmegaConf
from ..typedefs import Method, Model, Agent, Environment, DecodingParameters, State, Benchmark, MAX_SEED
from .. import MethodFactory, AgentDictFactory
import logging

@AgentDictFactory.register
class AgentDictTOT(TypedDict):
    step: Agent
    evaluate: Agent
    step_params: DecodingParameters
    evaluate_params: DecodingParameters

@MethodFactory.register
class MethodTOT_DFS(Method):
    def __init__(self,
                model: Model,
                agents: AgentDictTOT,
                env: Environment,
                config: OmegaConf
                ):
        super().__init__(model, agents, env, config)
        self.step_agent = agents["step"]
        self.eval_agent = agents["evaluate"]

        self.step_params = agents["step_params"]
        self.evaluate_params = agents["evaluate_params"]

        self.num_selections = config.num_selections
        self.num_steps = config.num_steps
        self.num_evaluations = config.num_evaluations

        self.pruning_threshold = config.pruning_threshold or None
        self.max_iterations = config.max_iterations or None

        """
        max_iterations: Attempts Unique branches
        """

    async def solve(self, idx:int, state: State, namespace: str, value_cache: dict = None):

        ## cachesaver
        randomness = idx
        random.seed(randomness)
        states = [state.clone(randomness=random.randint(0, MAX_SEED))]

        output = []
        iteration_count = 0

        # Inner recursive DFS function
        async def dfs(s, t):
            nonlocal iteration_count

            if t >= self.num_steps:
                iteration_count += 1
                
                action_coroutines = [

                    self.step_agent.act(
                        model=self.model,
                        state=state,
                        namespace=namespace,
                        request_id=f"idx{idx}-step{t}-{hash(state)}-agent{i}",
                        params=self.step_params,
                    )
                    for i, state in enumerate(s)
                ]

                actions = await asyncio.gather(*action_coroutines)

                state_proposals = []
                for state2, actions in zip(s, actions):
                    for action in actions:
                        next_state = self.env.step(state2, action)
                        state_proposals.append(next_state)
                
                if state_proposals == []:
                    return False

                value_coroutines = [
                    self.eval_agent.act(
                        model=self.model,
                        state=state,
                        n=self.num_evaluations,
                        namespace=namespace,
                        request_id=f"idx{idx}-evaluation{t}-{hash(state)}-agent{i}",
                        params=self.evaluate_params,
                        cache=value_cache
                    )
                    for i, state in enumerate(state_proposals)
                ]

                values = await asyncio.gather(*value_coroutines)
                state_value_pairs = list(zip(state_proposals, values))

                if not state_value_pairs:
                    return False
                
                best_state, best_value = max(state_value_pairs, key=lambda x: x[1])
                output.append((best_state, best_value))

                if self.env.evaluate(best_state)[1] == 1: ## early stopping
                    return True  # returning a list with best_state for consistency

                if (self.max_iterations is not None and iteration_count >= self.max_iterations):
                    print(f"Early stopping: Max iterations reached")
                    return True
                return False
                



            action_coroutines = [
                self.step_agent.act(
                    model=self.model,
                    state=state,
                    namespace=namespace,
                    request_id=f"idx{idx}-step{t}-{hash(state)}-agent{i}",
                    params=self.step_params,
                )
                for i, state in enumerate(s)
            ]
            actions = await asyncio.gather(*action_coroutines)

            state_proposals = []
            for state2, actions in zip(s, actions):
                for action in actions:
                    next_state = self.env.step(state2, action)
                    state_proposals.append(next_state)
            
            if state_proposals == []:
                return False


            # Evaluate all proposals
            value_coroutines = [
                self.eval_agent.act(
                    model=self.model,
                    state=state,
                    n=self.num_evaluations,
                    namespace=namespace,
                    request_id=f"idx{idx}-evaluation{t}-{hash(state)}-agent{i}",
                    params=self.evaluate_params,
                    cache=value_cache
                )
                for i, state in enumerate(state_proposals)
            ]
            values = await asyncio.gather(*value_coroutines)

            state_value_pairs = list(zip(state_proposals, values))
            sorted_pairs = sorted(state_value_pairs, key=lambda x: x[1], reverse=True)

            for state2, value in sorted_pairs[:self.num_selections]:
                if self.pruning_threshold is not None and value > self.pruning_threshold:
                    if await dfs([state2], t + 1): # Go one step deeper in the DFS search
                        return True
            return False
        
        try:
            result = await dfs(states, 1)
        except Exception as e:
            print(f"Error during initial dfs call: {e}")

        output = sorted(output, key=lambda x: x[1], reverse=True)[:self.num_selections] if output else [states]
        return [x[0] for x in output]
        
        
