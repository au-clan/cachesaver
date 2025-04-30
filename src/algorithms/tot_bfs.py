import random
import logging
import asyncio
from typing import TypedDict
from ..typedefs import Algorithm, Model, Agent, Environment, DecodingParameters, State, Benchmark, MAX_SEED
logger = logging.getLogger(__name__)

class AgentDictTOT(TypedDict):
    step: Agent
    evaluate: Agent
    step_params: DecodingParameters
    eval_params: DecodingParameters

class AlgorithmTOT(Algorithm):
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
        self.eval_agent = agents["evaluate"]

        self.step_params = agents["step_params"]
        self.eval_params = agents["eval_params"]

        self.num_selections = num_selections
        self.num_steps = num_steps
        self.num_evaluations = num_evaluations

    async def solve(self, idx:int, state: State, namespace: str, value_cache: dict = None):
        
        randomness = idx
        random.seed(randomness)
        states = [state.clone(randomness=random.randint(0, MAX_SEED))]

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

            # Choose the best states based on their value
            state_value_pairs = list(zip(state_proposals, values))
            sorted_pairs = sorted(state_value_pairs, key=lambda x: x[1], reverse=True)
            states, values = map(list, zip(*sorted_pairs[:self.num_selections]))
    
        return states
        
    async def solve_dfs(self, idx:int, state: State, namespace: str,convergence_threshold : int =0.6,pruning_threshold : int = 0,confidence_threshold: int = 20, max_iterations :int = 10 ,convergence_count : int = 5, value_cache: dict = None):

        randomness = idx
        random.seed(randomness)
        states = [state.clone(randomness=random.randint(0, MAX_SEED))]
        output = []
        iteration_count = 0
        consecutive_convergence_count = 0
        prev_best_value = None  # To keep track of the best value from the previous iteration

        async def dfs(s, t):
            nonlocal consecutive_convergence_count, prev_best_value, iteration_count

            if t >= self.num_steps:
                action_coroutines = [

                    self.step_agent.act(
                        model=self.model,
                        state=state,
                        namespace=namespace,
                        request_id=f"idx{idx}-step{t}-{hash(state)}",
                        params=self.step_params,
                    )
                    for state in s
                ]
                actions = await asyncio.gather(*action_coroutines)

                state_proposals = []
                for state2, actions in zip(states, actions):
                    for action in actions:
                        next_state = self.env.step(state2, action)
                        state_proposals.append(next_state)

                value_coroutines = [
                    self.eval_agent.act(
                        model=self.model,
                        state=state,
                        n=self.num_evaluations,
                        namespace=namespace,
                        request_id=f"idx{idx}-evaluation{t}-{hash(state)}",
                        params=self.eval_params,
                        cache=value_cache
                    )
                    for state in state_proposals
                ]
                values = await asyncio.gather(*value_coroutines)
                state_value_pairs = zip(state_proposals, values)
                best_state, best_value = max(state_value_pairs, key=lambda x: x[1])
                # Store only the best thought and its evaluation value
                output.append((best_state, best_value))

                # Early stopping if confidence is high
                if confidence_threshold is not None and best_value >= confidence_threshold:
                    print(f"Early stopping: Confidence threshold met. Value = {best_value}")
                    return True

                # Check for convergence (if the value has not improved significantly)
                if prev_best_value is not None and convergence_threshold is not None:
                    if abs(best_value - prev_best_value) < convergence_threshold:
                        consecutive_convergence_count += 1  # Increase the count if the change is small
                    else:
                        consecutive_convergence_count = 0  # Reset the count if there's a significant change

                prev_best_value = best_value  # Update the best value for the next iteration
                iteration_count += 1  # Increment the iteration count

                return False

            action_coroutines = [
                self.step_agent.act(
                    model=self.model,
                    state=state,
                    namespace=namespace,
                    request_id=f"idx{idx}-step{t}-{hash(state)}",
                    params=self.step_params,
                )
                for state in s
            ]
            actions = await asyncio.gather(*action_coroutines)

            state_proposals = []
            for state2, actions in zip(states, actions):
                print(f"Current state: {state2}")  

                for action in actions:
                    next_state = self.env.step(state2, action)
                    print(f"  Action: {action} → Next state: {next_state}")  
                    state_proposals.append(next_state)


            # Evaluate all proposals
            value_coroutines = [
                self.eval_agent.act(
                    model=self.model,
                    state=state,
                    n=self.num_evaluations,
                    namespace=namespace,
                    request_id=f"idx{idx}-evaluation{t}-{hash(state)}",
                    params=self.eval_params,
                    cache=value_cache
                )
                for state in state_proposals
            ]
            values = await asyncio.gather(*value_coroutines)
           
            state_value_pairs = list(zip(state_proposals, values))
            sorted_pairs = sorted(state_value_pairs, key=lambda x: x[1], reverse=True)

            for state2, value in sorted_pairs:
                if value > pruning_threshold:
                    print(f" State: {state2}, Value: {value}")
                    if await dfs([state2], t + 1):  # Go one step deeper in the DFS search
                        return True  # Stop and return if a solution is found

            return False  # Continue searching if no solution was found in this branch

        # Start the DFS from the initial state x, at depth 1
        await dfs(states, 1)

        # Return the best result found, or None if no result was found
        return sorted(output, key=lambda x: x[1], reverse=True)[:5] if output else None

    async def benchmark(self, benchmark: Benchmark, share_ns: bool=False, cache: bool=True):
        cache = {} if cache else None
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


