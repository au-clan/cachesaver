import random
import asyncio
from typing import TypedDict
from ..typedefs import Algorithm, Model, Agent, Environment, DecodingParameters, State, Benchmark, MAX_SEED
import logging

class AgentDictTOT(TypedDict):
    step: Agent
    evaluate: Agent
    step_params: DecodingParameters
    eval_params: DecodingParameters

class AlgorithmTOT_DFS(Algorithm):
    def __init__(self,
                model: Model,
                agents: AgentDictTOT,
                env: Environment,
                num_selections: int,
                num_steps: int,
                num_evaluations: int,
                convergence_threshold: float,
                pruning_threshold: float,
                confidence_threshold: float,
                max_iterations: int,
                convergence_count: int
                ):
        super().__init__(model, agents, env)
        self.step_agent = agents["step"]
        self.eval_agent = agents["evaluate"]

        self.step_params = agents["step_params"]
        self.eval_params = agents["eval_params"]

        self.num_selections = num_selections
        self.num_steps = num_steps
        self.num_evaluations = num_evaluations

        self.convergence_threshold = convergence_threshold
        self.pruning_threshold = pruning_threshold
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.convergence_count = convergence_count

        """
        max_iterations: Attempts Unique branches
        """

    async def solve(self, idx:int, state: State, namespace: str, value_cache: dict = None):

        ## cachesaver
        randomness = idx
        random.seed(randomness)
        states = [state.clone(randomness=random.randint(0, MAX_SEED))]

        output = []
        # Track the number of iterations and convergence status
        iteration_count = 0
        consecutive_convergence_count = 0
        prev_best_value = None  # To keep track of the best value from the previous iteration

        # Inner recursive DFS function
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
                for state2, actions in zip(s, actions):
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
                #if self.confidence_threshold is not None and best_value >= self.confidence_threshold:
                if self.env.evaluate(best_state) == 1:
                    #print(f"Early stopping: Confidence threshold met. Value = {best_value}")
                    return True

                # Check for convergence (if the value has not improved significantly)
                if prev_best_value is not None and self.convergence_threshold is not None:
                    if abs(best_value - prev_best_value) < self.convergence_threshold:
                        consecutive_convergence_count += 1  # Increase the count if the change is small
                    else:
                        consecutive_convergence_count = 0  # Reset the count if there's a significant change

                prev_best_value = best_value  # Update the best value for the next iteration
                iteration_count += 1  # Increment the iteration count

                # Stop if we've reached the max number of iterations or if convergence criteria are met
                if (self.max_iterations is not None and iteration_count >= self.max_iterations) or \
                        (self.convergence_count is not None and consecutive_convergence_count >= self.convergence_count):
                    #print(f"Early stopping: Max iterations or convergence criteria met.")
                    return True  # Stop if we've reached the max iteration count or convergence

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


            # Evaluate all proposals
            value_coroutines = [
                self.eval_agent.act(
                    model=self.model,
                    state=state,
                    n=self.num_evaluations,
                    namespace=namespace,
                    request_id=f"idx{idx}-evaluation{t}-{hash(state)}-agent{i}",
                    params=self.eval_params,
                    cache=value_cache
                )
                for i, state in enumerate(state_proposals)
            ]
            values = await asyncio.gather(*value_coroutines)
            # Print the results
            state_value_pairs = list(zip(state_proposals, values))
            sorted_pairs = sorted(state_value_pairs, key=lambda x: x[1], reverse=True)


            for state2, value in sorted_pairs:
                if t == 1:
                    # Apply pruning only at depth 1
                    if value > self.pruning_threshold:
                        if await dfs([state2], t + 1):  # Go one step deeper in the DFS search
                            return True  # Stop and return if a solution is found
                else:
                    # No pruning at other depths
                    if await dfs([state2], t + 1):  # Go one step deeper in the DFS search
                        return True  # Stop and return if a solution is found

            return False  # Continue searching if no solution was found in this branch

        # Start the DFS from the initial state x, at depth 1
        await dfs(states, 1)

        # Return the best result found, or None if no result was found
        output = sorted(output, key=lambda x: x[1], reverse=True)[:self.num_selections] if output else []
        return [x[0] for x in output]  # Return only the states, not the values


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