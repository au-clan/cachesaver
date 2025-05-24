import random
import asyncio
import logging
from tqdm.asyncio import tqdm
from typing import TypedDict
from ..typedefs import Algorithm, Model, Agent, Environment, DecodingParameters, State, Benchmark, MAX_SEED
logger = logging.getLogger(__name__)

class AgentDictGOT(TypedDict):
    step: Agent
    aggregate: Agent
    evaluate: Agent
    step_params: DecodingParameters
    aggregate_params: DecodingParameters
    eval_params: DecodingParameters

class AlgorithmGOT(Algorithm):
    def __init__(self, 
                 model: Model, 
                 agents: AgentDictGOT, 
                 env: Environment, 
                 num_selections: int, 
                 num_steps: int,
                 num_generate: int,
                 num_best: int,
                 num_evaluations: int,
                 ):
        super().__init__(model, agents, env)

        self.step_agent = agents["step"]
        self.aggregate_agent = agents["aggregate"]
        self.eval_agent = agents["evaluate"]

        self.step_params = agents["step_params"]
        self.aggregate_params = agents["aggregate_params"]
        self.eval_params = agents["eval_params"]

        self.num_selections = num_selections
        self.num_steps = num_steps
        self.num_generate = num_generate
        self.num_best = num_best
        self.num_evaluations = num_evaluations

    async def solve(self, idx: int, state: State, namespace: str, value_cache: dict = None):
        randomness = idx
        random.seed(randomness)
        states = [state.clone(randomness=random.randint(0, MAX_SEED))]
        logger.debug(f"Solving game: {idx}")

        solved = False
        for step in range(self.num_steps):
            if solved:
                logger.debug(f"Task {idx} solved at step {step - 1}.")
                break

            logger.debug(f"Step: {step} ({idx})")
            # Generate actions for each state
            action_coroutines = [
                self.step_agent.act(
                    model=self.model,
                    state=state,
                    n=self.num_generate,
                    namespace=namespace,
                    request_id=f"idx{idx}-step{step}-{hash(state)}-agent{i}",
                    params=self.step_params,
                )
                for i, state in enumerate(states)
            ]
            generated_actions = await asyncio.gather(*action_coroutines)
            logger.debug(f"{len(generated_actions)} Actions generated for task {idx}; \n {generated_actions}")

            # Aggregate actions
            aggregate_coroutines = [
                self.aggregate_agent.act(
                    model=self.model,
                    state=state,
                    actions=action,
                    k=self.num_selections,
                    namespace=namespace,
                    request_id=f"idx{idx}-aggregate{step}-{hash(state)}-agent{i}",
                    params=self.aggregate_params,
                )
                for i, (state, action) in enumerate(zip(states, generated_actions))
            ]

            actions = await asyncio.gather(*aggregate_coroutines)
            logger.debug(f"{len(actions)} Actions selected for task {idx}: \n{actions}")

            # Execute actions on environment
            proposed_states = []
            for state, actions in zip(states, actions):
                for action in actions:
                    proposed_states.append(self.env.step(state, action))
            
            if proposed_states == []:
                return states
            
            # Early stop in case any state is solved
            if any(self.env.evaluate(state)[1] == 1 for state in states):
                solved = True
            
            logger.debug(f"Env step for task {idx}: \n{proposed_states}")
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
                for i, state in enumerate(proposed_states)
            ]
            values = await asyncio.gather(*value_coroutines)
            logger.debug(f"Values given for task {idx}: \n{values}")

            # Choose the best states based on their value
            state_value_pairs = list(zip(proposed_states, values))
            sorted_pairs = sorted(state_value_pairs, key=lambda x: x[1], reverse=True)
            states, values = map(list, zip(*sorted_pairs[:self.num_best]))
        
        return states

            
    
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
        results = await tqdm.gather(*solve_coroutines)
        return results

