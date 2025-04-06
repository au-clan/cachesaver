import random
import asyncio
from ..typedefs import Algorithm, Model, Agent, Heuristic, Environment, DecodingParameters, State
from ..utils import Resampler


class AlgorithmFOA(Algorithm):
    def __init__(self, 
                 model: Model, 
                 agent: Agent, 
                 env: Environment, 
                 heuristic: Heuristic,
                 step_params: DecodingParameters,
                 eval_params: DecodingParameters,
                 num_agents: int, 
                 num_steps: int, 
                 k: int, 
                 backtrack: float, 
                 resampling: str, 
                 origin: float, 
                 min_steps: int):
        super().__init__(model, agent, env, heuristic)

        self.step_params = step_params
        self.eval_params = eval_params

        self.num_agents = num_agents
        self.num_steps = num_steps
        self.k = k
        self.backtrack = backtrack
        self.resampling = resampling
        self.origin = origin
        self.min_steps = min_steps


    async def run(self, idx: int, state: State, namespace: str, value_cache: dict = None):
        randomness = 0
        random.seed(randomness)
        resampler = Resampler(randomness)

        # Records of previously visited states (state_identifier, state_value, state)
        visited_states = [("INIT", self.origin, state)]

        # Initialize state for each agent
        states = [state.clone(randomness=random.randint(0, 1000)) for _ in range(self.num_agents)]

        solved = False
        for step in range(self.num_steps):
            print(f"Step {step}")

            if solved:
                break
            
            # Generate actions for each state and apply them
            action_coroutines = [
                self.agent.act(
                    model=self.model,
                    state=state,
                    n=1,
                    namespace=namespace,
                    request_id=f"idx{idx}-step{step}-{hash(state)}",
                    params=self.step_params
                )
                for state in states
            ]
            actions = await asyncio.gather(*action_coroutines)
            states = [self.environment.step(state, action) for state, action in zip(states, actions)]

            # Early stop in case any state is solved
            if any(self.environment.evaluate(state)[1] for state in states):
                solved = True
                break

            # Filter previously visited states records
            remaining_steps = self.num_steps - (step + 1)
            visited_states = [(identifier, value*self.backtrack, state) for identifier, value, state in visited_states]
            visited_states = [state for state in visited_states if  remaining_steps >= self.min_steps - len(state[2].steps)]

            # Pruning : Failed = Finished not correctly
            failed = [i for i, state in enumerate(states) if self.environment.is_final(state)]
            if visited_states != []:
                replacements, _ = resampler.resample(visited_states.copy(), len(failed), self.config.framework.resampling)
            else:
                replacements, _ = resampler.resample([("", 1, state) for state in states], len(failed), resampling_method="linear")
            states = [replacements.pop(0) if i in failed else state for i, state in enumerate(states)]

            # Evaluation phase
            if step < self.num_steps-1 and step % self.k == 0:
                
                # Evaluate the states
                value_coroutines = [
                    self.agent.evaluate(
                        model=self.model,
                        state=state,
                        n=1,
                        namespace=namespace,
                        request_id=f"idx{idx}-evaluation{step}-{hash(state)}",
                        params=self.eval_params,
                        cache=value_cache
                    )
                    for state in states
                ]
                values = await asyncio.gather(*value_coroutines)

                # Update previously visited states records
                for idx, (state, value) in enumerate(zip(states, values)):
                    if idx not in failed:
                        visited_states.append((f"{idx}.{step}", value, state))

                # Resampling
                states, resampled_idxs = resampler.resample(visited_states, self.num_agents, self.resampling)
        
        # Final evaluation -> Bettter to be called verifications
        evaluations = [self.environment.evaluate(state) for state in states]

        return evaluations



