from typing import List

from . import prompts_game24 as prompts
from .state_game24 import StateGame24, HeuristicGame24
from ...typedefs import Agent, Model, DecodingParameters

class AgentGame24(Agent):
    def __init__(self, model: Model):
        super().__init__()
        self.name = "Game24 Agent"

    @staticmethod
    async def bfs(model: Model, state: StateGame24, namespace: str, request_id: str, params: DecodingParameters) -> List[str]:
        """
        Returns a list of actions for the BFS algorithm.
        """
        
        # Format the prompt
        if state.current_state == "24":
            prompt = prompts.cot.format(input=state.puzzle) + "\nSteps:\n" + '\n'.join(state.steps) + "\nAnswer: "
        else:
            current_numbers = HeuristicGame24.get_current_numbers(state)
            prompt = prompts.bfs.format(input=current_numbers)

        # Generate the response
        response = await model.request(
            prompt=prompt,
            n=1,
            request_id=request_id,
            namespace=namespace,
            params=params
        )

        # Parse the response
        proposals = response[0].split("\n")
        return proposals
    
    @staticmethod
    async def evaluate(model: Model, state: StateGame24, n: int,namespace: str, request_id: str, params: DecodingParameters, cache: dict=None) -> List[float]:
        """
        Returns a value for the given state
        """

        # Check if the state is already in the cache
        if cache is not None and state.current_state in cache:
            return cache[state.current_state]

        # Format the prompt
        if "left" not in state.steps[-1]:
            formula = HeuristicGame24.get_formula(state)
            prompt = prompts.evaluate_answer.format(input=state.puzzle, answer=formula)
        else:
            prompt = prompts.evaluate.format(input=state.current_state)
        
        # Generate the responses
        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params
        )

        # Parse the response
        codes = [r.split('\n')[-1].lower() for r in responses]
        code_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}
        value = sum(value * codes.count(code) for code, value in code_map.items())

        # Cache the value
        if cache is not None:
            cache[state.current_state] = value
        return value



