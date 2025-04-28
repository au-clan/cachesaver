import re
import random
from typing import Tuple

from .state import StateHumanEval
from ...typedefs import Environment, MAX_SEED

class EnvironmentHumanEval(Environment):

    @staticmethod
    def step(state: StateHumanEval, action: str) -> StateHumanEval:
        """
        Takes a step in the environment based on the given action.
        """

        # Parse the action
        completion = parse_action(action)

        # randomness handling
        random.seed(state.randomness)
        randomness = random.randint(0, MAX_SEED)

        state = StateHumanEval(
            puzzle=state.puzzle,
            current_state=completion,
            steps=state.steps + [action],
            canonical_solution=state.canonical_solution,
            entry_point=state.entry_point,
            test=state.test,
            randomness=randomness
        )
        return state

    @staticmethod
    def is_valid(state: StateHumanEval, action: str) -> bool:
        """
        Checks if the action taken is valid.
        """
        raise NotImplementedError("Action validation logic is not implemented.")
    
    @staticmethod
    def is_final(state: StateHumanEval) -> bool:
        """
        Checks if the current state is a final state.
        """
        raise NotImplementedError("Final state logic is not implemented.")
    
    @staticmethod
    def evaluate(state: StateHumanEval) -> Tuple[bool, float]:
        """
        Evaluates the current state.
        """
        raise NotImplementedError("Evaluation logic is not implemented.")


#---Helper functions---#
def parse_action(string) -> str | None:
    pattern = r'```[^`]+```'
    match = re.match(pattern, string)
    return match.group(0) if match else None