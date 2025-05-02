import re
import random
from typing import Tuple
from sympy import simplify

from .state import StateSonnetWriting
from ...typedefs import Environment, MAX_SEED

class EnvironmentSonnetWriting(Environment):

    @staticmethod
    def step(state: StateSonnetWriting, action: str) -> StateSonnetWriting:
        """
        Takes a step in the environment based on the given action
        """

        # Parse the action based on the action type
        if "left" in action:
            current_state = action.split('left: ')[-1].split(')')[0]
        else:
            current_state = action.split(' = ')[-1]

        # Randomness handling
        random.seed(state.randomness)
        randomness = random.randint(0, MAX_SEED)

        state = StateSonnetWriting(
            puzzle=state.puzzle,
            current_state=current_state,
            steps=state.steps + [action],
            randomness=randomness
        )
        return state

    @staticmethod
    def is_valid(state: StateSonnetWriting, action: str) -> bool:
        """
        Checks if the action taken is valid.
        """
        raise NotImplementedError("Action validation logic is not implemented.")

    @staticmethod
    def is_final(state: StateSonnetWriting) -> bool:
        """
        Checks if the current state is a final state.
        """
        raise NotImplementedError("Action validation logic is not implemented.")
