import re
import random
from typing import Tuple
from sympy import simplify

from .state import StateLogiQA
from ...typedefs import Environment, MAX_SEED

class EnvironmentLogiQA(Environment):

    @staticmethod
    def step(state: StateLogiQA, action: str) -> StateLogiQA:
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

        state = StateLogiQA(
            puzzle=state.puzzle,
            current_state=current_state,
            steps=state.steps + [action],
            randomness=randomness
        )
        return state

    @staticmethod
    def is_valid(state: StateLogiQA, action: str) -> bool:
        """
        Checks if the action taken is valid.
        """
        raise NotImplementedError("Action validation logic is not implemented.")

    @staticmethod
    def is_final(state: StateLogiQA) -> bool:
        """
        Checks if the current state is a final state.
        """
        raise NotImplementedError("Action validation logic is not implemented.")
