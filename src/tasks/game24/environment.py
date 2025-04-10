import re
import random
from typing import Tuple
from sympy import simplify

from .state import StateGame24
from ...typedefs import Environment, MAX_SEED

class EnvironmentGame24(Environment):

    @staticmethod
    def step(state: StateGame24, action: str) -> StateGame24:
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

        state = StateGame24(
            puzzle=state.puzzle,
            current_state=current_state,
            steps=state.steps + [action],
            randomness=randomness
        )
        return state
    
    @staticmethod
    def is_valid(state: StateGame24, action: str) -> bool:
        """
        Checks if the action taken is valid.
        """
        raise NotImplementedError("Action validation logic is not implemented.")
    
    @staticmethod
    def is_final(state: StateGame24) -> bool:
        """
        Checks if the current state is a final state.
        """
        expression = state.steps[-1]
        if "left" in expression and len(state.current_state.split(' '))>1:
            return False
        else:
            return True

    @staticmethod
    def evaluate(state: StateGame24) -> Tuple[bool, float]:
        """
        Evaluates the current state.
        """
        is_final = EnvironmentGame24.is_final(state)
        if is_final is True:
            expression = state.steps[-1].lower().replace('answer: ', '').split('=')[0]
            numbers = re.findall(r'\d+', expression)
            problem_numbers = re.findall(r'\d+', state.puzzle)
            if sorted(numbers) != sorted(problem_numbers):
                return True, 0.0
            else:
                correct = simplify(expression) == 24
                return True, float(correct)

        else:
            return False, 0.0


            
