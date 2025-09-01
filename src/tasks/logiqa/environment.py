from gc import is_finalized
import random
from typing import Tuple

from .state import StateLogiQA
from ... import EnvironmentFactory
from ...typedefs import Environment, MAX_SEED

@EnvironmentFactory.register
class EnvironmentLogiQA(Environment):

    @staticmethod
    def step(state: StateLogiQA, action: str) -> StateLogiQA:
        """
        Takes a step in the environment based on the given action.
        """
        action_taken = get_answer(action)

        random.seed(state.randomness)
        randomness = random.randint(0, MAX_SEED)

        state = StateLogiQA(
            context=state.context,
            question=state.question,
            option_a=state.option_a,
            option_b=state.option_b,
            option_c=state.option_c,
            option_d=state.option_d,
            current_state=action_taken,
            steps=state.steps + [action_taken],
            correct_option=state.correct_option,
            randomness=randomness
        )
        return state
    

    @staticmethod
    def is_valid(state: StateLogiQA, action: str) -> bool:
        """
        Checks if the action taken is valid.
        """
        raise NotImplementedError("Action validation logic is not implemented yet.")
    
    @staticmethod
    def is_final(state: StateLogiQA) -> bool:
        """
        Checks if the current state is a final state.
        """
        current_state = get_answer(state.current_state)

        return current_state in "abcd"
    
    @staticmethod
    def evaluate(state: StateLogiQA) -> Tuple[bool | float]:
        """
        Evaluates the current state.
        """
        if EnvironmentLogiQA().is_final(state):
            answer = state.current_state.strip().lower()
            correct_answer = state.correct_option.strip().lower()
            if answer == correct_answer:
                return True, 1.0
            else:
                return True, 0.0
        else:
            return False, 0.0


# ---Helper functions---
def get_answer(text) -> str:
    valid_options = "abcd"
    action_taken = text.strip().lower()
    if action_taken not in valid_options and len(action_taken) == 1:
        action_taken = valid_options[int(action_taken)-1]
    elif action_taken not in valid_options:
        action_taken = action_taken.replace(".", " ").split(" ")[0].strip()

    return action_taken