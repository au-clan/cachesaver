import random
from typing import Tuple

from .state import StateLogiQA
from ...typedefs import Environment, MAX_SEED

class EnvironmentLogiQA(Environment):

    @staticmethod
    def step(state: StateLogiQA, action: str) -> StateLogiQA:
        """
        Takes a step in the environment based on the given action.
        """
        valid_options = "abcd"
        action_taken = action.strip().lower()
        if action_taken not in valid_options:
            try:
                action_taken = valid_options[int(action_taken)-1]
            except: # If the whole option from state is passed as last option.
                action_taken = action_taken[0].strip().lower()

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
        raise NotImplementedError("is_final is not implemented yet.")
    
    @staticmethod
    def evaluate(state: StateLogiQA) -> Tuple[bool | float]:
        """
        Evaluates the current state.
        """
        answer = state.current_state.strip().lower()
        correct_answer = state.correct_option.strip().lower()
        if answer == correct_answer:
            return True, 1.0
        else:
            return False, 0.0
