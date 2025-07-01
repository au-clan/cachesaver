import re
import math
import random
from typing import Tuple

from .state import StateMathArena
from ...typedefs import Environment, MAX_SEED
import logging
logger = logging.getLogger(__name__)

class EnvironmentMathArena(Environment):
    
    @staticmethod
    def step(state: StateMathArena, action: str) -> StateMathArena:
        
        # Randomness handling
        random.seed(state.randomness)
        randomness = random.randint(0, MAX_SEED)

        new_state = StateMathArena(
            problem=state.problem,
            current_state=state.current_state + f"\n{action}",
            steps=state.steps + [action],
            answer=state.answer,
            step_n=state.step_n + 1,
            values=state.values,
            randomness=randomness
        )
        logger.debug(f"Steps before update: {state.steps}")
        logger.debug(f"Action added: {action}")
        logger.debug(f"New steps: {new_state.steps}")
        return new_state

    @staticmethod
    def is_valid(state: StateMathArena, action: str) -> bool:
        """
        Checks if the action taken is valid.
        """
        raise NotImplementedError("Action validation logic is not implemented.")
    
    @staticmethod
    def is_final(state: StateMathArena) -> bool:

        """
    Checks if the current state is a final state.
    """
        try:
            # Check if the last step contains "The final answer is"
            if len(state.steps) > 0 and "the final answer is" in state.steps[-1].lower():
                return True
            
            # Check if the highest value in `state.values` is >= 0.9
            if len(state.values) > 0 and state.values[max(state.values)] >= 0.1:
                return True
            
            # Optional: Check if the maximum number of steps has been reached
            MAX_STEPS = 10  # Define a reasonable limit for steps
            if state.step_n >= MAX_STEPS:
                return True

            return False
        except Exception as e:
            logger.error(f"Error in is_final: {e}")
            return False
    
    @staticmethod
    def evaluate(state: StateMathArena) -> Tuple[bool, float]:
        """
        Evaluates the current state.
        """
        logger.debug(f"Evaluating state with steps: {state.steps}")
        logger.debug(f"Last step: {state.steps[-1] if state.steps else 'No steps'}")
        if not state.steps:
            logger.debug("No steps found - returning (False, 0.0)")
        logger.debug(f"State is - {state}")  
        final = EnvironmentMathArena.is_final(state)
        if final:
            score = verify_answer(state.answer, state.steps[-1])
            logger.debug(f"Final state detected. Score: {score}")
            return True, score
        else:
            logger.debug("State is not final.")

            return False, 0.0
    

def verify_answer(answer: float, output: str):
    if not output:
        #print(f'The output is empty and cannot match the answer!\n')
        return 0.0

    if 'In summary, ' in output:
        spl_ans = output.split('In summary, ')[-1]
        spl_ans = spl_ans.strip()
    else:
        spl_ans = output.strip()

    try:
        match = re.findall(r'[^^{.\-0123456789]-?[0-9]+\.?[0-9]*[^^}.0123456789]', spl_ans)[-1][1:][:-1]
        model_ans = float(match)

        # standard (adjustable)
        if abs(answer) >= 1:
            result = math.isclose(model_ans, answer, abs_tol=0.1)
        else:
            result = math.isclose(model_ans, answer, rel_tol=0.1)

        #print(f'The ans of model is:{model_ans}, while the ground truth is {answer}.\n')
        return result * 1.0

    except Exception as e:
        try:
            match = re.findall(r'-?[0-9]+\.?[0-9]*', spl_ans)[-1]
            model_ans = float(match)

            # standard (adjustable)
            if abs(answer) >= 1:
                result = math.isclose(model_ans, answer, abs_tol=0.1)
            else:
                result = math.isclose(model_ans, answer, rel_tol=0.1)

            #print(f'The ans of model is:{model_ans}, while the ground truth is {answer}.\n')
            return result * 1.0
        except Exception as e:
            #print(f'Result not matched, error type:{e}\n')
            #print(f'The ans of model is:{spl_ans}, while the ground truth is {answer}.\n')
            return 0.0