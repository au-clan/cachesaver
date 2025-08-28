import re
import random
from typing import Tuple

from .state import StateHLE
from ...typedefs import Environment, MAX_SEED
import logging
logger = logging.getLogger(__name__)
import math

OBS_CORRECT = "Answer is CORRECT."
OBS_INCORRECT = "Answer is INCORRECT."

class EnvironmentHLE(Environment):
    """
    Environment for the HLE (Human-Labeled Explanations) task.
    """

    @staticmethod
    def step(state: StateHLE, action: str) -> StateHLE:
        """
        Takes a step in the environment based on the given action.
        """
        
        # Randomness handling
        random.seed(state.randomness)
        randomness = random.randint(0, MAX_SEED)

        # Create new state with updated information
        new_state = StateHLE(
            id=state.id,
            question=state.question,
            image=state.image,
            image_preview=state.image_preview,
            answer=state.answer,
            answer_type=state.answer_type,
            author_name=state.author_name,
            rationale=state.rationale,
            rationale_image=state.rationale_image,
            raw_subject=state.raw_subject,
            category=state.category,
            canary=state.canary,
            steps=state.steps + [action],
            current_state=state.current_state + f"\n{action}",
            step_n=state.step_n + 1,
            values=state.values,
            randomness=randomness
        )
        logger.debug(f"Steps before update: {state.steps}")
        logger.debug(f"Action added: {action}")
        logger.debug(f"New steps: {new_state.steps}")
        return new_state

    @staticmethod
    def is_valid(state: StateHLE, action: str) -> bool:
        """
        Checks if the action taken is valid.
        """
        raise NotImplementedError("Action validation logic is not implemented.")


    @staticmethod
    def is_final(state: StateHLE) -> bool:
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
    def evaluate(state: StateHLE) -> Tuple[bool, float]:
        """
        Evaluates the current state.
        """
        logger.debug(f"Evaluating state with steps: {state.steps}")
        logger.debug(f"Last step: {state.steps[-1] if state.steps else 'No steps'}")
        if not state.steps:
            logger.debug("No steps found - returning (False, 0.0)")
        logger.debug(f"State is - {state}")  
        final = EnvironmentHLE.is_final(state)
        if final:
            score = verify_answer(state.answer, state.steps[-1])
            logger.debug(f"Final state detected. Score: {score}")
            return True, score
        else:
            logger.debug("State is not final.")

            return False, 0.0

#---Helper functions---#
def parse_action(string: str) -> Tuple[str, str]:
    """
    Parses an action string into its type and argument.
    Format: ActionType[argument]
    """
    pattern = r'^(\w+)\[(.+)\]$'
    match = re.match(pattern, string)
    
    if match:
        action_type = match.group(1)
        argument = match.group(2)
        return action_type.lower().capitalize(), argument.strip()
    return None, None

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