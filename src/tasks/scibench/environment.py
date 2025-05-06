import re
import math
import random
from typing import Tuple

from .state import StateSciBench
from ...typedefs import Environment, MAX_SEED

class EnvironmentSciBench(Environment):
    
    @staticmethod
    def step(state: StateSciBench, action: str) -> StateSciBench:
        
        # Randomness handling
        random.seed(state.randomness)
        randomness = random.randint(0, MAX_SEED)

        state = StateSciBench(
            puzzle=state.puzzle,
            current_state=state.current_state + action,
            steps=state.steps + [action],
            answer=state.answer,
            step_n=state.step_n + 1,
            values=state.values,
            randomness=randomness
        )
        return state

    @staticmethod
    def is_valid(state: StateSciBench, action: str) -> bool:
        """
        Checks if the action taken is valid.
        """
        raise NotImplementedError("Action validation logic is not implemented.")
    
    @staticmethod
    def is_final(state: StateSciBench) -> bool:
        """
        Checks if the current state is a final state.
        """
        if len(state.values) > 0 and state.values[max(state.values)] >= 0.9:
            return True
        elif len(state.steps) > 0 and "answer is" in state.steps[-1].lower():
            return True
        else:
            return False
    
    @staticmethod
    def evaluate(state: StateSciBench) -> Tuple[bool, float]:
        """
        Evaluates the current state.
        """
        final = EnvironmentSciBench.is_final(state)
        if final:
            return True, verify_answer(state.answer, state.steps[-1])
        else:
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