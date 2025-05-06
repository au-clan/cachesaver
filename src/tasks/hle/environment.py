import re
import random
from typing import Tuple

from .state import StateHLE
from ...typedefs import Environment, MAX_SEED

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
        # Parse the type and the argument of the action
        act = action.split("\n")[-1]
        action_type, argument = parse_action(act.split(": ")[-1])
        
        # Perform the action and obtain the observation
        obs = perform_action(action_type, argument, state)
        step = f"\nAction {len(state.steps)+ 1}: " + action + f"\nObservation {len(state.steps) + 1}: {obs}"

        # Randomness handling
        random.seed(state.randomness if hasattr(state, 'randomness') else 0)
        randomness = random.randint(0, MAX_SEED)

        # Create new state with updated information
        state = StateHLE(
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
            steps=state.steps + [step],
            randomness=state.randomness
        )
        return state

    @staticmethod
    def is_valid(state: StateHLE, action: str) -> bool:
        """
        Checks if the action taken is valid.
        """
        action_type, _ = parse_action(action.split("\n")[-1].split(": ")[-1])
        return action_type in ["Analyze", "Explain", "Finish"]

    @staticmethod
    def is_final(state: StateHLE) -> bool:
        """
        Checks if the current state is a final state.
        """
        if not state.steps:
            return False
        expression = state.steps[-1].split("\n")[-2]  # Get the last action
        action_type, _ = parse_action(expression.split(": ")[-1])
        return action_type == "Finish"

    @staticmethod
    def evaluate(state: StateHLE) -> Tuple[bool, float]:
        """
        Evaluates the current state.
        """
        is_final = EnvironmentHLE.is_final(state)
        if is_final:
            last_obs = state.steps[-1].split("\n")[-1]
            if last_obs == OBS_CORRECT:
                return True, 1.0
            return True, 0.0
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

def perform_action(action_type: str, argument: str, state: StateHLE) -> str:
    """
    Performs the specified action and returns an observation.
    """
    if action_type == "Analyze":
        # Analyze the image or question content
        return f"Analysis of '{argument}': Considering {state.category} category..."
    
    elif action_type == "Explain":
        # Provide explanation based on rationale
        return f"Explanation: {state.rationale if hasattr(state, 'rationale') else 'No rationale available'}"
    
    elif action_type == "Finish":
        # Check if the answer matches
        if argument.lower() == state.answer.lower():
            return OBS_CORRECT
        return OBS_INCORRECT
    
    else:
        return 'Invalid Action. Valid Actions are Analyze[<topic>], Explain[<aspect>], and Finish[<answer>].'