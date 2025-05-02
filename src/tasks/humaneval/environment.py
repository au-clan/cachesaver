import re
import random
from typing import Tuple
from sympy import simplify

from .state import StateHumanEval
from ...typedefs import Environment, MAX_SEED

import subprocess

class EnvironmentHumanEval(Environment):

    @staticmethod
    def step(state: StateHumanEval, action: str) -> StateHumanEval:
        """
        Takes a step in the environment based on the given action
        """

        # Parse the type and the argument of the action
        act = action.split("\n")[-1]
        action_type, argument = parse_action(act.split(": ")[-1])
        #assert "Action" in act, "Action not found in the action string."

        # Perform the action and obtain the observation
        obs = perform_action(action_type, argument, state.answer)
        step = f"\nAction {len(state.steps)+ 1}: " + action + f"\nObservation {len(state.steps) + 1}: {obs}"

        # Randomness handling
        random.seed(state.randomness)
        randomness = random.randint(0, MAX_SEED)

        state = StateHumanEval(
            puzzle=state.puzzle,
            current_state=current_state,
            steps=state.steps + [action],
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
        raise NotImplementedError("Action validation logic is not implemented.")

#---Helper functions---#
def parse_action(string):
    pattern = r'^(\w+)\[(.+)\]$'
    match = re.match(pattern, string)

    if match:
        action_type = match.group(1)
        argument = match.group(2)
        return action_type.lower().capitalize(), argument.strip()

    else:
        return None, None

def perform_action(docstore: DocstoreExplorer, action_type: str, argument: str, answer: str) -> str:
    if action_type == "Run":

        try:
            # Run the code in a subprocess and capture output
            result = subprocess.run(
                ['python', '-c', code_string],
                capture_output=True,
                text=True,
                check=True
            )
            obs = result.stdout
        except subprocess.CalledProcessError as e:
            obs = f"Error: {e.stderr}"

    else:
        obs = 'Invalid Action. Valid Actions are Lookup[<topic>], Search[<topic>] and Finish[<answer>].'

    return obs
