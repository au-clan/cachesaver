import re
import random
from typing import Tuple
from sympy import simplify

from .state import StateHLE
from ...typedefs import Environment, MAX_SEED

class EnvironmentHLE(Environment):

    @staticmethod
    def step(state: StateHLE, action: str) -> StateHLE:
        """
        Takes a step in the environment based on the given action
        """

        # Parse the type and the argument of the action
        act = action.split("\n")[-1]
        action_type, argument = parse_action(act.split(": ")[-1])
        #assert "Action" in act, "Action not found in the action string."

        # Perform the action and obtain the observation
        obs = perform_action(state.docstore, action_type, argument, state.answer)
        step = f"\nAction {len(state.steps)+ 1}: " + action + f"\nObservation {len(state.steps) + 1}: {obs}"

        # Randomness handling
        random.seed(state.randomness)
        randomness = random.randint(0, MAX_SEED)

        state = StateHLE(
            puzzle=state.puzzle,
            current_state=current_state,
            steps=state.steps + [action],
            docstore=state.docstore,
            randomness=randomness
        )
        return state

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
    if action_type == "Search":
        try:
            # Added '' around the argument. Not in reflexion. After some (small) testing, it seems to be equal or better.
            obs = docstore.search(f"\'{argument}\'").strip("\n").strip()
        except Exception as e:
            print(f"Error searching for '{argument}'")
            obs = 'Page does not exist, try something else.'
    elif action_type == "Lookup":
        try:
            obs = docstore.lookup(argument).strip('\n').strip()
        except Exception as e:
            print(f"Error looking up '{argument}'")
            obs = 'The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given in the previous observation.'

    elif action_type == "Finish":
        if argument.lower() == answer.lower():
            obs = OBS_CORRECT
        else:
            obs = OBS_INCORRECT

    else:
        obs = 'Invalid Action. Valid Actions are Lookup[<topic>], Search[<topic>] and Finish[<answer>].'
    return obs