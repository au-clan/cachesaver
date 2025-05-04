import random
import re
from typing import Tuple

from langchain.agents.react.base import DocstoreExplorer

from .state import StateHotpotQA
from ...typedefs import Environment, MAX_SEED

OBS_CORRECT = "Answer is CORRECT."
OBS_INCORRECT = "Answer is INCORRECT."


class EnvironmentHotpotQA(Environment):
    """
    Environment for the HotpotQA task.
    """

    @staticmethod
    def step(state: StateHotpotQA, action: str) -> StateHotpotQA:
        """
        Takes a step in the environment based on the given action.
        """

        # Parse the type and the argument of the action
        act = action.split("\n")[-1]
        action_type, argument = parse_action(act.split(": ")[-1])
        # assert "Action" in act, "Action not found in the action string."

        # Perform the action and obtain the observation
        obs = perform_action(state.docstore, action_type, argument, state.answer)
        step = f"\nAction {len(state.steps) + 1}: " + action + f"\nObservation {len(state.steps) + 1}: {obs}"

        # Randomness handling
        random.seed(state.randomness)
        randomness = random.randint(0, MAX_SEED)

        state = StateHotpotQA(
            puzzle=state.puzzle,
            current_state=(state.current_state + step).strip(),
            steps=state.steps + [step],
            answer=state.answer,
            docstore=state.docstore,
            randomness=randomness
        )
        return state

    @staticmethod
    def is_valid(state: StateHotpotQA, action: str) -> bool:
        """
        Checks if the action taken is valid.
        """
        raise NotImplementedError("Action validation logic is not implemented.")

    @staticmethod
    def is_final(state: StateHotpotQA) -> bool:
        """
        Checks if the current state is a final state.
        """
        expression = state.current_state.split("\n")[-2]
        action_type, _ = parse_action(expression.split(": ")[-1])
        if action_type == "Finish":
            return True
        else:
            return False

    @staticmethod
    def evaluate(state: StateHotpotQA) -> Tuple[bool, float]:
        """
        Evaluates the current state.
        """
        is_final = EnvironmentHotpotQA.is_final(state)
        if is_final is True:
            obs = state.current_state.split("\n")[-1]
            if obs == OBS_CORRECT:
                return True, 1.0
            else:
                return True, 0.0
        else:
            return False, 0.0


# ---Helper functions---#
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
