from typing import Tuple, List
from dataclasses import dataclass
from ...typedefs import Environment
from .state import StateMathArena

@dataclass
class EnvironmentMathArena(Environment):
    """Environment for MathArena task."""

    @staticmethod
    def evaluate(state: StateMathArena) -> Tuple[bool, float]:
        """
        Evaluates the current state.

        Args:
            state (StateMathArena): Current state to evaluate

        Returns:
            Tuple[bool, float]: (is_finished, score)
        """
        # Check if we have any steps
        if not state.steps:
            return False, 0.0

        last_step = state.steps[-1]

        # Not finished if last step isn't a Finish action
        if not last_step.startswith("Finish["):
            return False, 0.0

        # Extract and compare answer
        predicted_answer = last_step[7:-1].strip()  # Remove "Finish[" and "]"
        is_correct = predicted_answer == state.answer.strip()
        
        # Scoring logic: 
        # - 1.0 for correct answer
        # - 0.0 for incorrect answer
        return True, float(is_correct)

    @staticmethod
    def is_valid(state: StateMathArena) -> bool:
        """
        Checks if the current state is valid.

        Args:
            state (StateMathArena): State to check

        Returns:
            bool: True if state is valid
        """
        if not state.problem or not state.parsed_problem or not state.answer:
            return False

        if not isinstance(state.steps, list):
            return False

        valid_prefixes = ("Analyze[", "Explain[", "Finish[")
        
        for step in state.steps:
            if not isinstance(step, str):
                return False
            if not any(step.startswith(prefix) for prefix in valid_prefixes):
                return False
            if not step.endswith("]"):
                return False

        return True

    @staticmethod
    def get_valid_actions(state: StateMathArena) -> List[str]:
        """
        Returns list of valid actions for current state.

        Args:
            state (StateMathArena): Current state

        Returns:
            List[str]: List of valid actions
        """
        actions = [
            "Analyze[problem]",
            "Analyze[solution approach]",
            "Explain[math concepts]",
            "Explain[solution steps]"
        ]

        # Allow finishing only after at least 2 analysis/explanation steps
        if len(state.steps) >= 2:
            actions.append("Finish[answer]")

        return actions

    @staticmethod
    def apply_action(state: StateMathArena, action: str) -> StateMathArena:
        """
        Applies an action to the current state.

        Args:
            state (StateMathArena): Current state
            action (str): Action to apply

        Returns:
            StateMathArena: New state after applying action
        """
        new_state = state.copy()
        new_state.steps.append(action)
        return new_state