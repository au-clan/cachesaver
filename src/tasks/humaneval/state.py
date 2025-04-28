from dataclasses import dataclass
from typing import List

from ...typedefs import State

@dataclass(frozen=True)
class StateHumanEval(State):
    # The initial code to complete
    puzzle: str

    # Current completion
    current_state: str

    # Steps taken towards solution
    steps: List[str]

    # The canonical solution for the puzzle
    canonical_solution: str

    # Entry point for testing the code
    entry_point: str

    # The tests to run against the code
    test: str

    # A random number associated with the state
    randomness: int

    def serialize(self) -> dict:
        """
        Returns a dictionary representation of the state.
        """
        return {
            "current_state": self.current_state,
            "steps": " -> ".join(self.steps)
        }
    
    def clone(self, randomness: int= None) -> "StateHumanEval":
        """
        Returns a new instance of StateHumanEval with an optional new randomness value.
        """
        return StateHumanEval(
            puzzle=self.puzzle,
            current_state=self.current_state,
            steps=self.steps,
            canonical_solution=self.canonical_solution,
            entry_point=self.entry_point,
            test=self.test,
            randomness=randomness or self.randomness
        )
    
    def get_seed(self) -> int:
        """
        Returns the randomness value associated with the state.
        """
        return self.randomness
    
    def __hash__(self) -> int:
        """
        Returns a hash of the current state.
        """
        return hash(str(self.serialize()))