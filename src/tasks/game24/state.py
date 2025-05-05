from dataclasses import dataclass
from typing import List

from ...typedefs import State


@dataclass(frozen=True)
class StateGame24(State):
    # The initial puzzle to solve
    puzzle: str

    # Current state towards solving the puzzle
    current_state: str

    # Steps taken towards solving the puzzle
    steps: List[str]
    # A random number associated with the state
    randomness: int

    # for rafa
    history = []
    feedbacks = []
    current_step = 0

    def serialize(self) -> dict:
        """
        Returns a dictionary representation of the state.
        """
        return {
            "current_state": self.current_state,
            "steps": " -> ".join(self.steps)
        }

    def clone(self, randomness: int = None) -> "StateGame24":
        """
        Returns a new instance of GameOf24State with an optional new randomness value.
        """
        return StateGame24(
            puzzle=self.puzzle,
            current_state=self.current_state,
            steps=self.steps,
            randomness=randomness or self.randomness)

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
