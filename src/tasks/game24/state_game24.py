from dataclasses import dataclass
from typing import List

from ..basic import StateBasic


@dataclass(frozen=True)
class StateGame24(StateBasic):
    # The initial puzzle to solve
    puzzle: str

    # Current state towards solving the puzzle
    current_state: str

    # Steps taken towards solving the puzzle
    steps: List[str]

    # A random number associated with the state
    randomness: int

    def __hash__(self):
        return hash((self.puzzle, self.current_state, " -> ".join(self.steps)))
    
    def items(self):
        return self.puzzle, self.current_state, self.steps, self.randomness
    
    def duplicate(self, randomness: int=None) -> "StateGame24":
        """
        Returns a new instance of GameOf24State with an optional new randomness value.
        """
        return StateGame24(
            puzzle=self.puzzle,
            current_state=self.current_state,
            steps=self.steps,
            randomness=randomness if randomness else self.randomness)