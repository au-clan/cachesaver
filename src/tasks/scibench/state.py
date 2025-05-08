from dataclasses import dataclass, field
from typing import List, Dict

from ...typedefs import State

@dataclass(frozen=True)
class StateSciBench(State):
    # The initial question to solve
    puzzle: str

    # Current state towards solving the puzzle
    current_state: str

    # Steps taken towards solving the puzzle
    steps: List[str]

    # The true answer to the question
    answer: str

    # A random number associated with the state
    randomness: int
    
    # The number of steps taken so far
    step_n: int = 0

    # The value that the state had at its last evaluation
    values: Dict = field(default_factory=dict)

    def serialize(self) -> dict:
        """
        Returns a dictionary representation of the state.
        """
        return {
            "current_state": self.current_state,
            "steps": " -> ".join(self.steps)
        }
    
    def clone(self, randomness: int=None) -> "StateSciBench":
        """
        Returns a new instance of StateSciBench with an optional new randomness value.
        """
        return StateSciBench(
            puzzle=self.puzzle,
            current_state=self.current_state,
            steps=self.steps,
            answer=self.answer,
            step_n=self.step_n,
            values=self.values,
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