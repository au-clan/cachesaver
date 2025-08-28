from dataclasses import dataclass, field
from typing import List, Dict

from ...typedefs import State

@dataclass(frozen=True)
class StateMathArena(State):
    # The mathematical problem to solve
    problem: str
    
    # The expected answer
    answer: str
    
    # Current state of reasoning
    current_state: str
    
    # Steps taken in solution
    steps: List[str]

    # Random seed for exploration
    randomness: int

    # The number of steps taken so far
    step_n: int = 0

    # The value that the state had at its last evaluation
    values: Dict = field(default_factory=dict)

    def serialize(self) -> dict:
        """Returns a dictionary representation of the state."""
        return {
            "current_state": self.current_state,
            "steps": " -> ".join(self.steps)
        }
    
    def clone(self, randomness: int=None) -> "StateMathArena":
        """Returns a new instance with optional new randomness."""
        return StateMathArena(
            problem=self.problem,
            answer=self.answer,
            current_state=self.current_state,
            steps=self.steps,
            step_n=self.step_n,
            values=self.values,
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