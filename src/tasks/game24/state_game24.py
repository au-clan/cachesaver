from dataclasses import dataclass, field
from typing import List, Optional

from ..basic import StateBasic


@dataclass(frozen=True)
class StateGame24(StateBasic):
    # The initial puzzle to solve
    puzzle: Optional[str] = None

    # Current state towards solving the puzzle
    current_state: Optional[str] = None

    # Steps taken towards solving the puzzle
    steps: Optional[List[str]] = field(default_factory=list)
    # A random number associated with the state
    randomness: Optional[int] = None

    index: Optional[int] = None

    history: Optional[list[str]] = field(default_factory=list)
    feedbacks: Optional[list[str]] = field(default_factory=list)

    reflects: Optional[list[str]] = field(default_factory=list)
    value_reflects: Optional[list[str]] = field(default_factory=list)
    cur_step: Optional[int] = 0
    last_feedback: Optional[str] = ""
    feedback: Optional[list[str]] = field(default_factory=list)
    action: Optional[str] = ""

    # feedback = True

    def __hash__(self):
        return hash((self.puzzle, self.current_state, " -> ".join(self.steps)))

    def items(self):
        return self.puzzle, self.current_state, self.steps, self.randomness

    def duplicate(self, randomness: int = None) -> "StateGame24":
        """
        Returns a new instance of GameOf24State with an optional new randomness value.
        """
        return StateGame24(
            puzzle=self.puzzle,
            current_state=self.current_state,
            steps=self.steps,
            randomness=randomness if randomness else self.randomness)
