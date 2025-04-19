from dataclasses import dataclass, field
from typing import List, Optional

from ...typedefs import State



@dataclass(frozen=True)
class StateGame24(State):
    # The initial puzzle to solve
    puzzle: Optional[str] = None

    # Current state towards solving the puzzle
    current_state: Optional[str] = None

    # Steps taken towards solving the puzzle
    steps: Optional[List[str]] = field(default_factory=list)
    # A random number associated with the state
    randomness: Optional[int] = None



    def serialize(self) -> dict:
        """
        Returns a dictionary representation of the state.
        """
        return {
            "current_state": self.current_state,
            "steps": " -> ".join(self.steps)
        }
    
    def clone(self, randomness: int=None) -> "StateGame24":
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

@dataclass(frozen=True)
class GameState_rafa():
    # index: Optional[int] = None
    #
    # history: Optional[list[str]] = field(default_factory=list)
    # feedbacks: Optional[list[str]] = field(default_factory=list)
    cur_step: Optional[int] = 0

    #these two should maybe be shared across runs?
    # reflects: Optional[list[str]] = field(default_factory=list)
    # value_reflects: Optional[list[str]] = field(default_factory=list)

    # #these two below have they formatted weird in rafa // also for the reflect
    # feedback: Optional[list[str]] = field(default_factory=list)
    #
    # action: Optional[str] = ""
    #
    # feedback_string: Optional[list[str]] = field(default_factory=list)
    # answer_string: Optional[list[str]] = field(default_factory=list)
    puzzle: Optional[str] = None
    ##used attributes
    obs_feedback:Optional[str]=""
    obs_answer:Optional[str]=""

    reflects: Optional[list[str]] = field(default_factory=list)
    value_reflects: Optional[list[str]] = field(default_factory=list)

    obs_history: Optional[list[dict[str, str]]] = field(default_factory=list)
    env_history: Optional[list[str]] = field(default_factory=list)

    history: Optional[list[str]] = field(default_factory=list)