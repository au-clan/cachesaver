from dataclasses import dataclass
from typing import List
from langchain.agents.react.base import DocstoreExplorer

from ...typedefs import State

@dataclass(frozen=True)
class StateHotpotQA(State):

    # The initial question to solve
    puzzle: str

    # Current state towards solving the puzzle
    current_state: str

    # Steps taken towards solving the puzzle
    steps: List[str]

    # The true answer to the question
    answer: str

    # Docstore the current state is using
    docstore: DocstoreExplorer

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

    def clone(self, randomness: int=None) -> "StateHotpotQA":
        """
        Returns a new instance of StateHotpotQA with an optional new randomness value.
        """
        return StateHotpotQA(
            puzzle=self.puzzle,
            current_state=self.current_state,
            steps=self.steps,
            answer=self.answer,
            docstore=self.docstore,
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
    

