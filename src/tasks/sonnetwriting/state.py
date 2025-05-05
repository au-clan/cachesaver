from dataclasses import dataclass
from typing import List
from ...typedefs import State


@dataclass(frozen=True)
class StateSonnetWriting(State):
    input: str

    current_state: str

    steps: List[str]

    target: str

    randomness: int

    def serialize(self) -> dict:
        """
        Returns a dictionary representation of the state.
        """
        return {
            "input": self.input,
            "current_state": self.current_state,
            "steps": "->".join(self.steps),
            "target": self.target,
            "randomness": self.randomness,
        }

    def clone(self, randomness: int = None) -> "StateSonnetWriting":
        """
        Returns a new instance of StateHotpotQA with an optional new randomness value.
        """
        return StateSonnetWriting(
            input=self.input,
            current_state=self.current_state,
            steps=self.steps,
            target=self.target,
            randomness=self.randomness,
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
