from dataclasses import dataclass

from ...typedefs import State


@dataclass(frozen=True)
class StateSonnetWriting(State):
    input: str

    target: str

    def serialize(self) -> dict:
        """
        Returns a dictionary representation of the state.
        """
        return {
            "input": self.input,
            "target": self.target,
        }

    def clone(self, randomness: int = None) -> "StateSonnetWriting":
        """
        Returns a new instance of StateHotpotQA with an optional new randomness value.
        """
        return StateSonnetWriting(
            input=self.input,
            target=self.target,
        )

    def get_seed(self) -> int:
        """
        Returns the randomness value associated with the state.
        """
        pass
        return self.randomness

    def __hash__(self) -> int:
        """
        Returns a hash of the current state.
        """
        return hash(str(self.serialize()))
