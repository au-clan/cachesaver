from dataclasses import dataclass

from ...typedefs import State


@dataclass(frozen=True)
class StateHumanEval(State):
    prompt: str
    canonical_solution: str
    entry_point: str
    test: str

    def serialize(self) -> dict:
        """
        Returns a dictionary representation of the state.
        """
        return {
            "prompt": self.prompt,
            "canonical_solution": self.canonical_solution,
            "entry_point": self.entry_point,
            "test": self.test
        }

    def clone(self, randomness: int = None) -> "StateHumanEval":
        """
        Returns a new instance of StateHotpotQA with an optional new randomness value.
        """
        return StateHumanEval(
            prompt=self.prompt,
            canonical_solution=self.canonical_solution,
            entry_point=self.entry_point,
            test=self.test,
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
