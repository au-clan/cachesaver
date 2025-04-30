from dataclasses import dataclass

from ...typedefs import State


@dataclass(frozen=True)
class StateMathArena(State):

    problem_idx: str
    problem: str
    answer: str


    def serialize(self) -> dict:
        """
        Returns a dictionary representation of the state.
        """
        return {
            "problem_idx": self.problem_idx,
            "problem": self.problem,
            "answer": self.answer,
        }


    def clone(self, randomness: int = None) -> "StateMathArena":
        """
        Returns a new instance of StateHotpotQA with an optional new randomness value.
        """
        return StateMathArena(
            problem_idx=self.problem_idx,
            problem=self.problem,
            answer=self.answer
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
