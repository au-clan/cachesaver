from dataclasses import dataclass

from ...typedefs import State


@dataclass(frozen=True)
class StateLogiQA(State):
    right_choice: str
    context: str
    question: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str

    def serialize(self) -> dict:
        """
        Returns a dictionary representation of the state.
        """
        return {
            'right_choice': self.right_choice,
            'context': self.context,
            'question': self.question,
            'option_a': self.option_a,
            'option_b': self.option_b,
            'option_c': self.option_c,
            'option_d': self.option_d
        }

    def clone(self, randomness: int = None) -> "StateLogiQA":
        """
        Returns a new instance of GameOf24State with an optional new randomness value.
        """

        return StateLogiQA(
            right_choice=self.right_choice,
            context=self.context,
            question=self.question,
            option_a=self.option_a,
            option_b=self.option_b,
            option_c=self.option_c,
            option_d=self.option_d)

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
