from dataclasses import dataclass
import random
from typing import List

from ...typedefs import State


@dataclass(frozen=True)
class StateLogiQA(State):
    context: str
    question: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    current_state: str
    steps: List[str]
    correct_option: str
    randomness: int

    def serialize(self) -> dict:
        """
        Returns a dictionary representation of the state.
        """
        return {
            'context': self.context,
            'question': self.question,
            'option_a': self.option_a,
            'option_b': self.option_b,
            'option_c': self.option_c,
            'option_d': self.option_d,
            'current_state': self.current_state,
            'steps': '->'.join(self.steps),
            'correct_option': self.correct_option,
            'randomness': self.randomness
        }

    def clone(self, randomness: int = None) -> "StateLogiQA":
        """
        Returns a new instance of GameOf24State with an optional new randomness value.
        """

        return StateLogiQA(
            context=self.context,
            question=self.question,
            option_a=self.option_a,
            option_b=self.option_b,
            option_c=self.option_c,
            option_d=self.option_d,
            current_state=self.current_state,
            steps=self.steps,
            correct_option=self.correct_option,
            randomness=self.randomness
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
