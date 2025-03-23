from abc import abstractmethod, ABC

from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class StateBasic(ABC):
    # Current state towards solving the puzzle
    current_state: str

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def items(self):
        pass

    @abstractmethod
    def duplicate(self, randomness: int=None) -> "StateBasic":
        pass