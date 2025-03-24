from abc import abstractmethod, ABC

from dataclasses import dataclass
from typing import List, Any

@dataclass(frozen=True)
class StateBasic(ABC):
    # The initial puzzle to solve
    puzzle: Any
    
    # Current state towards solving the puzzle
    current_state: Any

    # Steps taken towards solving the puzzle
    steps: List[Any]

    # A random number associated with the state
    randomness: int

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def items(self):
        pass

    @abstractmethod
    def duplicate(self, randomness: int=None) -> "StateBasic":
        pass