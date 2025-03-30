from abc import abstractmethod, ABC

from dataclasses import dataclass, field
from typing import List, Any, Optional


@dataclass(frozen=True)
class StateBasic(ABC):
    # The initial puzzle to solve
    puzzle: Optional[Any] = None
    
    # Current state towards solving the puzzle
    current_state: Optional[Any]= None

    # Steps taken towards solving the puzzle
    steps: Optional[List[Any]] = field(default_factory=list)

    # A random number associated with the state
    randomness: Optional[int]= None

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def items(self):
        pass

    @abstractmethod
    def duplicate(self, randomness: int=None) -> "StateBasic":
        pass