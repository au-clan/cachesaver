from abc import abstractmethod, ABC

from dataclasses import dataclass
from typing import List, Any

from .state_basic import StateBasic

@dataclass(frozen=True)
class DataBasic(ABC):
    # The initial puzzle to solve
    path: str
    data: Any

    @abstractmethod
    def get_indices(self, set:str) -> List[int]:
        pass

    @abstractmethod
    def get_initial_state(self, idx: int, randomness: int=0) -> StateBasic:
        pass

    def download_data(self,):
        raise NotImplementedError