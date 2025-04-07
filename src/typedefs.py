from typing import List, Tuple, Any, NamedTuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from cachesaver.typedefs import Request, Batch, Response, SingleRequestModel, BatchRequestModel
from torch.utils.data import Dataset

class DecodingParameters(NamedTuple):
    max_completion_tokens: int
    temperature: float
    top_p: float
    stop: str
    logprobs: bool

class Model(SingleRequestModel, BatchRequestModel):
    def __init__(self):
        pass

    @abstractmethod
    async def request(self, request: Request) -> Response:
        pass

    @abstractmethod
    async def batch_request(self, batch: Batch) -> List[Response]:
        pass

class Benchmark(Dataset):
    def __init__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


@dataclass(frozen=True)
class State(ABC):

    @staticmethod
    @abstractmethod
    def serialize(self) -> dict:
        pass

    @staticmethod
    @abstractmethod
    def clone(self, randomness: int) -> "State":
        pass


class Agent(ABC):
    def __init__(self, model: Model):
        self.model = model

    @staticmethod
    @abstractmethod
    def act(model: Model, state: State, n: int) -> List[str]:
        pass

    @staticmethod
    @abstractmethod
    def bfs(model: Model, state: State) -> List[str]:
        pass

    @staticmethod
    @abstractmethod
    def react(model: Model, state: State, n: int) -> List[str]:
        pass

    @staticmethod
    @abstractmethod
    def reflect(model: Model, state: State, n: int) -> List[str]:
        pass

    @staticmethod
    @abstractmethod
    def evaluate(model: Model, state: State, n: int) -> int:
        pass
    
class Environment(ABC):
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def step(state: State, action: str) -> Tuple[State, float]:
        pass

    @staticmethod
    @abstractmethod
    def is_valid(state: State, a: str) -> bool:
        pass

    @staticmethod
    @abstractmethod
    def is_final(state: State) -> bool:
        pass

    @staticmethod
    @abstractmethod
    def evaluate(state: State) -> Tuple[bool, float]:
        pass

class Heuristic(ABC):
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def value(state: State) -> float:
        pass

class Algorithm(ABC):
    def __init__(self, model: Model, agent: Agent, env: Environment, heuristic: Heuristic):
        self.model = model
        self.agent = agent
        self.env = env
        self.heuristic = heuristic

    @abstractmethod
    async def solve(self):
        pass
