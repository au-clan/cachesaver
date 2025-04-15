from typing import List, Tuple, Any, NamedTuple, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from cachesaver.typedefs import Batch, Response, SingleRequestModel, BatchRequestModel
from cachesaver.typedefs import Request as CacheSaverRequest
from torch.utils.data import Dataset

MAX_SEED = 10000

@dataclass(frozen=True)
class Request(CacheSaverRequest):# Clean this up
    # model: str
    max_completion_tokens: Optional[int]=None
    temperature: Optional[float]=1.0
    top_p: Optional[float]=1.0
    stop: Optional[str]=None
    logprobs: Optional[bool]=False
    messages: Optional[list[str]]=None

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
    def __init__(self, path: str, set_name: str):
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

    @staticmethod
    @abstractmethod
    def get_seed(self) -> int:
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

class Agent(ABC):

    @staticmethod
    @abstractmethod
    def act(model: Model, state: State) -> Any:
        pass
    
class Algorithm(ABC):
    def __init__(self, model: Model, agents: dict[str, Agent], env: Environment):
        self.model = model
        self.agent = agents
        self.env = env

    @abstractmethod
    async def solve(self) -> List[State]:
        pass

    @abstractmethod
    async def benchmark(self, benchmark: Benchmark) -> List[List[State]]:
        pass