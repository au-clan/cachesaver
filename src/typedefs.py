import random
import asyncio
from typing import List, Tuple, Any, NamedTuple, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from omegaconf import OmegaConf
from cachesaver.typedefs import Batch, Response, SingleRequestModel, BatchRequestModel
from cachesaver.typedefs import Request as CacheSaverRequest
from torch.utils.data import Dataset
from src.utils import timed

MAX_SEED = 10000

@dataclass(frozen=True)
class Request(CacheSaverRequest):# Clean this up
    model: str
    max_completion_tokens: Optional[int]=None
    temperature: Optional[float]=1.0
    top_p: Optional[float]=1.0
    stop: Optional[str]=None
    logprobs: Optional[bool]=False

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
    
class Method(ABC):
    def __init__(self, model: Model, agents: dict[str, Agent], env: Environment, config: OmegaConf):
        self.model = model
        self.agent = agents
        self.env = env
        self.config = config

    @abstractmethod
    async def solve(self) -> List[State]:
        pass

    async def benchmark(self, benchmark: Benchmark, ns_ratio: bool=False, **kwargs) -> Tuple[List[float], List[List[State]]]:
        cache = {} if kwargs.pop("value_cache", False) else None

        # Set up Namespace distibution
        n_shared = int(ns_ratio * len(benchmark))
        n_unique = len(benchmark) - n_shared
        namespaces = [f"benchmark_{0}" for _ in range(n_shared)] + [f"benchmark_{i+1}" for i in range(n_unique)]
        
        random.seed(42)
        random.shuffle(namespaces)

        solve_coroutines = [
            timed(
                label=f"Idx: {index}",
                coroutine=self.solve(
                idx=index,
                state=state,
                namespace=ns,
                **{"value_cache": cache} if cache is not None else {},
                **kwargs
                )
            )
            for (index, state), ns in zip(benchmark, namespaces)
        ]
        
        # Run all solves in parallel
        # Results : [Label, Duration, States]
        results: List[Tuple[str, float, List[State]]] = await asyncio.gather(*solve_coroutines)
        results = sorted(results, key=lambda x: x[0])
        labels, durations, states = zip(*results)
        return durations, states