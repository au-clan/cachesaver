from abc import ABC, abstractmethod
from typing import Any

from .state_basic import StateBasic
from ...typedefs import Inference, Verification

class EnvironmentBasic(ABC):
    class Prompter:
        def __init__(self):
            self.name = "Environment Prompter"

        @staticmethod
        def cot(state: StateBasic) -> str:
            raise NotImplementedError

        @staticmethod
        def act(state: StateBasic) -> str:
            raise NotImplementedError

        @staticmethod
        def react(state: StateBasic) -> str:
            raise NotImplementedError

        @staticmethod
        def bfs(state: StateBasic) -> str:
            raise NotImplementedError

        @staticmethod
        def evaluate(state: StateBasic) -> str:
            raise NotImplementedError
    
    class Parser:
        def __init__(self):
            self.name = "Environment Parser"

        @staticmethod
        def cot(response: str) -> Inference:
            raise NotImplementedError

        @staticmethod
        def act(response: str) -> Inference:
            raise NotImplementedError

        @staticmethod
        def react(response: str) -> Inference:
            raise NotImplementedError

        @staticmethod
        def bfs(response: str) -> Inference:
            raise NotImplementedError

        @staticmethod
        def evaluate(response: str) -> Inference:
            raise NotImplementedError

    @classmethod
    def create(cls, task: str, data_path: str) -> "EnvironmentBasic":
        if task == "game24":
            from ..game24 import EnvironmentGame24
            return EnvironmentGame24(data_path)
        elif task == "hotpotqa":
            from ..hotpotqa import EnvironmentHotpotQA
            return EnvironmentHotpotQA(data_path)
        else:
            raise NotImplementedError(f"Task '{task}' is not implemented.")
    
    def __init__(self, task: str, data_path: str):
        pass  # The base class should not be directly instantiated

    @staticmethod
    @abstractmethod
    def reset(idx: int, randomness: int=0) -> StateBasic:
        """
        Given an index of the dataset and a randomness return the initial state.
        """
        pass


    @staticmethod
    @abstractmethod
    def get_next_state(inference: Inference, state: StateBasic, randomness: int) -> StateBasic:
        """
        Given a state and an inference return the next state.
        Inference is usually an action, a thought or both.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_value(state: StateBasic) -> Inference:
        """
        Deterministic evaluation mehtods. If there's no such methods returns None
        """
        inference = Inference(value=None)
        return inference

    @staticmethod
    @abstractmethod
    def verify(state: StateBasic) -> Verification:
        """
        Given a state return a verification
        """
        pass