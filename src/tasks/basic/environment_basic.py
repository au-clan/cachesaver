from abc import ABC, abstractmethod
from typing import Any

class EnvironmentBasic(ABC):
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
