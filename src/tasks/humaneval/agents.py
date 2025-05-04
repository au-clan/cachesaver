from typing import List
import re

from . import prompts as prompts
from .state import StateHumanEval
from ...typedefs import Request, Agent, Model, DecodingParameters

class AgentActHumanEval(Agent):
    """
    """
    @staticmethod
    async def act(model: Model, state: StateHumanEval, n: int, namespace: str, request_id: str, params: DecodingParameters) -> List[str]:
        raise NotImplementedError("The act method for AgentActHumanEval is not implemented.")
    
class AgentAggregateHumanEval(Agent):
    @staticmethod
    async def act(model: Model, state: StateHumanEval, actions: List[str], k: int, namespace: str, request_id: str, params: DecodingParameters) -> List[str]:
        """
        Returns the aggregated actions for the HumanEval task.
        """
        # Format the prompt
        raise NotImplementedError("The act method for AgentAggregateHumanEval is not implemented.")
    
        # Generate the response

        # Parse the response

class AgentBfsHumanEval(Agent):
    @staticmethod
    async def act(model: Model, state: StateHumanEval, namespace: str, request_id: str, params: DecodingParameters) -> List[str]
        raise NotImplementedError("The act method for AgentBfsHumanEval is not implemented.")
    
class AgentEvaluateHumanEval(Agent):
    @staticmethod
    async def act(model: Model, state: StateHumanEval, n: int, namespace: str, request_id: str, params: DecodingParameters, cache: dict = None) -> float:
        """
        Returns the evaluation score for the HumanEval task.
        """
        raise NotImplementedError("The act method for AgentEvaluateHumanEval is not implemented.")