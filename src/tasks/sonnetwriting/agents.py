from typing import List
import re

from . import prompts as prompts
from .state import StateSonnetWriting
from ...typedefs import Agent, Model, DecodingParameters

class AgentActSonnetWriting(Agent):
    """
    """
    @staticmethod
    async def act(model: Model, state: StateSonnetWriting, n: int, namespace: str, request_id: str, params: DecodingParameters) -> List[str]:
        # Format the prompt
        prompt = prompts.act.format(input=state.current_state)

        # Generate response
        responses = model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params
        )

        # Parse responses
        proposals = [r.strip() for r in responses]
        return proposals
    
class AgentAggregateSonnetWriting(Agent):
    """
    Returns the aggregate actions for the Sonnet Writing task.
    """
    @staticmethod
    async def act(model: Model, state: StateSonnetWriting, n: int, namespace: str, request_id: str, params: DecodingParameters) -> List[str]:
        raise NotImplementedError("AggregateAgent is not implemented yet.")
    
class AgentEvaluateSonnetWriting(Agent):
    """
    Returns the evaluations of states for the Sonnet Writing task.
    """
    @staticmethod
    async def act(model: Model, state: StateSonnetWriting, n: int, namespace: str, request_id: str, params: DecodingParameters) -> List[str]:
        raise NotImplementedError("EvaluationAgent is not implemented yet.")