from typing import List

from .state import StateHumanEval
from ...typedefs import Agent, Model, ModelRequestOptions


class AgentActHumanEval(Agent):
    """
    """

    @staticmethod
    async def act(model: Model, state: StateHumanEval, n: int, namespace: str, request_id: str,
                  params: ModelRequestOptions) -> List[str]:
        raise NotImplementedError("The act method for AgentActHumanEval is not implemented.")


class AgentAggregateHumanEval(Agent):
    @staticmethod
    async def act(model: Model, state: StateHumanEval, actions: List[str], k: int, namespace: str, request_id: str,
                  params: ModelRequestOptions) -> List[str]:
        """
        Returns the aggregated actions for the HumanEval task.
        """
        # Format the prompt
        raise NotImplementedError("The act method for AgentAggregateHumanEval is not implemented.")

        # Generate the response

        # Parse the response


class AgentBfsHumanEval(Agent):
    @staticmethod
    async def act(model: Model, state: StateHumanEval, namespace: str, request_id: str, params: ModelRequestOptions) -> \
    List[str]:
        raise NotImplementedError("The act method for AgentBfsHumanEval is not implemented.")


class AgentEvaluateHumanEval(Agent):
    @staticmethod
    async def act(model: Model, state: StateHumanEval, n: int, namespace: str, request_id: str,
                  params: ModelRequestOptions, cache: dict = None) -> float:
        """
        Returns the evaluation score for the HumanEval task.
        """
        raise NotImplementedError("The act method for AgentEvaluateHumanEval is not implemented.")
