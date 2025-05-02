from typing import List
import re

from . import prompts as prompts
from .state import StateHumanEval
from ...typedefs import Request, Agent, Model, DecodingParameters

class AgentReActHumanEval(Agent):

    @staticmethod
    async def act(model: Model, state: StateHumanEval, namespace: str, request_id: str, params: DecodingParameters) -> List[str]:
        """
        Returns a list of actions for the Game of 24 task.
        """

        # Format the prompt
        if state.current_state == "24":
            prompt = prompts.cot.format(input=state.puzzle) + "\nSteps:\n" + '\n'.join(state.steps) + "\nAnswer: "
        else:
            current_numbers = get_current_numbers(state)
            prompt = prompts.react.format(input=current_numbers)

        # Generate the response
        response = await model.request(
            prompt=prompt,
            n=1,
            request_id=request_id,
            namespace=namespace,
            params=params
        )

        react_actions = [r.strip() for r in responses]

        return react_actions
