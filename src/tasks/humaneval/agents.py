import itertools
import json
from typing import List
import re

from . import prompts as prompts
from .state import StateHumanEval
from ...typedefs import Request, Agent, Model, DecodingParameters


class AgentActHumanEval(Agent):
    """ """

    @staticmethod
    async def act(
        model: Model,
        state: StateHumanEval,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        raise NotImplementedError(
            "The act method for AgentActHumanEval is not implemented."
        )


class AgentActHumanEval(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StateHumanEval,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        """
        Returns actions generated for the HumanEval task.
        """
        # Format the prompt
        language = "py" if "def" in state.puzzle else "rs"
        instruct = prompts.SIMPLE_CHAT_INSTRUCTION_V2.format(lang=language)

        # Generate the response
        responses = await model.request(
            prompt=[
                {"role": "system", "content": instruct},
                {"role": "user", "content": state.current_state},
            ],
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        # Parse the responses
        actions = [r.strip() for r in responses]
        return actions


class AgentAggregateHumanEval(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StateHumanEval,
        actions: List[str],
        k: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        """
        Returns the aggregated actions for the HumanEval task.
        """
        # Format the prompt
        language = "py" if "def" in state.puzzle else "rs"
        instruct = prompts.SIMPLE_CHAT_INSTRUCTION_V2.format(lang=language)
        user_prompt = prompts.aggregate_prompt.format(
            prompt=state.current_state, k=k, implementations="\n".join(actions)
        )
        # Generate the response

        responses = await model.request(
            system_prompt=instruct,
            user_prompt=user_prompt,
            n=1,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        # Parse the response
        pattern = r"```[^`]+```"
        matchs = re.findall(pattern, responses[0])
        return matchs

class AgentBfsHumanEval(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StateHumanEval,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        language = "py" if "def" in state.puzzle else "rs"
        instruct = prompts.SIMPLE_CHAT_INSTRUCTION_BFS.format(lang=language, n=3)

        responses = await model.request(
            system_prompt=instruct,
            user_prompt=state.current_state,
            n=1,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        response_text = responses[0]

        code_blocks = re.findall(r'(```.*?```)', response_text, flags=re.DOTALL)

        actions = [block.strip() for block in code_blocks]

        return actions



def sum_overall_scores(text):
    if isinstance(text, list):

        text = " ".join(
            [str(item) for sublist in text for item in (sublist if isinstance(sublist, list) else [sublist])]
        )

    scores = re.findall(r'Overall Score:\s*(\d+)', text)

    if not scores:
        return None

    return sum(int(score) for score in scores)


class AgentEvaluateHumanEval(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StateHumanEval,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
        cache: dict = None,
    ) -> float:
        """
        Returns the evaluation score for the HumanEval task.
        """
        language = "py" if "def" in state.puzzle else "rs"
        instruct = prompts.SIMPLE_CHAT_INSTRUCTION_V2.format(lang=language)

        user_prompt = prompts.evaluation_prompt.format(
            prompt=state.puzzle,
            implementation=state.current_state
        )

        responses = await model.request(
            system_prompt=instruct,
            user_prompt=user_prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        return sum_overall_scores(responses)