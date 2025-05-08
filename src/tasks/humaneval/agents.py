import itertools
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


class AgentGenerateHumanEval(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StateHumanEval,
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
        # Updated call to match new function signature
        responses = await model.request(
            system_prompt=instruct,
            user_prompt=state.current_state,
            n=4,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        # Parse the responses
        # Flatten the list of responses first and then strip each string
        flattened_responses = list(itertools.chain.from_iterable(responses))

        # Now, strip each string
        actions = [r.strip() for r in flattened_responses]
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
        raise NotImplementedError(
            "The act method for AgentBfsHumanEval is not implemented."
        )


def sum_overall_scores(text):
    if isinstance(text, list):
        # Flatten and join all items that are strings
        text = " ".join(
            [str(item) for sublist in text for item in (sublist if isinstance(sublist, list) else [sublist])]
        )

    scores = re.findall(r'Overall Score:\s*(\d+)', text)

    if not scores:
        return None  # or return 0 if preferred

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
            prompt=state.puzzle,  # The function signature + docstring
            implementation=state.current_state  # The code you want to evaluate
        )

        # Generate the response
        responses = await model.request(
            system_prompt=instruct,
            user_prompt=user_prompt,
            n=3,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        # do not know how to evaluate for humaneval so just passing on 0
        return sum_overall_scores(responses)
