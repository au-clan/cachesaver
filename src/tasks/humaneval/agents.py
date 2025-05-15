from typing import List
import re
import itertools


from . import prompts as prompts
from .state import StateHumanEval
from ...typedefs import Request, Agent, Model, DecodingParameters

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
            prompt=[
                {"role": "system", "content": instruct},
                {"role": "user", "content": user_prompt},
            ],
            n=1,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )
        

        # Parse the response
        pattern = r"```[^`]+```"
        matchs = re.findall(pattern, responses[0])
        return matchs if matchs else responses[0]


class AgentBfsHumanEval(Agent):
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
        ### change n, depending on how many to generate
        instruct = prompts.SIMPLE_CHAT_INSTRUCTION_BFS.format(lang=language, n=5)
        responses = await model.request(
            prompt=[
                {"role": "system", "content": instruct},
                {"role": "user", "content": state.current_state},
            ],
            n=1,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )
        response_text = responses[0]

        code_blocks = re.findall(r'(```.*?```)', response_text, flags=re.DOTALL)

        # Strip each code block
        actions = [block.strip() for block in code_blocks]

        return actions



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

        responses = await model.request(
            prompt=[
                {"role": "system", "content": instruct},
                {"role": "user", "content": user_prompt},
            ],
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )
        value = sum_overall_scores(responses)
        return responses, value

# Helper function
def sum_overall_scores(evaluations):
    values = []
    pattern = r"\b(?:overall[\s_]?score|score)\b(?:\s*(?:is|=|:|was|stands at|of))?\s*(-?\d+(?:\.\d+)?)"
    
    for evaluation in evaluations:
        match = re.search(pattern, evaluation, re.IGNORECASE)
        if match:
            value = float(match.group(1))
        else:
            value = 1
        values.append(value)
    value = sum(values)

    return value