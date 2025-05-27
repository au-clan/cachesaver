from typing import List
import re

from . import prompts as prompts
from .state import StateHumanEval
from ..utils import scale_logprob_reward_linear
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

        return sum_overall_scores(responses)

    @staticmethod
    async def self_eval_reward(model: Model, state: StateHumanEval,
                               num_evaluations: int, namespace: str, request_id: str,
                               eval_params: DecodingParameters) -> float:
        responses = await model.request(
            prompt=prompts.self_evaluation_prompt.format(prompt = state.puzzle, implementation=state.current_state),
            namespace=namespace,
            n=num_evaluations,
            request_id=request_id,
            params=eval_params,
        )
        yes_count = sum(1 for r in responses if r.strip().lower().startswith("yes"))
        score = yes_count / num_evaluations
        score  = (score * 45) + 5 # based on task specific evaluation where score can be (5, 50)
        return score

    @staticmethod
    async def logprobs_reward(logprobs_model: Model, state: StateHumanEval, namespace: str, request_id: str,
                              eval_params: DecodingParameters) -> float:
        if logprobs_model is None or not eval_params.logprobs:
            return 0
        responses = await logprobs_model.request(
            prompt=prompts.self_evaluation_prompt.format(prompt = state.puzzle, implementation=state.current_state),
            namespace=namespace,
            n=1,
            request_id=request_id,
            params=eval_params,
            return_logprobs=True
        )
        _, token_logprobs = responses
        label, logprob = token_logprobs[0][0]
        if label.strip().lower() == 'yes':
            return scale_logprob_reward_linear(logprob, 45) + 5
        elif label.strip().lower() == 'no':
            return scale_logprob_reward_linear(logprob, 45, inverse=True) + 5
        return 0

# Helper function
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