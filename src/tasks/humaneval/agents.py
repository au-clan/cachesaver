from typing import List
import re
import itertools
import numpy as np

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
        instruct = prompts.SIMPLE_CHAT_INSTRUCTION_BFS.format(lang=language)
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
        return value

class AgentReactHumanEval(Agent):
    """
    Agent performing the ReAct operation for the HumanEval task.
    """
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
        Returns a list of n thought-action pairs for the HumanEval task.
        """
        # Format the prompt
        language = "py" if "def" in state.puzzle else "rs"
        instruct = prompts.SIMPLE_CHAT_INSTRUCTION_V2.format(lang=language)
        react_prompt = prompts.react.format(
            prompt=state.puzzle,
            current_state=state.current_state
        )

        # Generate the responses
        responses = await model.request(
            prompt=[
                {"role": "system", "content": instruct},
                {"role": "user", "content": react_prompt},
            ],
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        # Parse the responses
        react_actions = [r.strip() for r in responses]
        return react_actions

class AgentSelfEvaluateHumanEval(Agent):
    """
    Agent that performs self-evaluation of reasoning steps for HumanEval.
    Uses the LLM's own estimation of correctness by evaluating each reasoning step.
    Uses the probability of "Yes" as a reward signal for correct reasoning.
    """
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
        Returns a value estimation for the current state based on self-evaluation.
        """
        if cache is not None and state.current_state in cache:
            return cache[state.current_state]

        # Format the prompt based on whether we're evaluating a final answer or intermediate step
        language = "py" if "def" in state.puzzle else "rs"
        instruct = prompts.SIMPLE_CHAT_INSTRUCTION_V2.format(lang=language)

        if state.steps and "Finish" in state.steps[-1]:
            # Evaluating a final answer
            answer = state.steps[-1].replace("Finish[", "").replace("]", "")
            prompt = prompts.self_evaluate_answer.format(
                prompt=state.puzzle,
                steps='\n'.join(state.steps),
                answer=answer
            )
        else:
            # Evaluating intermediate reasoning steps
            last_step = state.steps[-1] if state.steps else ""
            prompt = prompts.self_evaluate_step.format(
                prompt=state.puzzle,
                current_state=state.current_state,
                step=last_step
            )

        eval_params = DecodingParameters(
            temperature=params.temperature,
            max_completion_tokens=params.max_completion_tokens,
            top_p=params.top_p,
            stop=params.stop,
            logprobs=True
        )

        responses = await model.request(
            prompt=[
                {"role": "system", "content": instruct},
                {"role": "user", "content": prompt},
            ],
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=eval_params,
        )

        # Calculate the average probability of "Yes" across all responses
        yes_probabilities = []
        for response in responses:
            # Get the logprobs for the first token after the prompt
            if hasattr(response, 'logprobs') and response.logprobs:
                first_token_logprobs = response.logprobs[0]
                # Look for Yes token probability
                yes_prob = next((prob for token, prob in first_token_logprobs.items() 
                               if token.lower() in ['yes', 'yes.', 'yes!']), 0.0)
                yes_probabilities.append(np.exp(yes_prob))  # Convert logprob to probability

        if yes_probabilities:
            value = sum(yes_probabilities) / len(yes_probabilities)
            value = value * 20  # Scale up the value similar to Game24
        else:
            value = 0.001

        if cache is not None:
            cache[state.current_state] = value

        return value

# Helper function
def sum_overall_scores(evaluations):
    if not evaluations:
        return 1

    values = []
    pattern = r"\b(?:overall[\s_]?score|score)\b(?:\s*(?:is|=|:|was|stands at|of))?\s*(-?\d+(?:\.\d+)?)"

    for evaluation in evaluations:
        if not isinstance(evaluation, str):
            values.append(1)
            continue

        match = re.search(pattern, evaluation, re.IGNORECASE)
        if match:
            try:
                value = float(match.group(1))
            except ValueError:
                value = 1
        else:
            value = 1

        values.append(value)

    return sum(values)