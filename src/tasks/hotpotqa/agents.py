import re
import random
from urllib import response
import numpy as np
from typing import List, Tuple

from . import prompts as prompts
from .state import StateHotpotQA
from ...typedefs import Agent, Model, DecodingParameters


class AgentActHotpotQA(Agent):
    """
    Agent performing the Act operation for the HotpotQA task.
    """

    @staticmethod
    async def act(
        model: Model,
        state: StateHotpotQA,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        """
        Returns a list of n actions for the HotpotQA task.
        """

        # Format the prompt
        num_examples = 2
        examples = "(Example)\n" + "\n\n(Example)\n".join(
            [example for example in prompts.examples_act[:num_examples]]
        )
        prompt = prompts.act.format(
            examples=examples, question=state.puzzle, current_state=state.current_state
        )

        # Generate the response
        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        proposals = [
            r.strip()
            for response in responses
            for r in reversed(response.split("\n"))
            if any(
                r.lower().startswith(action)
                for action in ["lookup", "search", "finish"]
            )
        ]
        return proposals


class AgentBfsHotpotQA(Agent):
    """
    Agent performing the BFS operation for the HotpotQA task.
    """

    @staticmethod
    async def act(
        model: Model,
        state: StateHotpotQA,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        """
        Returns a list of n actions for the HotpotQA task.
        """

        # Format the prompt
        num_examples = 2
        examples = "(Example)\n" + "\n\n(Example)\n".join(
            [example for example in prompts.examples_bfs[:num_examples]]
        )
        prompt = prompts.bfs.format(
            examples=examples, question=state.puzzle, current_state=state.current_state
        )

        # Generate the response
        response = await model.request(
            prompt=prompt,
            n=1,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        # Parse the response
        proposals = [r.strip() for r in response[0].split("\n")]
        return proposals


class AgentAggregateHotpotQA(Agent):
    """
    Agent performing the Aggregate operation for the HotpotQA task.
    """

    @staticmethod
    async def act(
        model: Model,
        state: StateHotpotQA,
        actions: List[str],
        k: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        """
        Returns a list of the k best actions for the HotpotQA task.
        """

        # if "Finish" in state.current_state:
        #     return [action for action in actions if "Finish" in action]

        # Format the prompt
        num_examples = 2
        examples = "(Example)\n" + "\n\n(Example)\n".join(
            [example for example in prompts.examples_aggregate[:num_examples]]
        )
        actions = "\n".join(action for action in actions)
        prompt = prompts.aggregate.format(
            examples=examples,
            question=state.puzzle,
            current_state=state.current_state,
            k=k,
            actions=actions,
        )

        # Generate the responses
        responses = await model.request(
            prompt=prompt,
            n=1,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        # Parse the responses
        try:
            aggregate_actions = [r.strip() for r in reversed(responses[0].split("\n")) if any(r.lower().startswith(action) for action in ["lookup", "search", "finish"])]
        except:
            aggregate_actions = []
        return aggregate_actions


class AgentReactHotpotQA(Agent):
    """
    Agent performing the ReAct operation for the HotpotQA task.
    """

    @staticmethod
    async def act(
        model: Model,
        state: StateHotpotQA,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[Tuple[str, str]]:
        """
        Returns a list of n thought-action pairs for the HotpotQA task.
        """

        # Format the prompt
        num_examples = 2
        examples = "(Example)\n" + "\n\n(Example)\n".join(
            [example for example in prompts.examples_react[:num_examples]]
        )
        prompt = prompts.react.format(
            examples=examples, question=state.puzzle, current_state=state.current_state
        )

        # Generate the responses
        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        # Parse the responses
        react_actions = [r.strip() for r in responses]
        return react_actions


class AgentSelfEvaluateHotpotQA(Agent):
    """
    Agent that performs self-evaluation of reasoning steps for HotpotQA.
    Uses the LLM's own estimation of correctness by evaluating each reasoning step.
    Uses the probability of "Yes" as a reward signal for correct reasoning.
    """

    @staticmethod
    async def act(
        model: Model,
        state: StateHotpotQA,
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
        if state.steps and "Finish" in state.steps[-1]:
            # Evaluating a final answer
            answer = state.steps[-1].replace("Finish[", "").replace("]", "")
            prompt = prompts.self_evaluate_answer.format(
                question=state.puzzle, steps="\n".join(state.steps), answer=answer
            )
        else:
            # Evaluating intermediate reasoning steps
            last_step = state.steps[-1] if state.steps else ""
            prompt = prompts.self_evaluate_step.format(
                current_state=state.current_state, step=last_step
            )

        eval_params = DecodingParameters(
            temperature=params.temperature,
            max_completion_tokens=params.max_completion_tokens,
            top_p=params.top_p,
            stop=params.stop,
            logprobs=True,
        )

        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=eval_params,
        )

        # Calculate the average probability of "Yes" across all responses
        yes_probabilities = []
        for response in responses:
            # Get the logprobs for the first token after the prompt
            if hasattr(response, "logprobs") and response.logprobs:
                first_token_logprobs = response.logprobs[0]
                # Look for Yes token probability
                yes_prob = next(
                    (
                        prob
                        for token, prob in first_token_logprobs.items()
                        if token.lower() in ["yes", "yes.", "yes!"]
                    ),
                    0.0,
                )
                yes_probabilities.append(np.exp(yes_prob))

        if yes_probabilities:
            value = sum(yes_probabilities) / len(yes_probabilities)
            value = value * 20
        else:
            value = 0.001

        if cache is not None:
            cache[state.current_state] = value

        return value


class AgentEvaluateHotpotQA(Agent):
    """
    Agent performing the Evaluate operation for the HotpotQA task.
    """

    @staticmethod
    async def act(
        model: Model,
        state: StateHotpotQA,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
        cache: dict = None,
    ) -> float:
        """
        Returns an evaluations for the HotpotQA task.
        """
        # Check if the state is already in the cache
        if cache is not None and state.current_state in cache:
            return cache[state.current_state]

        # Format the prompt
        num_examples = 2
        examples = "(Example)\n" + "\n\n(Example)\n".join(
            [example for example in prompts.examples_evaluate[:num_examples]]
        )
        prompt = prompts.evaluate.format(
            examples=examples, question=state.puzzle, current_state=state.current_state
        )

        # Generate the responses
        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        # Parse the responses
        values = []
        pattern = r"\b(?:correctness[\s_]?score|score for correctness|correctness)\b(?:\s*(?:is|=|:|was|stands at|of))?\s*(-?\d+(?:\.\d+)?)"

        for response in responses:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                value = float(match.group(1))
            else:
                # print(f"Unable to parse value from response : {response}")
                value = 1
            values.append(value)
        value = sum(values)

        # Cache the value
        if cache is not None:
            cache[state.current_state] = value
        return value
