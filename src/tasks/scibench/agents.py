import re
import numpy as np
from typing import List

from . import prompts as prompts
from .state import StateSciBench
from ... import AgentFactory
from ...typedefs import Agent, Model, DecodingParameters


@AgentFactory.register
class AgentActSciBench(Agent):

    @staticmethod
    async def act(
        model: Model,
        state: StateSciBench,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        """
        Returns the action for the SciBench task.
        """
        # Format the prompt
        existing_steps = "\n".join(state.steps) if len(state.steps) > 0 else "None\n"
        if (len(state.values) > 0 and state.values[max(state.values)] >= 0.9) or (
            len(state.steps) > 0 and "answer is" in state.steps[-1].lower()
        ):  # some hacky stuff from rest-mcts*
            prompt = prompts.summary.format(
                problem=state.puzzle, existing_steps=existing_steps
            )
        else:

            prompt = prompts.act.format(
                problem=state.puzzle, existing_steps=existing_steps
            )

        # Generate the response
        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        # Parse the response
        proposals = [r.strip().split("\n")[:5] for r in responses]
        proposals = [parse_proposal(r, state.step_n, existing_steps) for r in proposals]
        return proposals


@AgentFactory.register
class AgentReactSciBench(Agent):

    @staticmethod
    async def act(
        model: Model,
        state: StateSciBench,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        """
        Returns the action for the SciBench task.
        """
        # Format the prompt
        existing_steps = "\n".join(state.steps) if len(state.steps) > 0 else "None\n"
        if (len(state.values) > 0 and state.values[max(state.values)] >= 0.9) or (
            len(state.steps) > 0 and "answer is" in state.steps[-1].lower()
        ):  # some hacky stuff from rest-mcts*
            prompt = prompts.summary.format(
                problem=state.puzzle, existing_steps=existing_steps
            )
        else:

            prompt = prompts.react.format(
                problem=state.puzzle, existing_steps=existing_steps
            )

        # Generate the response
        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        # Parse the response
        proposals = [r.strip().split("\n")[:5] for r in responses]
        proposals = [parse_proposal(r, state.step_n, existing_steps) for r in proposals]
        return proposals


@AgentFactory.register
class AgentBfsSciBench(Agent):

    @staticmethod
    async def act(
        model: Model,
        state: StateSciBench,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        """
        Returns the action for the SciBench task.
        """
        # Format the prompt
        existing_steps = "\n".join(state.steps) if len(state.steps) > 0 else "None\n"
        if (len(state.values) > 0 and state.values[max(state.values)] >= 0.9) or (
            len(state.steps) > 0 and "answer is" in state.steps[-1].lower()
        ):  # some hacky stuff from rest-mcts*
            prompt = prompts.summary.format(
                problem=state.puzzle, existing_steps=existing_steps
            )
        else:

            prompt = prompts.bfs.format(
                problem=state.puzzle, existing_steps=existing_steps
            )

        # Generate the response
        responses = await model.request(
            prompt=prompt,
            n=1,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )
        # Parse the response
        proposals = [
            "Next step: " + step.strip()
            for step in responses[0].split("Next step:")
            if step.strip()
        ]
        proposals = [r.strip().split("\n")[:5] for r in proposals]
        proposals = [parse_proposal(r, state.step_n, existing_steps) for r in proposals]
        return proposals


@AgentFactory.register
class AgentAggregateSciBench(Agent):

    @staticmethod
    async def act(
        model: Model,
        state: StateSciBench,
        actions: List[str],
        k: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        """
        Returns the aggregated action for SciBench task.
        """
        if len(actions) == 0:
            return []
        
        if (len(state.values) > 0 and state.values[max(state.values)] >= 0.9) or (
            len(state.steps) > 0 and "answer is" in state.steps[-1].lower()
        ):  # some hacky stuff from rest-mcts*
            return actions

        # Format the prompt
        steps = "\n".join(actions)
        prompt = prompts.aggregate.format(problem=state.puzzle, k=k, steps=steps)

        # Generate the response
        responses = await model.request(
            prompt=prompt,
            n=1,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        # Parse the response
        try:
            indexes = [int(i.strip()) - 1 for i in re.findall(r"\d+", responses[0])]
            out = [actions[i] for i in indexes if i < len(actions)]
        except Exception as e:
            out = []
        return out


@AgentFactory.register
class AgentEvaluateSciBench(Agent):

    @staticmethod
    async def act(
        model: Model,
        state: StateSciBench,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
        cache: dict = None,
    ) -> float:

        # Check if the state is already in the cache
        if cache is not None and state.current_state in cache:
            value = cache[state.current_state]

        else:
            # Format the promp
            num_examples = 2
            examples = "Example:\n" + "\n\nExample:\n".join(
                [example for example in prompts.examples_evaluate[:num_examples]]
            )
            prompt = prompts.evaluate.format(
                examples=examples,
                problem=state.puzzle,
                existing_steps="\n".join(state.steps),
            )

            # Generate the response
            responses = await model.request(
                prompt=prompt,
                n=n,
                request_id=request_id,
                namespace=namespace,
                params=params,
            )

            # Parse the response
            values = [parse_value(r) for r in responses]
            value = sum(values) / len(values)

            # Cache the value
            if cache is not None:
                cache[state.current_state] = value
            state.values[state.step_n] = value
        return value

@AgentFactory.register
class AgentSelfEvaluateSciBench(Agent):
    """
    Agent that performs self-evaluation of reasoning steps for HotpotQA.
    Uses the LLM's own estimation of correctness by evaluating each reasoning step.
    Uses the probability of "Yes" as a reward signal for correct reasoning.
    """

    @staticmethod
    async def act(
        model: Model,
        state: StateSciBench,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
        cache: dict=None,        
    ) -> float:
        """
        Returns a value estimation for the current state based on self-evaluation.
        """

        if cache is not None and state.current_state in cache:
            value = cache[state.current_state]
        
        if len(state.steps) == 0:
            print("No steps to evaluate!")
            return 0.0
        
        if (len(state.steps) > 0) and ("the final answer is" in state.steps[-1].lower()):
            prompt = prompts.self_evaluate_answer.format(input=state.puzzle, answer=state.steps[-1])
        else:

            prompt = prompts.self_evaluate_step.format(input=state.puzzle, previous_steps=state.steps[:-1] if len(state.steps) > 1 else "None", step=state.steps[-1])

        evaluate_params = DecodingParameters(
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
            params=evaluate_params,
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




# ---Helper functions---#
def parse_proposal(response: List[str], step_n: int, existing_steps: str) -> str:
    p = ""
    for _ in response:
        p = p + _ + " "
    p = p.strip()

    if "Next step:" in p:
        stp = p.split("Next step:")[1].strip()
        if len(stp) < 2:
            # print('Output step too short!\n')
            return ""
        if stp in existing_steps:
            # print('Output step repeated!\n')
            return ""

        revised_ = "Step " + str(step_n) + ": " + stp

    elif "Step" in p and ":" in p:
        pre_len = len(p.split(":")[0])
        p_ = p[pre_len:]
        p_ = p_.split("Step")[0].strip()
        if len(p_) < 4:
            # print('Output step too short!\n')
            return ""
        p_ = p_[1:].strip()
        if p_ in existing_steps:
            # print('Output step repeated!\n')
            return ""

        revised_ = "Step " + str(step_n) + ": " + p_

    else:
        p_ = p.strip()
        if len(p_) < 3:
            # print('Output step too short!\n')
            return ""
        if p_ in existing_steps:
            # print('Output step repeated!\n')
            return ""

        revised_ = "Step " + str(step_n) + ": " + p_
    revised = revised_ + "\n"
    return revised


def parse_value(response: str, low=0.0, high=1.0) -> float:
    out_value = low

    # score expected in output
    if "score" not in response.lower():
        return out_value

    response = response.lower().split("score")[-1].strip()
    try:
        match = re.findall(r"-?[0-9]+\.?[0-9]*", response)[-1]
        out_value = float(match)
        out_value = min(max(low, out_value), high)
    except Exception as e:
        out_value = low
    return out_value
