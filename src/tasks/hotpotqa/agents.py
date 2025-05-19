import re
import random
from typing import List, Tuple

from . import prompts as prompts
from .state import StateHotpotQA
from ..utils import scale_prob_reward_linear, \
    scale_logprob_reward_linear
from ...typedefs import Agent, Model, DecodingParameters

act_cache = {}

class AgentActHotpotQA(Agent):
    """
    Agent performing the Act operation for the HotpotQA task.
    """

    @staticmethod
    async def act(model: Model, state: StateHotpotQA, n: int, namespace: str, request_id: str, params: DecodingParameters) -> List[str]:
        """
        Returns a list of n actions for the HotpotQA task.
        """

        # Format the prompt
        num_examples = 2
        examples = "(Example)\n" + "\n\n(Example)\n".join([example for example in prompts.examples_bfs[:num_examples]])
        prompt = prompts.bfs.format(examples=examples, question=state.puzzle, current_state=state.current_state)

        if prompt in act_cache:
            proposals = act_cache[prompt][:n]
            act_cache[prompt] = act_cache[prompt][n:]
        else:
            proposals = []
            act_cache[prompt] = []

        while len(proposals) < n:
            
            # Generate the response
            response = await model.request(
                prompt=prompt,
                n=1,
                request_id=request_id,
                namespace=namespace,
                params=params
            )

            # Parse the response
            proposals.extend(r.strip() for r in response[0].split("\n"))
        random.seed(state.randomness)
        random.shuffle(proposals)
        act_cache[prompt].extend(proposals[n:])
        return proposals[:n]
        

class AgentBfsHotpotQA(Agent):
    """
    Agent performing the BFS operation for the HotpotQA task.
    """

    @staticmethod
    async def act(model: Model, state: StateHotpotQA, namespace: str, request_id: str, params: DecodingParameters) -> List[str]:
        """
        Returns a list of n actions for the HotpotQA task.
        """

        # Format the prompt
        num_examples = 2
        examples = "(Example)\n" + "\n\n(Example)\n".join([example for example in prompts.examples_bfs[:num_examples]])
        prompt = prompts.bfs.format(examples=examples, question=state.puzzle, current_state=state.current_state)

        # Generate the response
        response = await model.request(
            prompt=prompt,
            n=1,
            request_id=request_id,
            namespace=namespace,
            params=params
        )

        # Parse the response
        proposals = [r.strip() for r in response[0].split("\n")]
        return proposals

class AgentAggregateHotpotQA(Agent):
    """
    Agent performing the Aggregate operation for the HotpotQA task.
    """

    @staticmethod
    async def act(model: Model, state: StateHotpotQA, actions: List[str], k: int, namespace: str, request_id: str, params: DecodingParameters) -> List[str]:
        """
        Returns a list of the k best actions for the HotpotQA task.
        """

        if any("Finish" in action for action in actions):
            return [action for action in actions if "Finish" in action]

        # Format the prompt
        num_examples = 2
        examples = "(Example)\n" + "\n\n(Example)\n".join([example for example in prompts.examples_aggregate[:num_examples]])
        actions = "\n".join(action for action in actions)
        prompt = prompts.aggregate.format(examples=examples, question=state.puzzle, current_state=state.current_state, k=k, actions=actions)

        # Generate the responses
        responses = await model.request(
            prompt=prompt,
            n=1,
            request_id=request_id,
            namespace=namespace,
            params=params
        )

        # Parse the responses
        aggregate_actions = [r.strip() for response in responses for r in response.split("\n")]
        return aggregate_actions
    
class AgentReactHotpotQA(Agent):
    """
    Agent performing the ReAct operation for the HotpotQA task.
    """

    @staticmethod
    async def act(model: Model, state: StateHotpotQA, n:int, namespace: str, request_id: str, params: DecodingParameters) -> List[Tuple[str, str]]:
        """
        Returns a list of n thought-action pairs for the HotpotQA task.
        """

        # Format the prompt
        num_examples = 2
        examples = "(Example)\n" + "\n\n(Example)\n".join([example for example in prompts.examples_react[:num_examples]])
        prompt = prompts.react.format(examples=examples, question=state.puzzle, current_state=state.current_state)

        # Generate the responses
        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params
        )

        # Parse the responses
        react_actions = [r.strip() for r in responses]
        return react_actions
    
class AgentEvaluateHotpotQA(Agent):
    """
    Agent performing the Evaluate operation for the HotpotQA task.
    """

    @staticmethod
    async def act(model: Model, state: StateHotpotQA, n: int, namespace: str, request_id: str, params: DecodingParameters, cache: dict=None) -> float:
        """
        Returns an evaluations for the HotpotQA task.
        """
        # Check if the state is already in the cache
        if cache is not None and state.current_state in cache:
            return cache[state.current_state]

        # Format the prompt
        num_examples = 2
        examples = "(Example)\n" + "\n\n(Example)\n".join([example for example in prompts.examples_evaluate[:num_examples]])
        prompt = prompts.evaluate.format(examples=examples, question=state.puzzle, current_state=state.current_state)

        # Generate the responses
        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params
        )

        # Parse the responses
        values = []
        pattern = r"\b(?:correctness[\s_]?score|score for correctness|correctness)\b(?:\s*(?:is|=|:|was|stands at|of))?\s*(-?\d+(?:\.\d+)?)"
        
        for response in responses:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                value = float(match.group(1))
            else:
                #print(f"Unable to parse value from response : {response}")
                value = 1
            values.append(value)
        value = sum(values)

        # Cache the value
        if cache is not None:
            cache[state.current_state] = value
        return value

    @staticmethod
    async def self_eval_reward(model: Model, state: StateHotpotQA, action: str, next_state: StateHotpotQA,
                               num_evaluations: int, namespace: str, request_id: str,
                               eval_params: DecodingParameters) -> float:
        num_examples = 2
        examples = "(Example)\n" + "\n\n(Example)\n".join(
            [example for example in prompts.examples_evaluate_self_eval[:num_examples]])
        responses = await model.request(
            prompt=prompts.self_evaluate_rap.format(examples=examples, state_before=state.current_state, action=action,
                                                    state_after=next_state.current_state),
            namespace=namespace,
            n=num_evaluations,
            request_id=request_id,
            params=eval_params,
        )
        yes_count = sum(1 for r in responses if r.strip().lower().startswith("yes"))
        score = yes_count / num_evaluations
        return scale_prob_reward_linear(score, num_evaluations)

    @staticmethod
    async def logprobs_reward(logprobs_model: Model, state: StateHotpotQA, action: str, next_state: StateHotpotQA,
                              num_evaluations: int, namespace: str, request_id: str, eval_params: DecodingParameters) -> float:
        if logprobs_model is None or not eval_params.logprobs:
            return 0
        num_examples = 2
        examples = "(Example)\n" + "\n\n(Example)\n".join(
            [example for example in prompts.examples_evaluate_self_eval[:num_examples]])
        responses = await logprobs_model.request(
            prompt=prompts.self_evaluate_rap.format(examples=examples, state_before=state.current_state, action=action,
                                                    state_after=next_state.current_state),
            namespace=namespace,
            n=1,
            request_id=request_id,
            params=eval_params,
            return_logprobs=True
        )
        _, token_logprobs = responses
        label, logprob = token_logprobs[0][0]
        if label.strip().lower() == 'yes':
            return scale_logprob_reward_linear(logprob, num_evaluations)
        elif label.strip().lower() == 'no':
            return scale_logprob_reward_linear(logprob, num_evaluations, inverse=True)
        return 0