import re
from typing import List, Tuple

from . import prompts as prompts
from .state import StateHotpotQA
from ...typedefs import Agent, Model, DecodingParameters

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
        examples = "(Example)\n" + "\n\n(Example)\n".join([example for example in prompts.examples_act[:num_examples]])
        prompt = prompts.act.format(examples=examples, question=state.puzzle, current_state=state.current_state)
        
        # Generate the responses
        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params
        )

        # Parse the responses
        actions = [r.strip() for r in responses]
        return actions

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
        num_examples = 3
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

        # take actions
        # execute 

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
        for response in responses:
            try:
                value = int(re.search(r"correctness score is (\d+)", response).group(1))
            except AttributeError:
                print(f"Unable to parse value from response : {response}")
                value = 1
            values.append(value)
        value = sum(values)

        # Cache the value
        if cache is not None:
            cache[state.current_state] = value
        return value