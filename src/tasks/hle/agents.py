import re
from typing import List, Tuple

from . import prompts as prompts
from .state import StateHLE
from ...typedefs import Agent, Model, DecodingParameters

class AgentActHLE(Agent):
    """
    Agent performing the Act operation for the HLE task.
    """

    @staticmethod
    async def act(model: Model, state: StateHLE, n: int, namespace: str, request_id: str, params: DecodingParameters) -> List[str]:
        """
        Returns a list of n actions for the HLE task.
        """

        # Format the prompt
        num_examples = 2
        examples = "(Example)\n" + "\n\n(Example)\n".join([example for example in prompts.examples_act[:num_examples]])
        prompt = prompts.act.format(examples=examples, question=state.question, current_state=state.serialize())

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
    
class AgentAggregateHLE(Agent):
    """
    Agent performing the Aggregate operation for the HLE task.
    """

    @staticmethod
    async def act(model: Model, state: StateHLE, actions: List[str], k: int, namespace: str, request_id: str, params: DecodingParameters) -> List[str]:
        """
        Returns a list of k selected actions for the HLE task.
        """
        # Format the prompt
        prompt = prompts.aggregate.format(k=k, question=state.question, actions='\n'.join(actions))

        # Generate the responses
        responses = await model.request(
            prompt=prompt,
            n=1,
            request_id=request_id,
            namespace=namespace,
            params=params
        )

        # Parse the responses
        selected_actions = [r.strip() for r in responses[0].split("\n")]
        return selected_actions

class AgentBfsHLE(Agent):
    """
    Agent performing the BFS operation for the HLE task.
    """

    @staticmethod
    async def act(model: Model, state: StateHLE, namespace: str, request_id: str, params: DecodingParameters) -> List[str]:
        """
        Returns a list of n actions for the HLE task.
        """

        # Format the prompt
        num_examples = 3
        examples = "(Example)\n" + "\n\n(Example)\n".join([example for example in prompts.examples_bfs[:num_examples]])
        prompt = prompts.bfs.format(examples=examples, question=state.question, current_state=state.serialize())

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

class AgentReactHLE(Agent):
    """
    Agent performing the ReAct operation for the HLE task.
    """

    @staticmethod
    async def act(model: Model, state: StateHLE, n: int, namespace: str, request_id: str, params: DecodingParameters) -> List[Tuple[str, str]]:
        """
        Returns a list of n thought-action pairs for the HLE task.
        """

        # Format the prompt
        num_examples = 2
        examples = "(Example)\n" + "\n\n(Example)\n".join([example for example in prompts.examples_react[:num_examples]])
        prompt = prompts.react.format(examples=examples, question=state.question, current_state=state.serialize())

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

class AgentEvaluateHLE(Agent):
    """
    Agent performing the Evaluate operation for the HLE task.
    """

    @staticmethod
    async def act(model: Model, state: StateHLE, n: int, namespace: str, request_id: str, params: DecodingParameters, cache: dict = None) -> float:
        """
        Returns an evaluation for the HLE task.
        """
        # Check if the state is already in the cache
        if cache is not None and state.serialize() in cache:
            return cache[state.serialize()]

        # Format the prompt
        num_examples = 2
        examples = "(Example)\n" + "\n\n(Example)\n".join([example for example in prompts.examples_evaluate[:num_examples]])
        prompt = prompts.evaluate.format(examples=examples, question=state.question, current_state=state.serialize())

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
                print(f"Unable to parse value from response: {response}")
                value = 1
            values.append(value)
        value = sum(values)

        # Cache the value
        if cache is not None:
            cache[state.serialize()] = value
        return value