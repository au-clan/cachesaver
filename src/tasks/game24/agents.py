import re
from typing import List

from . import prompts as prompts
from .state import StateGame24
from ...typedefs import Agent, Model, DecodingParameters


# Helper functions
def get_current_numbers(state: StateGame24) -> str:
    """
    Returns the current numbers in the state.
    """
    last_line = state.current_state.strip().split('\n')[-1]
    return last_line.split('left: ')[-1].split(')')[0]


def get_formula(state: StateGame24) -> str:
    formula = state.steps[-1].lower().replace("answer: ", "")
    return formula


class AgentActGame24(Agent):
    """
    """

    async def act(model: Model, state: StateGame24, n: int, namespace: str, request_id: str,
                  params: DecodingParameters) -> List[str]:
        # Format the prompt
        if state.current_state == "24":
            prompt = prompts.cot.format(input=state.puzzle) + "\nSteps:\n" + '\n'.join(state.steps) + "\nAnswer: "
        else:
            current_numbers = get_current_numbers(state)
            prompt = prompts.act.format(input=current_numbers)

        # Generate the response
        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params
        )

        # Parse the response
        proposals = [r.strip() for r in responses]
        return proposals


class AgentAggregateGame24(Agent):

    @staticmethod
    async def act(model: Model, state: StateGame24, actions: List[str], k: int, namespace: str, request_id: str,
                  params: DecodingParameters) -> List[str]:
        """
        Returns the aggregated actions for the Game of 24 task.
        """
        if any("left" not in action for action in actions):
            return [action for action in actions if "left" not in action]

        # Format the prompt
        proposals = ''
        for idx, action in enumerate(actions):
            proposals += f'({idx + 1}) ' + action + '\n'

        prompt = prompts.aggregate.format(state=state.current_state, proposal=proposals, n_select_sample=k)

        responses = await model.request(
            prompt=prompt,
            n=1,
            request_id=request_id,
            namespace=namespace,
            params=params
        )

        # Parse the response
        pattern = r"\(\d+\)\s(\d+ [+\-*/] \d+ = \d+ \(left: [^)]+\))"
        matchs = re.findall(pattern, responses[0])

        proposal = [match.strip() for match in matchs]
        return proposal


class AgentBfsGame24(Agent):

    @staticmethod
    async def act(model: Model, state: StateGame24, namespace: str, request_id: str, params: DecodingParameters) -> \
            List[str]:
        """
        Returns a list of actions for the Game of 24 task.
        """

        # Format the prompt
        if len(state.current_state.strip().split(' ')) == 1:
            prompt = prompts.cot.format(input=state.puzzle) + "\nSteps:\n" + '\n'.join(state.steps) + "\nAnswer: "
        else:
            current_numbers = get_current_numbers(state)
            prompt = prompts.bfs.format(input=current_numbers)

        # Generate the response
        response = await model.request(
            prompt=prompt,
            n=1,
            request_id=request_id,
            namespace=namespace,
            params=params
        )

        # Parse the response
        if state.current_state != "24":
            response = [response[0].rpartition(")")[0] + ")"]
        proposals = [r.strip() for r in response[0].split("\n")]
        return proposals


class AgentEvaluateGame24(Agent):

    @staticmethod
    async def act(model: Model, state: StateGame24, n: int, namespace: str, request_id: str, params: DecodingParameters,
                  cache: dict = None) -> float:
        """
        Returns a value for the given state
        """

        # Check if the state is already in the cache
        if cache is not None and state.current_state in cache:
            return cache[state.current_state]

        # Format the prompt
        if "left" not in state.steps[-1]:
            formula = get_formula(state)
            prompt = prompts.evaluate_answer.format(input=state.puzzle, answer=formula)
        else:
            prompt = prompts.evaluate.format(input=state.current_state)

        # Format the request
        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params
        )

        # Parse the response
        codes = [r.split('\n')[-1].lower() for r in responses]
        code_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}
        value = sum(value * codes.count(code) for code, value in code_map.items())

        # Cache the value
        if cache is not None:
            cache[state.current_state] = value
        return value
