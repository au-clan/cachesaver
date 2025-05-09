import re
import random
from typing import List

from . import prompts as prompts
from .state import StateGame24
from ...typedefs import Request, Agent, Model, DecodingParameters

act_cache = {}

class AgentActGame24(Agent):
    """
    """
    async def act(model: Model, state: StateGame24, n: int, namespace: str, request_id: str, params: DecodingParameters) -> List[str]:
        # Format the prompt
        if state.current_state == "24":
            prompt = prompts.cot.format(input=state.puzzle) + "\nSteps:\n" + '\n'.join(state.steps) + "\nAnswer: "
        else:
            current_numbers = get_current_numbers(state)
            prompt = prompts.bfs.format(input=current_numbers)

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
            if state.current_state != "24":
                response = [response[0].rpartition(")")[0] + ")"]
            proposals.extend(r.strip() for r in response[0].split("\n"))
            if 'Possible next steps:' in proposals:
                    proposals.remove('Possible next steps:')
        random.seed(state.randomness)
        random.shuffle(proposals)
        act_cache[prompt].extend(proposals[n:])
        return proposals[:n]
    
class AgentAggregateGame24(Agent):

    @staticmethod
    async def act(model: Model, state: StateGame24, actions: List[str], k: int, namespace: str, request_id: str, params: DecodingParameters) -> List[str]:
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
    async def act(model: Model, state: StateGame24, namespace: str, request_id: str, params: DecodingParameters) -> List[str]:
        """
        Returns a list of actions for the Game of 24 task.
        """
        # Format the prompt
        if len(state.current_state.strip().split(' ')) == 1:
            prompt = prompts.cot.format(input=state.puzzle) + "\nSteps:\n" + '\n'.join(state.steps).strip() + "\nAnswer: "

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
        if 'Possible next steps:' in proposals:
                proposals.remove('Possible next steps:')
        return proposals


class AgentEvaluateGame24(Agent):

    @staticmethod
    async def act(model: Model, state: StateGame24, n: int,namespace: str, request_id: str, params: DecodingParameters, cache: dict=None) -> float:
        """
        Returns a value for the given state
        """

        # Check if the state is already in the cache
        if cache is not None and state.current_state in cache:
            return cache[state.current_state]

        # Format the prompt
        if state.steps and "left" not in state.steps[-1]:
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
        codes = [r.split('\n')[-1].lower().strip() for r in responses]
        code_map = {r'impossible': 0.001, r'likely': 1, r'sure': 20}
        value = 0
        for pattern, weight in code_map.items():
            matches = [code for code in codes if re.search(pattern, code)]
            value += weight * len(matches)

        # Cache the value
        if cache is not None:
            cache[state.current_state] = value
        return value


class AgentReactGame24(Agent):
    """
    Agent for React algorithm
    """
    @staticmethod
    async def act(model: Model, state: StateGame24, n: int, namespace: str, request_id: str, params: DecodingParameters) -> List[str]:
        if state.current_state == "24":
            prompt = prompts.cot.format(input=state.puzzle) + "\nSteps:\n" + '\n'.join(state.steps) + "\nAnswer: "
        else:
            current_numbers = get_current_numbers(state)
            prompt = prompts.react.format(input=current_numbers)

        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params
        )

        proposals = [r.strip() for r in responses]
        return proposals


def get_current_numbers(state: StateGame24) -> str:
    """
    Returns the current numbers in the state.
    """
    last_line = state.current_state.strip().split('\n')[-1]
    return last_line.split('left: ')[-1].split(')')[0]

def get_formula(state: StateGame24) -> str:
    if state.steps:
        formula = state.steps[-1].lower().replace("answer: ", "")
        return formula
    else:
        # Should do some error handling here but for the moment we'll take it as it is
        return ""