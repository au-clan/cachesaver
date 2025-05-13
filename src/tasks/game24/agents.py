import math
from typing import List
import re

from . import prompts as prompts
from .state import StateGame24
from ...typedefs import Request, Agent, Model, DecodingParameters

class AgentActGame24(Agent):
    """
    """
    async def act(model: Model, state: StateGame24, n: int, namespace: str, request_id: str, params: DecodingParameters) -> List[str]:
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
        proposals = [r.split(")")[0] + ")" for r in proposals]
        return proposals
    
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
        for r in response:
            print(r)

        # Parse the response
        if state.current_state != "24":
            response = [response[0].rpartition(")")[0] + ")"]
        proposals = [r.strip() for r in response[0].split("\n")]
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
        for r in responses:
            print(r)
            print("===")

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

    @staticmethod
    async def self_eval_reward(model: Model, state: StateGame24, action: str, next_state: StateGame24, num_evaluations: int,namespace: str, request_id: str, eval_params: DecodingParameters) -> float:
        responses = await model.request(
            prompt=prompts.self_evaluate_rap.format(state_before=state.current_state, action=action, state_after=next_state.current_state),
            namespace=namespace,
            n=num_evaluations,
            request_id=request_id,
            params=eval_params,
        )
        yes_count = sum(1 for r in responses if r.strip().lower().startswith("yes"))
        score = yes_count / num_evaluations
        return _scale_prob_reward(score, 20)

    @staticmethod
    async def logprobs_reward(logprobs_model: Model, state: StateGame24, action: str, next_state: StateGame24, namespace: str, request_id: str, eval_params: DecodingParameters) -> float:
        if logprobs_model is None:
            return 0
        logprobs_eval_params = eval_params
        logprobs_eval_params.logprobs = True
        responses = await logprobs_model.request(
            prompt=prompts.self_evaluate_rap.format(state_bedore=state.current_state, action=action, state_after=next_state.current_state),
            namespace=namespace,
            n=1,
            request_id=request_id,
            params=logprobs_eval_params,
            return_logprobs=True
        )
        _, token_logprobs = responses[0]
        label, logprob = token_logprobs[0]
        if label == 'yes':
            return _scale_logprob_reward(logprob, 20)
        elif label == 'no':
            return _scale_logprob_reward(logprob, 20, inverse=True)
        return 0


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

def _scale_prob_reward(p, max_value, inverse=False):
    if inverse:
        p = 1 - p

    p = min(max(p, 1e-10), 1.0)
    if p <= 0:
        return 0  # Impossible
    if p >= 1:
        return max_value  # Certain

    logit = math.log(p / (1 - p))

    # Clamp logit values to avoid extreme outputs
    min_logit = -7
    max_logit = 7
    logit = max(-7, min(7, logit))

    # Normalize and scale to [0, max_value]
    scaled = (logit - min_logit) / (max_logit - min_logit) * max_value
    return scaled

def _scale_logprob_reward(logprob, max_value, inverse=False):
    p = math.exp(logprob)
    return _scale_prob_reward(p, max_value, inverse=inverse)
