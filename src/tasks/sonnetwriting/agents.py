from typing import List
import re

from . import prompts as prompts
from .state import StateSonnetWriting
from ...typedefs import Agent, Model, DecodingParameters

class AgentActSonnetWriting(Agent):
    """
    """
    @staticmethod
    async def act(model: Model, state: StateSonnetWriting, n: int, namespace: str, request_id: str, params: DecodingParameters) -> List[str]:
        # Format the prompt
        prompt = prompts.act.format(input=state.current_state)

        # Generate response
        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params
        )

        # Parse responses
        proposals = [r.strip() for r in responses]
        return proposals
    
class AgentAggregateSonnetWriting(Agent):
    """
    Returns the aggregate actions for the Sonnet Writing task.
    """
    @staticmethod
    async def act(model: Model, state: StateSonnetWriting, actions: List[str], k: int, namespace: str, request_id: str, params: DecodingParameters) -> List[str]:
        # Format the prompt
        seperator = '---END-OF-SONNET---'
        sonnets = f'\n\n{seperator}\n\n'.join(actions) + f'\n\n{seperator}'
        examples = prompts.aggregate_examples
        prompt = prompts.aggregate.format(task=state.puzzle, k=k, examples=examples, sonnets=sonnets)

        # Generate response
        responses = await model.request(
            prompt=prompt,
            n=1,
            request_id=request_id,
            namespace=namespace,
            params=params
        )

        # Parse responses
        selections = responses[0].split('\n')[-k:]
        proposals = [actions[int(r.strip()) - 1] for r in selections]
        return proposals
    
class AgentEvaluateSonnetWriting(Agent):
    """
    Returns the evaluations of states for the Sonnet Writing task.
    """
    @staticmethod
    async def act(model: Model, state: StateSonnetWriting, n: int, namespace: str, request_id: str, params: DecodingParameters) -> float:
        # Format prompt
        seperator ='---END-OF-SONNET---'
        examples = '(Example)\n' + '\n\n(Example)\n'.join(prompts.examples_evaluate[[0, 2, 3]])
        rhyme_scheme, words = state.target.split(', ')
        sonnet = state.current_state + f'\n {seperator}'
        prompt = prompts.evaluate.format(rhyme_scheme=rhyme_scheme, words=words, examples=examples, sonnet=sonnet)

        # Generate response
        response = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params
        )

        # Parse response
        evaluation = response[0].lower().replace("evaluation: ", "")
        return float(evaluation.strip())


