from typing import List
import re

from . import prompts as prompts
from .state import StateLogiQA
from ...typedefs import Agent, Model, DecodingParameters

class AgentActLogiQA(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StateLogiQA,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        """
        Returns actions generated for the LogiQA task.
        """
        # Format the prompt
        paragraph = state.context
        question = state.question
        choises = '\n'.join(get_choices(state))
        prompt = prompts.act.format(paragraph=paragraph, question=question, choises=choises)

        # Format the request
        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params
        )

        # Parse the response
        proposals = [r.lower().replace("answer: ", "").strip() for r in responses]
        return proposals
    

class AgentAggregateLogiQA(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StateLogiQA,
        actions: List[str],
        k: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        """
        Returns the selected actions for the LogiQA task.
        """
        # Format the prompt
        choices = '\n'.join(get_choices(state))
        actions = '\n'.join([f"Answer: {a}" for a in actions])
        prompt = prompts.aggregate.format(paragraph=state.context, question=state.question, choices=choices, k=k, actions=actions)

        # Format the request
        responses = await model.request(
            prompt=prompt,
            n=1,
            request_id=request_id,
            namespace=namespace,
            params=params
        )

        # Parse the response
        response = responses[0].strip().lower().replace("answer: ", "")
        selected_proposals = [r.strip() for r in response.split('\n')]
        return selected_proposals
    
class AgentEvaluateLogiQA(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StateLogiQA,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
        cache: dict = None,
    ) -> float:
        """
        Returns the evaluation score for the LogiQA task.
        """
        # Check if the state is already in the cache
        if cache is not None and state.current_state in cache:
            return cache[state.current_state]

        # Format the prompt
        choices = '\n'.join(get_choices(state))
        examples = "(Example)\n" + "\n\n(Example)\n".join([prompts.evaluate_examples[1:]])
        prompt = prompts.evaluate.format(examples=examples, paragraph=state.context, question=state.question, choices=choices, answer=state.current_state)

        # Format the request
        responses = await model.request(
            prompt=prompt,
            n=n,
            namespace=namespace,
            request_id=request_id,
            params=params
        )

        # Parse the response
        valuations = [r.lower().strip() for r in responses]
        mapping = {r'incorrect': 0.001, r'plausible': 1, r'correct': 10}
        value = 0
        for pattern, weight in mapping.items():
            matches = [code for code in valuations if re.search(pattern, code)]
            value += weight * len(matches)


        # Cache the value
        if cache is not None:
            cache[state.current_state] = value
        return value

#---Helper functions---
def get_choices(state: StateLogiQA) -> List[str]:
    return [state.option_a, state.option_b, state.option_c, state.option_d]