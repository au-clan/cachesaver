from typing import List
import re
import numpy as np

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
        actions = '\n'.join([f"({i+1}) Answer: {a}" for i, a in enumerate(actions)])
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
        try:
            indexes = [int(i.strip()) - 1 for i in re.findall(r'\d+', responses[0])]
            selected_proposals = [actions[i] for i in indexes if i < len(actions)]
        except:
            selected_proposals = []
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
        examples = "(Example)\n" + "\n\n(Example)\n".join(prompts.evaluate_examples[1:])
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

class AgentReactLogiQA(Agent):
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
        Returns actions generated for the LogiQA task using ReAct-style reasoning.
        """
        # Format the prompt
        choices = '\n'.join(get_choices(state))
        prompt = prompts.react.format(
            paragraph=state.context,
            question=state.question,
            choices=choices,
            current_state=state.current_state
        )

        # Format the request
        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params
        )

        # Parse the response to extract actions
        proposals = []
        for response in responses:
            # Look for the answer after the last "---"
            if "---" in response:
                answer = response.split("---")[-1].strip()
                if answer.lower().startswith("answer: "):
                    answer = answer.lower().replace("answer: ", "").strip()
                    if answer in ['a', 'b', 'c', 'd']:
                        proposals.append(answer)
            else:
                # If no "---" found, try to extract from the last line
                last_line = response.strip().split('\n')[-1].lower()
                if last_line.startswith('answer: '):
                    answer = last_line.replace('answer: ', '').strip()
                    if answer in ['a', 'b', 'c', 'd']:
                        proposals.append(answer)

        return proposals

class AgentSelfEvaluateLogiQA(Agent):
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
        Returns the evaluation score for the LogiQA task using self-evaluation.
        """
        # Check if the state is already in the cache
        if cache is not None and state.current_state in cache:
            return cache[state.current_state]

        # Format the prompt based on whether we're evaluating a final answer or intermediate steps
        choices = '\n'.join(get_choices(state))
        if state.current_state in ['a', 'b', 'c', 'd']:
            # Evaluating final answer
            prompt = prompts.self_evaluate_answer.format(
                paragraph=state.context,
                question=state.question,
                choices=choices,
                steps=state.steps,
                answer=state.current_state
            )
        else:
            # Evaluating intermediate step
            prompt = prompts.self_evaluate_step.format(
                paragraph=state.context,
                question=state.question,
                choices=choices,
                previous_steps=state.current_state,
                step=state.current_state
            )

        # Format the request
        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params
        )

        # Calculate average probability of "Yes"
        yes_probs = []
        for response in responses:
            response = response.strip().lower()
            if response == 'yes':
                yes_probs.append(1.0)
            elif response == 'no':
                yes_probs.append(0.0)
            else:
                # If response is not clear, use 0.5 as default
                yes_probs.append(0.5)
        
        value = np.mean(yes_probs)

        # Scale the value to match the evaluation range
        if state.current_state in ['a', 'b', 'c', 'd']:
            # For final answers, scale to match the evaluation range (0.001 to 10)
            value = 0.001 + (10 - 0.001) * value
        else:
            # For intermediate steps, use a smaller range (0 to 1)
            value = value

        # Cache the value
        if cache is not None:
            cache[state.current_state] = value
        return value

#---Helper functions---
def get_choices(state: StateLogiQA) -> List[str]:
    return [state.option_a, state.option_b, state.option_c, state.option_d]