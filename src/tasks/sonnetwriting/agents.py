from typing import List
import numpy as np
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
        try:
            proposals = [actions[int(i) - 1] for i in re.findall(r'\d+', responses[0])]
        except:
            proposals = []
        return proposals
    
class AgentEvaluateSonnetWriting(Agent):
    """
    Returns the evaluations of states for the Sonnet Writing task.
    """
    @staticmethod
    async def act(model: Model, state: StateSonnetWriting, n: int, namespace: str, request_id: str, params: DecodingParameters, cache: dict = None) -> float:
        
        # Check if the state is already in the cache
        if cache is not None and state.current_state in cache:
            return cache[state.current_state]
        
        # Format prompt
        seperator ='---END-OF-SONNET---'
        examples = '(Example)\n' + '\n\n(Example)\n'.join(prompts.examples_evaluate[1:])
        rhyme_scheme, words = state.target.split(', ')
        sonnet = state.current_state + f'\n {seperator}'
        prompt = prompts.evaluate.format(rhyme_scheme=rhyme_scheme, words=words, examples=examples, sonnet=sonnet)

        # Generate response
        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params
        )

        # Parse response
        evaluations = [r.lower().replace("evaluation: ", "").strip() for r in responses]
        value = sum([int(e) for e in evaluations]) / n
        
        # Cache the value
        if cache is not None:
            cache[state.current_state] = value
        return value

class AgentReactSonnetWriting(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StateSonnetWriting,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        """
        Returns actions generated for the SonnetWriting task using ReAct-style reasoning.
        """
        # Format the prompt
        prompt = prompts.react.format(
            input=state.puzzle,
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
        actions = []
        for response in responses:
            # Look for the sonnet after the last "---"
            if "---" in response:
                sonnet = response.split("---")[-1].strip()
                if sonnet:
                    # Add separator to the sonnet if not already present
                    if not sonnet.endswith("---END-OF-SONNET---"):
                        sonnet = sonnet + "\n---END-OF-SONNET---"
                    actions.append(sonnet)
            else:
                # If no "---" found, try to extract the sonnet from the last complete sonnet
                lines = response.strip().split('\n')
                sonnet_lines = []
                for line in lines:
                    if line.strip():
                        sonnet_lines.append(line.strip())
                        if len(sonnet_lines) >= 14:  # Complete sonnet
                            sonnet = '\n'.join(sonnet_lines)
                            if not sonnet.endswith("---END-OF-SONNET---"):
                                sonnet = sonnet + "\n---END-OF-SONNET---"
                            actions.append(sonnet)
                            sonnet_lines = []

        return actions

class AgentSelfEvaluateSonnetWriting(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StateSonnetWriting,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
        cache: dict = None,
    ) -> float:
        """
        Returns the evaluation score for the SonnetWriting task using self-evaluation.
        """
        # Check if the state is already in the cache
        if cache is not None and state.current_state in cache:
            return cache[state.current_state]

        # Format the prompt based on whether we're evaluating a complete sonnet or a step
        if '---END-OF-SONNET---' in state.current_state:  # Complete sonnet
            # Remove separator for evaluation
            sonnet = state.current_state.replace('---END-OF-SONNET---', '').strip()
            prompt = prompts.self_evaluate_answer.format(
                input=state.puzzle,
                sonnet=sonnet
            )
        else:
            prompt = prompts.self_evaluate_step.format(
                input=state.puzzle,
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
        if '---END-OF-SONNET---' in state.current_state:  # Complete sonnet
            # For complete sonnets, scale to match the evaluation range (0 to 10)
            value = 10 * value
        else:
            # For intermediate steps, use a smaller range (0 to 1)
            value = value

        # Cache the value
        if cache is not None:
            cache[state.current_state] = value
        return value


