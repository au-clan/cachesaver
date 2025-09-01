import re
from typing import List, Tuple

from . import prompts as prompts
from .state import StateMathArena
from ... import AgentFactory
from ...typedefs import Agent, Model, DecodingParameters

@AgentFactory.register
class AgentActMathArena(Agent):
    """
    Agent performing the Act operation for the MathArena task.
    """

    @staticmethod
    async def act(model: Model, state: StateMathArena, n: int, namespace: str, request_id: str, params: DecodingParameters) -> List[str]:
        """
        Returns a list of n actions for the MathArena task.
        """
        # Format the prompt
        prompt = prompts.act.format(input=state.problem)
        
        # Generate the responses
        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params
        )

        # Parse the responses and ensure they follow the action format
        actions = []
        for response in responses:
            # Extract action format (Analyze[...], Explain[...], or Finish[...])
            if any(response.startswith(prefix) for prefix in ["Analyze[", "Explain[", "Finish["]):
                actions.append(response.strip())

        return actions

@AgentFactory.register
class AgentBfsMathArena(Agent):
    """
    Agent performing the BFS operation for the MathArena task.
    """

    @staticmethod
    async def act(model: Model, state: StateMathArena, namespace: str, request_id: str, params: DecodingParameters) -> List[str]:
        """
        Returns a list of possible solution approaches.
        """
        # Format the prompt
        prompt = prompts.bfs.format(input=state.problem)

        # Generate the response
        response = await model.request(
            prompt=prompt,
            n=1,
            request_id=request_id,
            namespace=namespace,
            params=params
        )

        # Parse the response into individual approaches
        approaches = []
        for line in response[0].split('\n'):
            if any(line.startswith(prefix) for prefix in ["Analyze[", "Explain[", "Finish["]):
                approaches.append(line.strip())

        return approaches

@AgentFactory.register
class AgentCotMathArena(Agent):
    """
    Agent performing the Chain-of-Thought operation for the MathArena task.
    """

    @staticmethod
    async def act(model: Model, state: StateMathArena, n: int, namespace: str, request_id: str, params: DecodingParameters) -> List[str]:
        """
        Returns a complete solution chain for the math problem.
        """
        # Format the prompt
        prompt = prompts.cot.format(input=state.problem)

        # Generate the responses
        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params
        )

        # Parse the responses to extract solution steps
        solution_chains = []
        for response in responses:
            steps = []
            for line in response.split('\n'):
                if any(line.startswith(prefix) for prefix in ["Analyze[", "Explain[", "Finish["]):
                    steps.append(line.strip())
            if steps:
                solution_chains.append('\n'.join(steps))

        return solution_chains

@AgentFactory.register
class AgentEvaluateMathArena(Agent):
    """
    Agent performing the Evaluate operation for the MathArena task.
    """

    @staticmethod
    async def act(model: Model, state: StateMathArena, n: int, namespace: str, request_id: str, params: DecodingParameters, cache: dict=None) -> float:
        """
        Returns an evaluation score for the current state.
        """
        # Check cache
        if cache is not None and str(state.steps) in cache:
            return cache[str(state.steps)]

        # Format the prompt
        prompt = prompts.evaluate.format(input=state.problem)

        # Generate the responses
        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params
        )

        # Parse responses to determine solvability score
        scores = []
        for response in responses:
            score = 0.0
            response = response.lower().strip()
            if "solvable" in response:
                score = 1.0
            elif "likely" in response:
                score = 0.5
            elif "needs more info" in response:
                score = 0.25
            scores.append(score)

        # Average the scores
        final_score = sum(scores) / len(scores)

        # Cache the result
        if cache is not None:
            cache[str(state.steps)] = final_score

        return final_score