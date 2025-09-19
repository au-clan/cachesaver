import re
from typing import List, Tuple

from . import prompts as prompts
from .state import StateHLE
from ... import AgentFactory
from ...typedefs import Agent, Model, DecodingParameters

@AgentFactory.register
class AgentIoHLE(Agent):
    async def act(
        model: Model,
        state: StateHLE,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters
        ) -> List[str]:

        mcq = "Only return the letter or number of the choice you've made and not the answer itself." if state.answer_type == "multipleChoice" else ""

        responses = await model.request(
                prompt=[
                {"role": "system", "content": prompts.io + mcq},
                {"role": "user", "content": state.question},
                ],
                n=n,
                request_id=request_id,
                namespace=namespace,
                params=params,
            )
        proposals = [r.split("Final Answer:")[-1].strip() for r in responses]
        return responses
    
@AgentFactory.register
class AgentCotHLE(Agent):
    async def act(
        model: Model,
        state: StateHLE,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters
        ) -> List[str]:

        mcq = "Only return the letter or number of the choice you've made and not the answer itself." if state.answer_type == "multipleChoice" else ""

        responses = await model.request(
                prompt=[
                {"role": "system", "content": prompts.cot + mcq},
                {"role": "user", "content": state.question},
                ],
                n=n,
                request_id=request_id,
                namespace=namespace,
                params=params,
            )
        proposals = [r.split("Final Answer:")[-1].strip(" *") for r in responses]
        return proposals
    
@AgentFactory.register
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
        existing_steps = "\n".join(state.steps) if len(state.steps) > 0 else "None\n"

        # Early termination heuristic
        if (len(state.values) > 0 and state.values[max(state.values)] >= 0.9) or (
            len(state.steps) > 0 and "answer is" in state.steps[-1].lower()
        ):  # some hacky stuff from rest-mcts*
            prompt = prompts.summary.format(
                problem=state.question, existing_steps=existing_steps
            )
        else:

            prompt = prompts.act.format(
                problem=state.question, existing_steps=existing_steps
            )

        # Generate the responses
        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params
        )

        # Parse the responses
        proposals = [r.strip().split("\n")[:5] for r in responses]
        proposals = [parse_proposal(r, state.step_n, existing_steps) for r in proposals]
        return proposals

@AgentFactory.register
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
        existing_steps = "\n".join(state.steps) if len(state.steps) > 0 else "None\n"

        # Early termination heuristic
        if (len(state.values) > 0 and state.values[max(state.values)] >= 0.9) or (
            len(state.steps) > 0 and "answer is" in state.steps[-1].lower()
        ):  # some hacky stuff from rest-mcts*
            prompt = prompts.summary.format(
                problem=state.question, existing_steps=existing_steps
            )
        else:

            prompt = prompts.bfs.format(
                problem=state.question, existing_steps=existing_steps
            )
        
        responses = await model.request(
            prompt=prompt,
            n=1,
            request_id=request_id,
            namespace=namespace,
            params=params
        )

         # Generate the responses
        proposals = [
            "Next step: " + step.strip()
            for step in responses[0].split("Next step:")
            if step.strip()
        ]

        # Parse the responses
        proposals = [r.strip().split("\n")[:5] for r in proposals]
        proposals = [parse_proposal(r, state.step_n, existing_steps) for r in proposals]
        return proposals

@AgentFactory.register
class AgentAggregateHLE(Agent):

    @staticmethod
    async def act(
        model: Model,
        state: StateHLE,
        actions: List[str],
        k: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        """
        Returns the aggregated action for SciBench task.
        """
        if len(actions) == 0:
            return []
        
        if (len(state.values) > 0 and state.values[max(state.values)] >= 0.9) or (
            len(state.steps) > 0 and "answer is" in state.steps[-1].lower()
        ):  # some hacky stuff from rest-mcts*
            return actions

        # Format the prompt
        steps = "\n".join(actions)
        prompt = prompts.aggregate.format(problem=state.question, k=k, steps=steps)

        # Generate the response
        responses = await model.request(
            prompt=prompt,
            n=1,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        # Parse the response
        try:
            indexes = [int(i.strip()) - 1 for i in re.findall(r"\d+", responses[0])]
            out = [actions[i] for i in indexes if i < len(actions)]
        except Exception as e:
            out = []
        return out
    
@AgentFactory.register
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

@AgentFactory.register
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
        if cache is not None and state.current_state in cache:
            value = cache[state.current_state]

        else:
            # Format the promp
            prompt = prompts.evaluate.format(
                problem=state.question,
                existing_steps="\n".join(state.steps),
            )

            # Generate the response
            responses = await model.request(
                prompt=prompt,
                n=n,
                request_id=request_id,
                namespace=namespace,
                params=params,
            )

            # Parse the response
            values = [parse_value(r) for r in responses]
            value = sum(values) / len(values)

            # Cache the value
            if cache is not None:
                cache[state.current_state] = value
            state.values[state.step_n] = value
        return value
    
# ---Helper functions---#
def parse_proposal(response: List[str], step_n: int, existing_steps: str) -> str:
    p = ""
    for _ in response:
        p = p + _ + " "
    p = p.strip()

    if "Next step:" in p:
        stp = p.split("Next step:")[1].strip()
        if len(stp) < 2:
            # print('Output step too short!\n')
            return ""
        if stp in existing_steps:
            # print('Output step repeated!\n')
            return ""

        revised_ = "Step " + str(step_n) + ": " + stp

    elif "Step" in p and ":" in p:
        pre_len = len(p.split(":")[0])
        p_ = p[pre_len:]
        p_ = p_.split("Step")[0].strip()
        if len(p_) < 4:
            # print('Output step too short!\n')
            return ""
        p_ = p_[1:].strip()
        if p_ in existing_steps:
            # print('Output step repeated!\n')
            return ""

        revised_ = "Step " + str(step_n) + ": " + p_

    else:
        p_ = p.strip()
        if len(p_) < 3:
            # print('Output step too short!\n')
            return ""
        if p_ in existing_steps:
            # print('Output step repeated!\n')
            return ""

        revised_ = "Step " + str(step_n) + ": " + p_
    revised = revised_ + "\n"
    return revised

def parse_value(response: str, low=0.0, high=1.0) -> float:
    out_value = low

    # score expected in output
    if "score" not in response.lower():
        return out_value

    response = response.lower().split("score")[-1].strip()
    try:
        match = re.findall(r"-?[0-9]+\.?[0-9]*", response)[-1]
        out_value = float(match)
        out_value = min(max(low, out_value), high)
    except Exception as e:
        out_value = low
    return out_value
