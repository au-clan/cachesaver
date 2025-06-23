import re
from typing import List
import random
from . import prompts
from .state import StateMathArena
from ...typedefs import Agent, Model, DecodingParameters
import logging
logger = logging.getLogger(__name__)
act_cache = {}

class AgentActMathArena(Agent):
    """Agent for direct mathematical problem-solving actions."""
    
    # async def act(model: Model, state: StateMathArena, n: int, namespace: str, request_id: str, params: DecodingParameters) -> List[str]:
    #     # Format the prompt

    #     prompt = prompts.bfs.format(
    #         problem=state.problem,
    #         existing_steps="\n".join(state.steps)
    #     )

    #     if prompt in act_cache:
    #         proposals = act_cache[prompt][:n]
    #         act_cache[prompt] = act_cache[prompt][n:]
    #     else:
    #         proposals = []
    #         act_cache[prompt] = []
        
    #     while len(proposals) < n:
    #         response = await model.request(
    #             prompt=prompt,
    #             n=1,
    #             request_id=request_id,
    #             namespace=namespace,
    #             params=params
    #         )
    #         logger.debug(f"[ACT] LLM raw response:\n{response[0]}")
            
    #         raw = response[0]
    #         extracted = AgentActMathArena._parse_multiple_steps(raw, step_n=state.step_n + 1, existing_steps="\n".join(state.steps))

    #         logger.debug(f"[ACT] Extracted steps:\n{extracted}")
    #         proposals.extend(extracted)


    #     # Shuffle & return
    #     random.seed(state.randomness)
    #     random.shuffle(proposals)
    #     act_cache[prompt].extend(proposals[n:])
    #     return proposals[:n]

    # @staticmethod
    # def _parse_multiple_steps(response: str, step_n: int, existing_steps: str) -> List[str]:
    #     """Parses multiple 'Next step X: $...$' lines into step strings."""
    #     lines = response.strip().split("\n")
    #     steps = []

    #     for line in lines:
    #         match = re.search(r"Next step\s*\d*\s*:\s*(.+)", line.strip())
    #         if not match and line.startswith("Next step"):
    #             parts = line.split(":")
    #             if len(parts) > 1:
    #                 step_text = parts[1].strip()
    #                 if step_text not in existing_steps and len(step_text) > 1:
    #                     formatted = f"Step {step_n}: {step_text}"
    #                     steps.append(formatted)
    #         else:
    #             step_text = match.group(1).strip()
    #             if step_text not in existing_steps and len(step_text) > 1:
    #                 formatted = f"Step {step_n}: {step_text}"
    #                 steps.append(formatted)

    #     return steps
    
    @staticmethod
    async def act(model: Model, state: StateMathArena, n: int, namespace: str, 
                  request_id: str, params: DecodingParameters) -> List[str]:
        
        existing_steps = "\n".join(state.steps) if state.steps else 'None\n'
        if (state.values and state.values[max(state.values)] >= 0.9) or \
           (state.steps and "answer is" in state.steps[-1].lower()):
            prompt = prompts.summary.format(
                problem=state.problem,
                existing_steps=existing_steps
            )
        else:
            prompt = prompts.bfs.format(
                problem=state.problem,
                existing_steps=existing_steps
            )

        # Generate multiple possible next steps
        responses = await model.request(
            prompt=prompt,
            n=n,  # BFS uses single response with multiple steps
            request_id=request_id,
            namespace=namespace,
            params=params
        )
        
        # Parse multiple steps from single response
        proposals = ["Next step: " + step.strip() 
                    for step in responses[0].split("Next step:") 
                    if step.strip()]
        proposals = [r.strip().split('\n')[:5] for r in proposals]
        proposals = [parse_proposal(r, state.step_n, existing_steps) 
                    for r in proposals]

        logger.debug(f"Prompt sent to model: {prompt}")
        logger.debug(f"Responses from model: {responses}")
        return proposals

class AgentBfsMathArena(Agent):
    """Agent for exploring multiple solution paths using BFS."""
    
    @staticmethod
    async def act(model: Model, state: StateMathArena, namespace: str,
                  request_id: str, params: DecodingParameters, n: int=1) -> List[str]:
        # Format BFS exploration prompt
        existing_steps = "\n".join(state.steps) if state.steps else 'None\n'
        if (state.values and state.values[max(state.values)] >= 0.9) or \
           (state.steps and "answer is" in state.steps[-1].lower()):
            prompt = prompts.summary.format(
                problem=state.problem,
                existing_steps=existing_steps
            )
        else:
            prompt = prompts.bfs.format(
                problem=state.problem,
                existing_steps=existing_steps
            )

        # Generate multiple possible next steps
        responses = await model.request(
            prompt=prompt,
            n=n,  # BFS uses single response with multiple steps
            request_id=request_id,
            namespace=namespace,
            params=params
        )
        
        # Parse multiple steps from single response
        proposals = ["Next step: " + step.strip() 
                    for step in responses[0].split("Next step:") 
                    if step.strip()]
        proposals = [r.strip().split('\n')[:5] for r in proposals]
        proposals = [parse_proposal(r, state.step_n, existing_steps) 
                    for r in proposals]

        logger.debug(f"Prompt sent to model: {prompt}")
        logger.debug(f"Responses from model: {responses}")
        return proposals
    
class AgentAggregateMathArena(Agent):
    
    @staticmethod
    async def act(model: Model, state: StateMathArena, actions: List[str], k: int, namespace: str, request_id: str, params: DecodingParameters) -> List[str]:
        """
        Returns the aggregated action for SciBench task.
        """
        # Format the prompt
        steps = '\n'.join(actions)
        prompt = prompts.aggregate.format(problem=state.puzzle, k=k, steps=steps)

        # Generate the response
        responses = await model.request(
            prompt=prompt,
            n=1,
            request_id=request_id,
            namespace=namespace,
            params=params
        )

        # Parse the response
        pattern = r'\d+'
        matchs = re.findall(pattern, responses[0])
        return [actions[int(i.strip()) - 1] for i in matchs]
    
class AgentEvaluateMathArena(Agent):
    """Agent for evaluating mathematical solutions."""
    
    @staticmethod
    async def act(model: Model, state: StateMathArena, n: int, namespace: str,
                  request_id: str, params: DecodingParameters, cache: dict = None) -> float:
        # Check if the state is already in the cache
        logger.debug(f"Agent evaluate called with state: {state}")

        if cache is not None and state.current_state in cache:
            value = cache[state.current_state]
        
        else:
            # Format the promp  
            num_examples = 2
            examples = "Example:\n" + "\n\nExample:\n".join([example for example in prompts.examples_evaluate[:num_examples]])
            prompt = prompts.evaluate.format(examples=examples, problem=state.problem, existing_steps="\n".join(state.steps))

            # Generate the response
            responses = await model.request(
                prompt=prompt,
                n=n,
                request_id=request_id,
                namespace=namespace,
                params=params
            )
            logger.debug(f"Model responses: {responses}")

            # Parse the response
            values = [parse_value(r) for r in responses]
            value = sum(values) / len(values)

            # Cache the value
            if cache is not None:
                cache[state.current_state] = value
            state.values[state.step_n] = value
        return value
    
class AgentAggregateMathArena(Agent):
    
    @staticmethod
    async def act(model: Model, state: StateMathArena, actions: List[str], k: int, namespace: str, request_id: str, params: DecodingParameters) -> List[str]:
        """
        Returns the aggregated action for Matharena task.
        """
        # Format the prompt
        steps = '\n'.join(actions)
        prompt = prompts.aggregate.format(problem=state.puzzle, k=k, steps=steps)

        # Generate the response
        responses = await model.request(
            prompt=prompt,
            n=1,
            request_id=request_id,
            namespace=namespace,
            params=params
        )

        # Parse the response
        pattern = r'\d+'
        matchs = re.findall(pattern, responses[0])
        return [actions[int(i.strip()) - 1] for i in matchs]

#---Helper functions---#
def parse_proposal(response: List[str], step_n: int, existing_steps: str) -> str:
    """Parse and format a proposal into a valid step."""
    p = ' '.join(response).strip()

    if "Next step:" in p:
        stp = p.split('Next step:')[1].strip()
        if len(stp) < 2 or stp in existing_steps:
            return ''
        revised_ = f'Step {step_n}: {stp}'

    elif "Step" in p and ":" in p:
        pre_len = len(p.split(':')[0])
        p_ = p[pre_len:].split('Step')[0].strip()
        if len(p_) < 4 or p_ in existing_steps:
            return ''
        p_ = p_[1:].strip()
        revised_ = f'Step {step_n}: {p_}'

    else:
        p_ = p.strip()
        if len(p_) < 3 or p_ in existing_steps:
            return ''
        revised_ = f'Step {step_n}: {p_}'

    return revised_ + '\n'

def parse_value(response: str, low: float = 0.0, high: float = 1.0) -> float:
    """Extract numerical score from evaluation response."""
    if "score" not in response.lower():
        return low
    
    try:
        response = response.lower().split("score")[-1].strip()
        match = re.findall(r'-?[0-9]+\.?[0-9]*', response)[-1]
        value = float(match)
        return min(max(low, value), high)
    except Exception:
        return low