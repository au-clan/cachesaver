import re
from typing import List

from . import prompts as prompts
from .state import StateSciBench
from ...typedefs import Agent, Model, DecodingParameters

class AgentActSciBench(Agent):
    
    @staticmethod
    async def act(model: Model, state: StateSciBench, n: int, namespace:str, request_id: str, params: DecodingParameters) -> List[str]:
        """
        Returns the action for the SciBench task.
        """
        # Format the prompt
        existing_steps = "\n".join(state.steps) if len(state.steps) > 0 else 'None\n'
        if (len(state.values)>0 and state.values[max(state.values)] >=0.9) or (len(state.steps) > 0 and "answer is" in state.steps[-1].lower()): # some hacky stuff from rest-mcts*
            prompt = prompts.summary.format(problem=state.puzzle, existing_steps=existing_steps)
        else:
            
            prompt = prompts.act.format(problem=state.puzzle, existing_steps=existing_steps)

        # Generate the response
        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params
        )
        
        # Parse the response
        proposals = [r.strip().split('\n')[:5] for r in responses]
        proposals = [parse_proposal(r, state.step_n, existing_steps) for r in proposals]
        return proposals
    
class AgentReactSciBench(Agent):
    
    @staticmethod
    async def act(model: Model, state: StateSciBench, n: int, namespace:str, request_id: str, params: DecodingParameters) -> List[str]:
        """
        Returns the action for the SciBench task.
        """
        # Format the prompt
        existing_steps = "\n".join(state.steps) if len(state.steps) > 0 else 'None\n'
        if (len(state.values)>0 and state.values[max(state.values)] >=0.9) or (len(state.steps) > 0 and "answer is" in state.steps[-1].lower()): # some hacky stuff from rest-mcts*
            prompt = prompts.summary.format(problem=state.puzzle, existing_steps=existing_steps)
        else:
            
            prompt = prompts.react.format(problem=state.puzzle, existing_steps=existing_steps)

        # Generate the response
        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params
        )
        
        for r in responses:
            print(r)
            print('---')
        # Parse the response
        proposals = [r.strip().split('\n')[:5] for r in responses]
        proposals = [parse_proposal(r, state.step_n, existing_steps) for r in proposals]
        return proposals
    
class AgentBfsSciBench(Agent):
    
    @staticmethod
    async def act(model: Model, state: StateSciBench, namespace:str, request_id: str, params: DecodingParameters) -> List[str]:
        """
        Returns the action for the SciBench task.
        """
        # Format the prompt
        existing_steps = "\n".join(state.steps) if len(state.steps) > 0 else 'None\n'
        if (len(state.values)>0 and state.values[max(state.values)] >=0.9) or (len(state.steps) > 0 and "answer is" in state.steps[-1].lower()): # some hacky stuff from rest-mcts*
            prompt = prompts.summary.format(problem=state.puzzle, existing_steps=existing_steps)
        else:
            
            prompt = prompts.bfs.format(problem=state.puzzle, existing_steps=existing_steps)

        # Generate the response
        responses = await model.request(
            prompt=prompt,
            n=1,
            request_id=request_id,
            namespace=namespace,
            params=params
        )
        # Parse the response
        proposals = ["Next step: " + step.strip() for step in responses[0].split("Next step:") if step.strip()]
        proposals = [r.strip().split('\n')[:5] for r in proposals]
        proposals = [parse_proposal(r, state.step_n, existing_steps) for r in proposals]
        return proposals
    
class AgentAggregateSciBench(Agent):
    
    @staticmethod
    async def act(model: Model, state: StateSciBench, actions: List[str], k: int, namespace: str, request_id: str, params: DecodingParameters) -> List[str]:
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

        def is_safe_index(idx):
            max_idx = len(actions) -1
            if idx >=0 and idx < max_idx:
                return True
            else:
                return False

        ret = []
        for i in matchs:
            idx = int(i.strip())
            if is_safe_index(idx):
                ret.append(actions[idx])
        return ret

class AgentEvaluateSciBench(Agent):

    @staticmethod
    async def act(model: Model, state: StateSciBench, n: int, namespace: str, request_id: str, params: DecodingParameters, cache: dict=None) -> float:

        # Check if the state is already in the cache
        if cache is not None and state.current_state in cache:
            value = cache[state.current_state]
        
        else:
            # Format the promp  
            num_examples = 2
            examples = "Example:\n" + "\n\nExample:\n".join([example for example in prompts.examples_evaluate[:num_examples]])
            prompt = prompts.evaluate.format(examples=examples, problem=state.puzzle, existing_steps="\n".join(state.steps))

            # Generate the response
            responses = await model.request(
                prompt=prompt,
                n=n,
                request_id=request_id,
                namespace=namespace,
                params=params
            )

            # Parse the response
            values = [parse_value(r) for r in responses]
            value = sum(values) / len(values)

            # Cache the value
            if cache is not None:
                cache[state.current_state] = value
            state.values[state.step_n] = value
        return value

        
        

#---Helper functions---#
def parse_proposal(response: List[str], step_n: int, existing_steps: str) ->  str:
    p = ''
    for _ in response:
        p = p + _ + ' '
    p = p.strip()

    if "Next step:" in p:
        stp = p.split('Next step:')[1].strip()
        if len(stp) < 2:
            #print('Output step too short!\n')
            return ''
        if stp in existing_steps:
            #print('Output step repeated!\n')
            return ''

        revised_ = 'Step ' + str(step_n) + ': ' + stp

    elif "Step" in p and ":" in p:
        pre_len = len(p.split(':')[0])
        p_ = p[pre_len:]
        p_ = p_.split('Step')[0].strip()
        if len(p_) < 4:
            #print('Output step too short!\n')
            return ''
        p_ = p_[1:].strip()
        if p_ in existing_steps:
            #print('Output step repeated!\n')
            return ''

        revised_ = 'Step ' + str(step_n) + ': ' + p_

    else:
        p_ = p.strip()
        if len(p_) < 3:
            #print('Output step too short!\n')
            return ''
        if p_ in existing_steps:
            #print('Output step repeated!\n')
            return ''

        revised_ = 'Step ' + str(step_n) + ': ' + p_
    revised = revised_ + '\n'
    return revised

def parse_value(response: str, low=0.0, high=1.0) -> float:
    out_value=low
    
    # score expected in output
    if "score" not in response.lower():
        return out_value
    
    response = response.lower().split("score")[-1].strip()
    try :
        match = re.findall(r'-?[0-9]+\.?[0-9]*', response)[-1]
        out_value = float(match)
        out_value = min(max(low, out_value), high)
    except Exception as e:
        out_value = low
    return out_value