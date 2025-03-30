import re
from typing import List
from typing import Tuple

from langchain import Wikipedia
from langchain.agents.react.base import DocstoreExplorer

import warnings
from langchain._api import LangChainDeprecationWarning
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)


from ...typedefs import Verification, Inference
from ..basic import EnvironmentBasic
from .state_hotpotqa import StateHotpotQA
from .data_hotpotqa import DataHotpotQA
from . import prompts_hotpotqa as prompts

OBS_CORRECT = "Answer is CORRECT."
OBS_INCORRECT = "Answer is INCORRECT."

class EnvironmentHotpotQA(EnvironmentBasic):
    class Prompter:
        def __init__(self):
            self.name = "HotpotQA Environment Prompter"
        
        @staticmethod
        def cot(state: StateHotpotQA) -> str:
            pass

        @staticmethod
        def bfs(state: StateHotpotQA) -> str:
            num_examples = 3
            examples = "(Example)\n" + "\n\n(Example)\n".join([example for example in prompts.examples_bfs[:num_examples]])
            prompt = prompts.bfs.format(examples=examples, question=state.puzzle, current_state=state.current_state)
            return prompt
        
        @staticmethod
        def react(state: StateHotpotQA) -> str:
            num_examples = 2
            examples = "(Example)\n" + "\n\n(Example)\n".join([example for example in prompts.examples_react[:num_examples]])
            prompt = prompts.react.format(examples=examples, question=state.puzzle, current_state=state.current_state)
            return prompt
        
        @staticmethod
        def act(state: StateHotpotQA) -> str:
            num_examples = 2
            examples = "(Example)\n" + "\n\n(Example)\n".join([example for example in prompts.examples_act[:num_examples]])
            prompt = prompts.act.format(examples=examples, question=state.puzzle, current_state=state.current_state)
            return prompt
        
        @staticmethod
        def evaluate(state: StateHotpotQA) -> str:
            num_examples = 2
            examples = "(Example)\n" + "\n\n(Example)\n".join([example for example in prompts.examples_evaluate[:num_examples]])
            prompt = prompts.evaluate.format(examples=examples, question=state.puzzle, current_state=state.current_state)
            return prompt

    class Parser:
        def __init__(self):
            self.name = "HotpotQA Environment Parser"
        
        @staticmethod
        def cot(response: str) -> Inference:
            pass

        @staticmethod
        def bfs(response: str) -> List[Inference]:
            suggestions = response.split("\n")
            inferences = [Inference(action=suggestion) for suggestion in suggestions]
            return inferences
        @staticmethod
        def evaluate(response: str) -> Inference:
            pass

        @staticmethod
        def react(response: str) -> Inference:
            try:
                thought, action = response.split("\n")
            except Exception as e:
                print(f"Unable to parse response : {response}")
                raise e
            inference = Inference(thought=thought.strip(), action=action.strip())
            return inference
        
        @staticmethod
        def act(response: str) -> Inference:
            """
            Not really needed but keeping here in case the prompt changes
            """
            inference = Inference(action=response.strip())
            return inference
        
        @staticmethod
        def evaluate(responses: str) -> Inference:
            values = []
            for response in responses:
                try:
                    value = int(re.search(r"correctness score is (\d+)", response).group(1))
                except AttributeError:
                    print(f"Unable to parse value from response : {response}")
                    value = 1
                values.append(value)
            inference = Inference(value=sum(values))
            return inference

    
    def __init__(self, data_path: str):
            self.prompter = self.Prompter()
            self.parser = self.Parser()
            self.data = DataHotpotQA(path=data_path)
    
    @classmethod
    def create(cls, data_path: str) -> "EnvironmentHotpotQA":
        return cls(data_path)
    

    def reset(self, idx: int, randomness: int=0)->StateHotpotQA:
        question, answer = self.data.read(idx=idx)
        state = StateHotpotQA(
            puzzle=question,
            current_state="",
            thoughts=[],
            actions=[],
            observations=[],
            steps=[],
            answer=answer,
            docstore=DocstoreExplorer(Wikipedia()),
            randomness=randomness,
        )
        return state
    
    @staticmethod
    def get_next_state(inference: Inference, state: StateHotpotQA, randomness: int) -> StateHotpotQA:
        action = inference.action.split(": ")[-1] if inference.action else inference.action
        thought = inference.thought.split(": ")[-1] if inference.thought else inference.thought
        
        assert action is not None, f"Action is None. Inference: {inference}"
        assert len(state.actions) == len(state.observations), f"Number of actions: {len(state.actions)} not equal to number observations: {len(state.observations)}"
        idx = len(state.actions) + 1
        
        action_type, argument = parse_action(action)
        obs = perform_action(state.docstore, action_type, argument, state.answer)
        
        if thought is None:
            step = f"""\nAction {idx}: {action}\nObservation {idx}: {obs}"""
        else:
            step = f"""\nThought {idx}: {thought}\nAction {idx}: {action}\nObservation {idx}: {obs}"""

        next_state = StateHotpotQA(
            puzzle=state.puzzle,
            current_state=state.current_state + step,
            actions=state.actions + [action],
            thoughts=state.thoughts + [thought] if thought else state.thoughts,
            observations = state.observations + [obs],
            steps=state.steps + [thought, action, obs] if thought else state.steps + [action, obs
            ],
            answer=state.answer,
            docstore=state.docstore,
            randomness=randomness
        )
        return next_state
    
    @staticmethod
    def get_value(state: StateHotpotQA) -> Inference:
        """
        No deterministc evaluation methods
        """
        inference = Inference(value=None)
        return inference

    @staticmethod
    def verify(state: StateHotpotQA) -> Verification:
        # Latest action
        action = state.actions[-1]
        action_type, _ = parse_action(action)
        if action_type == "Finish":
            # Latest observation
            obs = state.observations[-1]
            if obs == OBS_CORRECT:
                v = Verification(finished=True, correct=True, message="Correct")
            elif obs == OBS_INCORRECT:
                v = Verification(finished=True, correct=False, message="Incorrect")
            else:
                print(f"Error with state: {state}")
                raise ValueError("Last action is 'Finish' but the last observation is not within the expected observations : [OBS_CORRECT, OBS_INCORRECT]. Check environmnet.hotpotqa.py for the actual expected observations are (not the variables previously mentioned).")

        else:
            v = Verification(finished=False, correct=False, message="Not finished")
        return v

def parse_action(string):
    pattern = r'^(\w+)\[(.+)\]$'
    match = re.match(pattern, string)
    
    if match:
        action_type = match.group(1)
        argument = match.group(2)
        return action_type.lower().capitalize(), argument.strip()
    
    else:
        return None, None
    
def perform_action(docstore: DocstoreExplorer, action_type: str, argument: str, answer: str) -> str:
    if action_type == "Search":
        try:
            # Added '' around the argument. Not in reflexion. After some (small) testing, it seems to be equal or better.
            obs = docstore.search(f"\'{argument}\'").strip("\n").strip()
        except Exception as e:
            print(f"Error searching for '{argument}'")
            obs = 'Page does not exist, try something else.'
    elif action_type == "Lookup":
        try:
            obs = docstore.lookup(argument).strip('\n').strip()
        except Exception as e:
            print(f"Error looking up '{argument}'")
            obs = 'The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given in the previous observation.'
    
    elif action_type == "Finish":
        if argument.lower() == answer.lower():
            obs = OBS_CORRECT
        else:
            obs = OBS_INCORRECT

    else:
        obs = 'Invalid Action. Valid Actions are Lookup[<topic>], Search[<topic>] and Finish[<answer>].'
    return obs
