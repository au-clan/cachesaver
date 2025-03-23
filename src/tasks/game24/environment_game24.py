import re
from typing import List
from sympy import simplify

from ...typedefs import Verification, Inference
from ..basic import EnvironmentBasic
from .state_game24 import StateGame24
from .data_game24 import DataGame24
from . import prompts_game24 as prompts


class EnvironmentGame24(EnvironmentBasic):
    class Prompter:
        def __init__(self):
            self.name = "Game24 Environment Prompter"
        
        @staticmethod
        def cot(state: StateGame24) -> str:
            raise NotImplementedError
        
        @staticmethod
        def react(state: StateGame24) -> str:
            raise NotImplementedError
        
        @staticmethod
        def bfs(state: StateGame24) -> str:
            if state.current_state == "24":
                prompt = prompts.cot.format(input=state.puzzle) + "\nSteps:\n" + '\n'.join(state.steps) + "\nAnswer: "
            else:
                prompt = prompts.bfs.format(input=state.current_state)
            return prompt
        
        @staticmethod
        def evaluate(state: StateGame24) -> str:
            
            if "left" not in state.steps[-1]:
                # Answer has been found
                answer = state.steps[-1].lower().replace("answer: ", "")
                prompt = prompts.evaluate_answer.format(input=state.puzzle, answer=answer)
            else:
                # Answer has not been found
                prompt = prompts.evaluate.format(input=state.current_state)
            return prompt  
    
    class Parser:
        def __init__(self):
            self.name = "Game24 Environment Parser"

        @staticmethod
        def cot(response: str) -> Inference:
            raise NotImplementedError
        
        @staticmethod
        def bfs(response: str) -> List[Inference]:
            if "left" in response:
                response = response.rpartition(")")[0] + ")" # In case suggestion was cut in the middle
            suggestions = response.split("\n")
            inferences = [Inference(action=suggestion) for suggestion in suggestions]
            return inferences
        
        @staticmethod
        def evaluate(responses: str) -> Inference:
            code_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}
            codes = [response.split('\n')[-1].lower() for response in responses]
            value = sum(value * codes.count(name) for name, value in code_map.items())
            inference = Inference(value=value)
            return inference
        
        @staticmethod
        def react(response: str) -> Inference:
            raise NotImplementedError
        
        @staticmethod
        def act(response: str) -> Inference:
            raise NotImplementedError
    
    def __init__(self, data_path: str):
        self.prompter = self.Prompter()
        self.parser = self.Parser()
        self.data = DataGame24(path=data_path)

    @classmethod
    def create(cls, data_path: str) -> "EnvironmentGame24":
        return cls(data_path)
    
    def reset(self, idx: int, randomness: int=0)->StateGame24:
        puzzle = self.data.read(idx=idx)
        state = StateGame24(
            puzzle=puzzle,
            current_state=puzzle,
            steps=[],
            randomness=randomness
        )
        return state
    
    @staticmethod
    def get_next_state(inference: Inference, state: StateGame24, randomness: int) -> StateGame24:
        """
        Based on an observation and the current state, returns the next state.
        """
        action = inference.action
        thought = inference.thought
        assert action is not None, f"Action is none, Inference: {inference}"

        if "left" in action:
            current_state = action.split('left: ')[-1].split(')')[0]
        else:
            current_state = action.split(' = ')[-1]
        
        new_state = StateGame24(
            puzzle = state.puzzle,
            current_state = current_state,
            steps = state.steps + [action],
            randomness = randomness
        )
        return new_state
    
    @staticmethod
    def get_value(state: StateGame24) -> Inference:
        """
        Deterministc evaluation methods
        """
        if len(state.steps) == 4 and "answer" not in "\n".join(state.steps).lower():
            value = 0
        else:
            value = None
        
        inference = Inference(value=value)
        return inference
    
    @staticmethod
    def verify(state: StateGame24)-> dict:
        """
        Given a state returns its verification. 
        Verification pertains to whether the state is finished and if the answer is correct.
        """
        current_states = state.current_state.split(" ")
        if len(current_states) !=1 or len(state.steps)<=3:
            v = Verification(finished=False, correct=False, message="Not finished")
        elif current_states[0] != "24":
            v = Verification(finished=True, correct=False, message="Final number != 24")
        else:
            # One number left and it is 24
            expression = state.steps[-1].lower().replace('answer: ', '').split('=')[0]
            numbers = re.findall(r'\d+', expression)
            problem_numbers = re.findall(r'\d+', state.puzzle)
            if sorted(numbers) != sorted(problem_numbers):
                # Numbers used are not the same as the ones provided
                v = Verification(finished=True, correct=False, message="Numbers used are not the same as the ones provided")
            try:
                if simplify(expression) == 24:
                    v = Verification(finished=True, correct=True, message="Correct")
                else:
                    # Operations performed do not result to 24
                    v = Verification(finished=True, correct=False, message="Operations performed do not result to 24")
            except Exception as e:
                v = Verification(finished=True, correct=False, message="Invalid expression")
        return v