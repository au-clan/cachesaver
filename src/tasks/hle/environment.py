import re
import random
from typing import Tuple, Literal
from pydantic import BaseModel


from .state import StateHLE
from .prompts import JUDGE_PROMPT
from ... import EnvironmentFactory
from ...typedefs import Environment, MAX_SEED

from openai import OpenAI

OBS_CORRECT = "Answer is CORRECT."
OBS_INCORRECT = "Answer is INCORRECT."

client = OpenAI(timeout=300, max_retries=1)
JUDGE_MODEL = "gpt-4.1-nano"


@EnvironmentFactory.register
class EnvironmentHLE(Environment):
    """
    Environment for the HLE (Human-Labeled Explanations) task.
    """

    @staticmethod
    def step(state: StateHLE, action: str) -> StateHLE:
       
        # Randomness handling
        random.seed(state.randomness if hasattr(state, 'randomness') else 0)
        randomness = random.randint(0, MAX_SEED)

        # Create new state with updated information
        state = StateHLE(
            id=state.id,
            question=state.question,
            image=state.image,
            image_preview=state.image_preview,
            answer=state.answer,
            answer_type=state.answer_type,
            author_name=state.author_name,
            rationale=state.rationale,
            rationale_image=state.rationale_image,
            raw_subject=state.raw_subject,
            category=state.category,
            canary=state.canary,
            steps=state.steps + [action],
            current_state= state.current_state + action,
            randomness=randomness
        )
        return state

    @staticmethod
    def is_valid(state: StateHLE, action: str) -> bool:
        """
        Checks if the action taken is valid.
        """
        action_type, _ = parse_action(action.split("\n")[-1].split(": ")[-1])
        return action_type in ["Analyze", "Explain", "Finish"]

    @staticmethod
    def is_final(state: StateHLE) -> bool:
        """
        Checks if the current state is a final state.
        """
        if not state.steps:
            return False
        expression = state.steps[-1].split("\n")[-2]  # Get the last action
        action_type, _ = parse_action(expression.split(": ")[-1])
        return action_type == "Finish"

    # @staticmethod
    # def evaluate(state: StateHLE) -> Tuple[bool, float]:
    #     """
    #     Evaluates the current state.
    #     """
    #     is_final = EnvironmentHLE.is_final(state)
    #     if is_final:
    #         last_obs = state.steps[-1].split("\n")[-1]
    #         if last_obs == OBS_CORRECT:
    #             return True, 1.0
    #         return True, 0.0
    #     return False, 0.0
    @staticmethod
    def evaluate(state: StateHLE) -> Tuple[bool, float]:
        """Evaluates the current state"""
        #is_final = EnvironmentHLE.is_final(state)
        answer = extract_answer(state.question, state.answer, state.steps[-1])
        if answer["correct"] == "yes":
            return True, 1.0
        else:
            return True, 0.0 #TODO: Change True to is_final

#---Helper functions---#
class ExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int
    strict: Literal[True] # 100% reliability

def extract_answer(question, correct_answer, response):
    prompt = JUDGE_PROMPT.format(question=question, correct_answer=correct_answer, response=response)
    try:
        response = client.beta.chat.completions.parse(
                model=JUDGE_MODEL,
                max_completion_tokens=4096, # overkill for judge
                messages=[
                    {"role": "user", "content": prompt}
                ],
                response_format=ExtractedAnswer, 
            ) 
        content = response.choices[0].message.parsed
        return { 
            "correct_answer": correct_answer,
            "model_answer": content.extracted_final_answer,
            "reasoning": content.reasoning,
            "correct": content.correct,
            "confidence": content.confidence
        }
    except Exception as e: # very, very rare
        print("Error:", e)
        return None
        

def parse_action(string: str) -> Tuple[str, str]:
    """
    Parses an action string into its type and argument.
    Format: ActionType[argument]
    """
    pattern = r'^(\w+)\[(.+)\]$'
    match = re.match(pattern, string)
    
    if match:
        action_type = match.group(1)
        argument = match.group(2)
        return action_type.lower().capitalize(), argument.strip()
    return None, None

# def perform_action(action_type: str, argument: str, state: StateHLE) -> str:
#     """
#     Performs the specified action and returns an observation.
#     """
#     if action_type == "Analyze":
#         # Analyze the image or question content
#         return f"Analysis of '{argument}': Considering {state.category} category..."
    
#     elif action_type == "Explain":
#         # Provide explanation based on rationale
#         return f"Explanation: {state.rationale if hasattr(state, 'rationale') else 'No rationale available'}"
    
#     elif action_type == "Finish":
#         # Check if the answer matches
#         if argument.lower() == state.answer.lower():
#             return OBS_CORRECT
#         return OBS_INCORRECT
    
#     else:
#         return 'Invalid Action. Valid Actions are Analyze[<topic>], Explain[<aspect>], and Finish[<answer>].'