import itertools
import re
from dataclasses import replace
from typing import List

import sympy
from sympy import simplify

from . import prompts_game24 as prompts
from .data_game24 import DataGame24
from .state_game24 import StateGame24
from ..basic import EnvironmentBasic
from ...typedefs import Verification, Inference


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

        ##Below is from RAFA:
        ##--------------------------------------------------------------------------------------------------------
        @staticmethod
        def get_current_numbers(y: str) -> str:
            last_line = y.strip().split('\n')[-1]
            return last_line.split('left: ')[-1].split(')')[0]

        @staticmethod
        def standard_prompt_wrap(x: str, y: str = '') -> str:
            return prompts.standard_prompt.format(input=x) + y

        @staticmethod
        def cot_prompt_wrap(x: str, y: str = '') -> str:
            # return cot_prompt.format(input=x) + y
            return prompts.cot.format(input=x) + y

        @staticmethod
        def propose_prompt_wrap(x: str, y: str = '') -> str:
            # current_numbers = get_current_numbers(y if y else x)
            current_numbers = EnvironmentGame24.Prompter.get_current_numbers(y if y else x)
            if current_numbers == '24':
                prompt = prompts.cot.format(input=x) + 'Steps:\n' + y
            else:
                prompt = prompts.propose_prompt.format(input=current_numbers)
            return prompt

        @staticmethod
        def validation_prompt_wrap(x: str, y: str) -> str or None:
            last_line = y.strip().split('\n')[-1]
            if 'left: ' not in last_line:  # last step
                return
            if len(y.strip().split('\n')) > 1:
                prev_line = EnvironmentGame24.Prompter.get_current_numbers(y.strip().split('\n')[-2])
            else:
                prev_line = x
            return prompts.validation_prompt.format(input=prev_line, formula=last_line)

        @staticmethod
        def value_prompt_wrap(x: str, y: str) -> str:
            last_line = y.strip().split('\n')[-1]
            if 'left: ' not in last_line:  # last step
                ans = last_line.lower().replace('answer: ', '')
                return prompts.evaluate_answer.format(input=x, answer=ans)
            current_numbers = EnvironmentGame24.Prompter.get_current_numbers(y)
            return prompts.value_prompt.format(input=current_numbers)

        @staticmethod
        def validation_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
            validations = [_.split('\n')[-1] for _ in value_outputs]
            if "invalid" in validations:
                return 0
            return 1

        @staticmethod
        def reflect_prompt_wrap(x: str, y: str, feedback: str) -> str:

            return prompts.reflect_prompt.format(input=x, answer=y,
                                                 feedback=feedback), prompts.value_reflect_prompt.format(input=x,
                                                                                                         answer=y,
                                                                                                         feedback=feedback)

        @staticmethod
        def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
            if len(y.strip().split('\n')) == 4 and 'answer' not in y.lower():
                return 0
            value_names = [_.split('\n')[-1].lower() for _ in value_outputs]
            value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}  # TODO: ad hoc
            value = sum(
                value * sum(name in value_name for value_name in value_names) for name, value in value_map.items())
            return value
        ##--------------------------------------------------------------------------------------------------------

    class Parser:
        def __init__(self):
            self.name = "Game24 Environment Parser"

        @staticmethod
        def cot(response: str) -> Inference:
            raise NotImplementedError

        @staticmethod
        def bfs(response: str) -> List[Inference]:
            if "left" in response:
                response = response.rpartition(")")[0] + ")"  # In case suggestion was cut in the middle
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

        ##RAFA code below-----------------------------------------------

        @staticmethod
        def check_answer(problem: str, answer: str):
            expression = answer.strip().split('\n')[-1].lower().replace('answer: ', '').split('=')[0]
            numbers = re.findall(r'\d+', expression)
            problem_numbers = re.findall(r'\d+', problem)
            if sorted(numbers) != sorted(problem_numbers):
                return False, "The numbers you use are not the original numbers from the problem."
            try:
                if sympy.simplify(expression) == 24:
                    return True, "The formula is correct."
                else:
                    return False, "The formula does not lead to 24."

            except Exception as e:
                return False, "The formula is invalid."

        @staticmethod
        def check_valid_move_rafa(state, cur_step):
            if state.index == 1:
                original_nums = [float(num) for num in state.history[-1].split(" ")]
            else:
                original_nums = [float(num) for num in state.history[-1].split('left:')[-1].strip("()").split(" ") if
                                 num != '']
            formula = [op for op in cur_step.split('left:')[0].strip("()").split(" ") if op != '']
            new_nums = [float(num) for num in cur_step.split('left:')[-1].strip("()").split(" ") if num != '']

            try:
                print(original_nums, new_nums, formula)
                original_nums.remove(float(eval(formula[0])))
                original_nums.remove(float(eval(formula[2])))
                for num in original_nums:
                    new_nums.remove(num)
                new_nums.remove(float(formula[4]))
                assert len(new_nums) == 0
            except ValueError:
                return False, "You use value that does not exists in last step or you use them repeatedly; or you drop numbers from the last step."
            except AssertionError:
                return False, "You have more numbers left than expected."

            return True, "The move the valid and correct."

        @staticmethod
        def check_equation(equation):
            try:
                left, right = equation.split("=")
                err = abs(eval(left) - float(right))
                if err < 1e-10:
                    return True, "The Equation is correct."
                else:
                    return False, f"The Equation is incorrect, the result should be {eval(left)}"
            except Exception as e:
                print(e)
                return False, "The Equation is not valid."

        @staticmethod
        def check_twentyfour(cur_step):
            cards = [float(num) for num in cur_step.split('left:')[-1].strip("()").split(" ") if num != '']

            try:
                for nums in itertools.permutations(cards):  # 四个数
                    for ops in itertools.product('+-*/', repeat=len(cards) - 1):  # 三个运算符（可重复！）
                        # 构造三种中缀表达式 (bsd)
                        if len(cards) == 4:
                            bds1 = '({0}{4}{1}){5}({2}{6}{3})'.format(*nums, *ops)  # (a+b)*(c-d)
                            bds2 = '(({0}{4}{1}){5}{2}){6}{3}'.format(*nums, *ops)  # (a+b)*c-d
                            bds3 = '{0}{4}({1}{5}({2}{6}{3}))'.format(*nums, *ops)  # a/(b-(c/d))
                            bdss = [bds1, bds2, bds3]
                        elif len(cards) == 3:
                            bds1 = '({0}{3}{1}){4}{2}'.format(*nums, *ops)  # (a+b)*c
                            bds2 = '{0}{3}({1}{4}{2})'.format(*nums, *ops)  # a+(b*c)
                            bdss = [bds1, bds2]
                        elif len(cards) == 2:
                            bds1 = '({0}{2}{1})'.format(*nums, *ops)  # a+b
                            bdss = [bds1]
                        else:
                            if len(nums) == 1 and abs(nums[0] - 24) < 1e-5:
                                return True, ""
                            return False, ""
                        for bds in bdss:  # 遍历
                            try:
                                if abs(eval(bds) - 24.0) < 1e-10:  # eval函数
                                    return True, ""
                            except ZeroDivisionError:  # 零除错误！
                                continue
            except Exception as e:
                print(e)
                return False, "It is not a valid formula."
            return False, ""

    def __init__(self, data_path: str):
        self.prompter = self.Prompter()
        self.parser = self.Parser()
        self.data = DataGame24(path=data_path)

    @classmethod
    def create(cls, data_path: str) -> "EnvironmentGame24":
        return cls(data_path)

    def reset(self, idx: int, randomness: int = 0) -> StateGame24:
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
            puzzle=state.puzzle,
            current_state=current_state,
            steps=state.steps + [action],
            randomness=randomness
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
    def verify(state: StateGame24) -> Verification:
        """
        Given a state returns its verification. 
        Verification pertains to whether the state is finished and if the answer is correct.
        """
        current_states = state.current_state.split(" ")
        if len(current_states) != 1 or len(state.steps) <= 3:
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
                v = Verification(finished=True, correct=False,
                                 message="Numbers used are not the same as the ones provided")
            try:
                if simplify(expression) == 24:
                    v = Verification(finished=True, correct=True, message="Correct")
                else:
                    # Operations performed do not result to 24
                    v = Verification(finished=True, correct=False, message="Operations performed do not result to 24")
            except Exception as e:
                v = Verification(finished=True, correct=False, message="Invalid expression")
        return v

    ##RAFA below:
    def reset_rafa(self, idx: int):
        puzzle = self.data.read(idx=idx)
        state = StateGame24(
            puzzle=puzzle,
            history=[],
            feedbacks=[],
            cur_step=0,
            current_state=puzzle,
            action=""
        )

        return state

    def check_step_rafa(self,environment, state: StateGame24, action):
        try:
            if "answer" in action.lower():
                correct, feedback = self.Parser.check_answer(state.puzzle, action)
                if not correct:
                    return f"Step {state.index} tries to give an answer but it is incorrect. {feedback}", 0
                return f"Step {state.index} is correct. {feedback}", 10
            else:
                # Check if the step is valid
                # correct, feedback = self.Parser.check_valid_move(idx, last_step, cur_step)
                correct, feedback = environment.Parser.check_valid_move_rafa(state, action)
                if not correct:
                    return f"Step {state.index} is illegal. {feedback}", 0

                formula = action.split('left:')[0].strip("()")
                correct, feedback = environment.Parser.check_equation(formula)
                if not correct:
                    return f"Step {state.index} is not correctly calculated. {feedback}", 0

                correct, feedback = environment.Parser.check_twentyfour(action)
                if not correct:
                    return f"Step {state.index} is impossible to lead to 24. {feedback}", 0

                return f"Step {state.index} is correct and can lead to 24.", 1

        except Exception as e:
            print(e)
            return f"Step {state.index} is invalid.", 0

    def generate_feedback_rafa(self, action, state: StateGame24, environment, config):
        feedbacks = ["Evaluation:"]  # feedbacks for each step
        rewards = 0
        if isinstance(action, list):
            action = action[0]
        actions = action.strip(" \n").split('\n')
        idx = len(state.history)

        #

        for action in actions:
            if idx == 0:
                last_step = state.puzzle
            else:
                last_step = state.history[-1]
            print(last_step)
            # print(action)
            if config.framework.feedback_print:
                idx += 1
            feedback, reward = self.check_step_rafa(state=state, action=action, environment=environment)
            if config.framework.feedback_print:
                state=replace(state, feedbacks=state.feedbacks.append(feedback))
                feedbacks.append(feedback)
            if reward > 0:
                if config.framework.feedback_print:
                    state = replace(state, history=state.history.append(action))
                    # state.history.append(action)
                rewards += reward
            else:
                break

        total_feedback = " ".join(feedbacks) if config.framework.feedback_print else None
        return state, total_feedback, rewards

    def step_rafa(self, config, action, state: StateGame24, environment):
        # state.cur_step += 1 #todo frozen dataclass
        state= replace(state, cur_step=state.cur_step+1)
        prev_len = len(state.history)
        generated_state, feedback, reward = self.generate_feedback_rafa(action, state, environment=environment, config=config)
        new_len = len(state.history)
        delta = new_len - prev_len + 1 if new_len < 4 else new_len - prev_len
        assert delta > 0
        done = (reward >= 10) or (generated_state.cur_step > config.framework.max_step)
        answer = [f"Step {i + 1}: {x}" for i, x in enumerate(action.split('\n')[:delta]) if x != ""]
        answer = "Attempt answer: " + "\n".join(answer)
        if generated_state.feedback:
            info = {'action': action, 'history': generated_state.history}
            obs = {'answer': answer, 'feedback': feedback}
        else:
            info = {'action': action, 'history': []}
            obs = {'answer': answer, 'feedback': []}
        return generated_state, obs, reward, done, info
