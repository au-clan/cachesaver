import itertools
import random
import re
from typing import Any

import sympy

from . import prompts as prompts, StateGame24
from ...algorithm_options.rafa import RafaRequest
from ...typedefs import Agent, Model, ModelRequestOptions


class AgentRafaGame24_eval(Agent):

    @staticmethod
    def check_answer(problem: str, answer: str):
        expression = answer.strip().split('\n')[-1].lower().replace('answer: ', '').split('=')[0]
        numbers = re.findall(r'\d+', expression)
        problem_numbers = re.findall(r'\d+', problem)
        if sorted(numbers) != sorted(problem_numbers):
            return False, "The numbers you use are not the original numbers from the problem."
        try:
            # print(sympy.simplify(expression))
            if sympy.simplify(expression) == 24:
                return True, "The formula is correct."
            else:
                return False, "The formula does not lead to 24."

        except Exception as e:
            # print(e)
            return False, "The formula is invalid."

    @staticmethod
    def check_valid_move(idx, last_step, cur_step):
        if idx == 1:
            original_nums = [float(num) for num in last_step.split(" ")]
        else:
            original_nums = [float(num) for num in last_step.split('left:')[-1].strip("()").split(" ") if
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

    def check_step(idx, last_step, cur_step, puzzle):
        puzzle = puzzle

        try:
            if "answer" in cur_step.lower():
                correct, feedback = AgentRafaGame24_eval.check_answer(puzzle, cur_step)
                if not correct:
                    return f"Step {idx} tries to give an answer but it is incorrect. {feedback}", 0
                return f"Step {idx} is correct. {feedback}", 10
            else:
                # Check if the step is valid
                correct, feedback = AgentRafaGame24_eval.check_valid_move(idx, last_step, cur_step)
                if not correct:
                    return f"Step {idx} is illegal. {feedback}", 0

                formula = cur_step.split('left:')[0].strip("()")
                correct, feedback = AgentRafaGame24_eval.check_equation(formula)
                if not correct:
                    return f"Step {idx} is not correctly calculated. {feedback}", 0

                correct, feedback = AgentRafaGame24_eval.check_twentyfour(cur_step)
                if not correct:
                    return f"Step {idx} is impossible to lead to 24. {feedback}", 0

                return f"Step {idx} is correct and can lead to 24.", 1

        except Exception as e:
            print(e)
            return f"Step {idx} is invalid.", 0

    @staticmethod
    def generate_feedback(action, history, puzzle, feedbacks):
        self_feedbacks = feedbacks
        feedbacks = ["Evaluation:"]  # feedbacks for each step
        rewards = 0
        self_history = history
        puzzle = puzzle
        # feedbacks=feedbacks
        if isinstance(action, list):
            action = action[0]
        actions = action.strip(" \n").split('\n')
        idx = len(self_history)

        for action in actions:
            if idx == 0:
                last_step = puzzle
            else:
                last_step = self_history[-1]
            print(last_step)
            # print(action)
            if True:  # self.feedback:
                idx += 1
            feedback, reward = AgentRafaGame24_eval.check_step(idx, last_step, action, puzzle=puzzle)
            if True:  # self.feedback:
                self_feedbacks.append(feedback)
                feedbacks.append(feedback)
            if reward > 0:
                if True:  # self.feedback:
                    self_history.append(action)
                rewards += reward
            else:
                break

        total_feedback = " ".join(feedbacks) if True else None  # self.feedback else None
        return total_feedback, rewards, self_history, self_feedbacks

    @staticmethod
    def validation_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        validations = [_.split('\n')[-1] for _ in value_outputs]
        if "invalid" in validations:
            return 0
        return 1

    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        if len(y.strip().split('\n')) == 4 and 'answer' not in y.lower():
            return 0
        value_names = [_.split('\n')[-1].lower() for _ in value_outputs]
        value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}  # TODO: ad hoc
        value = sum(value * sum(name in value_name for value_name in value_names) for name, value in value_map.items())
        return value

    @staticmethod
    def act(model: Model, state: StateGame24, **kwargs) -> Any:
        if "action" not in kwargs:
            raise ValueError("Missing required parameter: 'action'")
        if "max_steps" not in kwargs:
            raise ValueError("Missing required parameter: 'max_steps'")
        if "cur_step" not in kwargs:
            raise ValueError("Missing required parameter: 'cur_step'")
        if "history" not in kwargs:
            raise ValueError("Missing required parameter: 'history'")
        if "puzzle" not in kwargs:
            raise ValueError("Missing required parameter: 'puzzle'")
        if "feedbacks" not in kwargs:
            raise ValueError("Missing required parameter: 'feedbacks'")

        action = kwargs["action"]
        puzzle = kwargs["puzzle"]
        max_steps = kwargs["max_steps"]
        cur_step = kwargs["cur_step"]
        history = kwargs["history"]
        self_feedbacks = kwargs["feedbacks"]
        cur_step += 1
        prev_len = len(history)

        feedback, reward, self_history, self_feedbacks = AgentRafaGame24_eval.generate_feedback(action=action,
                                                                                                history=history,
                                                                                                puzzle=puzzle,
                                                                                                feedbacks=self_feedbacks)
        # update the history here:
        # todo oskar
        ##
        new_len = len(history)
        delta = new_len - prev_len + 1 if new_len < 4 else new_len - prev_len
        assert delta > 0
        done = (reward >= 10) or (cur_step > max_steps)
        answer = [f"Step {i + 1}: {x}" for i, x in enumerate(action.split('\n')[:delta]) if x != ""]
        answer = "Attempt answer: " + "\n".join(answer)

        info = {'action': action, 'history': history}
        obs = {'answer': answer, 'feedback': feedback}

        return obs, reward, done, info, self_history, self_feedbacks, cur_step


class AgentRAFA_reflect(Agent):
    @staticmethod
    async def act(model: Model, state: StateGame24, **kwargs) -> Any:
        if "request_options" not in kwargs:
            raise ValueError("Missing required parameter: 'request_options'")

        if "n_propose_sample" not in kwargs:
            raise ValueError("Missing required parameter: 'n_propose_sample'")

        if "observations_answer" not in kwargs:
            raise ValueError("Missing required parameter: 'observations_answer'")

        if "observations_feedback" not in kwargs:
            raise ValueError("Missing required parameter: 'observations_feedback'")

        request_options = kwargs["request_options"]

        n_propose_sample = kwargs["n_propose_sample"]
        observations_answer = kwargs["observations_answer"]
        observations_feedback = kwargs["observations_feedback"]

        reflect_prompt = prompts.reflect_prompt.format(input=state.puzzle,
                                                       answer=observations_answer,
                                                       feedback=observations_feedback
                                                       )

        reflect_messages = RafaRequest.from_request_options(request_options=request_options,
                                                            n=n_propose_sample)

        reflect_messages.add_user_message(reflect_prompt)

        reflects = await model.request(
            prompt=reflect_messages.messages,
            n=reflect_messages.n,
            request_id=reflect_messages.request_id,
            namespace=reflect_messages.namespace,
            params=ModelRequestOptions(
                max_completion_tokens=reflect_messages.max_completion_tokens,
                temperature=reflect_messages.temperature,
                top_p=reflect_messages.top_p,
                stop=reflect_messages.stop_token,
                logprobs=reflect_messages.logprobs,
            )
        )

        return reflects


class AgentRAFA_reflect_value(Agent):
    @staticmethod  # todo this is 1:1 with reflect except the prompt is value here
    async def act(model: Model, state: StateGame24, **kwargs) -> Any:
        if "request_options" not in kwargs:
            raise ValueError("Missing required parameter: 'request_options'")

        if "n_propose_sample" not in kwargs:
            raise ValueError("Missing required parameter: 'n_propose_sample'")

        if "observations_answer" not in kwargs:
            raise ValueError("Missing required parameter: 'observations_answer'")

        if "observations_feedback" not in kwargs:
            raise ValueError("Missing required parameter: 'observations_feedback'")

        request_options = kwargs["request_options"]

        n_propose_sample = kwargs["n_propose_sample"]
        observations_answer = kwargs["observations_answer"]
        observations_feedback = kwargs["observations_feedback"]

        value_reflect_prompt = prompts.value_reflect_prompt.format(input=state.puzzle,
                                                                   answer=observations_answer,
                                                                   feedback=observations_feedback)
        value_reflects_messages = RafaRequest.from_request_options(request_options=request_options,
                                                                   n=n_propose_sample)
        value_reflects_messages.add_user_message(value_reflect_prompt)

        value_reflects = await model.request(prompt=value_reflects_messages.messages,
                                             n=value_reflects_messages.n,
                                             request_id=value_reflects_messages.request_id,
                                             namespace=value_reflects_messages.namespace,
                                             params=ModelRequestOptions(
                                                 max_completion_tokens=value_reflects_messages.max_completion_tokens,
                                                 temperature=value_reflects_messages.temperature,
                                                 top_p=value_reflects_messages.top_p,
                                                 stop=value_reflects_messages.stop_token,
                                                 logprobs=value_reflects_messages.logprobs
                                             )
                                             )

        return value_reflects


class AgentRAFA_plan(Agent):
    @staticmethod
    def get_current_numbers(y: str) -> str:
        last_line = y.strip().split('\n')[-1]
        return last_line.split('left: ')[-1].split(')')[0]

    @staticmethod
    async def act(model: Model, state: StateGame24, **kwargs) -> Any:
        if "request_options" not in kwargs:
            raise ValueError("Missing required parameter: 'request_options'")
        if "candidate" not in kwargs:
            raise ValueError("Missing required parameter: 'y'")
        if "n_propose_sample" not in kwargs:
            raise ValueError("Missing required parameter: 'n_propose_sample'")
        if "n_generate_sample" not in kwargs:
            raise ValueError("Missing required parameter: 'n_generate_sample'")
        if "reflects_list" not in kwargs:
            raise ValueError("Missing required parameter: 'reflects_list'")

        n_propose_sample = kwargs["n_propose_sample"]
        n_generate_sample = kwargs["n_generate_sample"]
        reflects_list = kwargs["reflects_list"]

        request_options = kwargs["request_options"]

        candidate = kwargs["candidate"]

        prompt = "Now we would like to play a game of 24. That is, given 4 numbers, try to use "
        "them with arithmetic operations (+ - * /) to get 24. "

        history = [{"feedback": prompt},
                   {"feedback": "What you have learned about the puzzle are summarized below.\n" + "\n".join(
                       reflects_list)}]

        # this is the old generate feedback

        current_numbers = AgentRAFA_plan.get_current_numbers(candidate if candidate else state.puzzle)
        if current_numbers == '24':
            prompt = prompts.cot.format(input=state.puzzle) + 'Steps:\n' + candidate
        else:
            prompt = prompts.propose_prompt.format(input=current_numbers)
        propose_prompt = prompt
        history_messages = RafaRequest.from_request_options(request_options=request_options,
                                                            n=n_generate_sample)

        for h in history:
            if 'answer' in h:
                history_messages.add_assistant_message(h["answer"])
            if 'feedback' in h:
                history_messages.add_user_message(h["feedback"])
        history_messages.add_user_message(propose_prompt)
        # history_messages.stop_token = ["\n\n"]  # todo i dont get how their method works with this, in groq it doesnt work

        result = await model.request(prompt=history_messages.messages,
                                     n=history_messages.n,
                                     request_id=f"{history_messages.request_id}-randomnes-{random.randint(1, 10000)}",
                                     namespace=history_messages.namespace,
                                     params=ModelRequestOptions(
                                         max_completion_tokens=history_messages.max_completion_tokens,
                                         temperature=history_messages.temperature,
                                         top_p=history_messages.top_p,
                                         stop=history_messages.stop_token,
                                         logprobs=history_messages.logprobs,
                                     )
                                     )
        # result = await model.request(history_messages)
        # todo this logic is flawed, in the event you get a response where the first line is a "here is suggestions:.." and then the next line is a new line with nothing on it and then the third is a suggestion. You will "learn" nothing as the sample could be two:
        pattern = r'\d+\.\s+([^\n]+?\(left: [^\n]+?\))'  # todo this is the numbered
        pattern = r'[-\d]+\.\s+([^\n]+?\(left: [^\n]+?\))|-\s+([^\n]+?\(left: [^\n]+?\))'  # todo this is with the "-"
        all_matches = []

        for text in result:
            matches = re.findall(pattern, text)
            cleaned = [m[0] if m[0] else m[1] for m in matches]
            all_matches.extend(cleaned)

        proposals = all_matches[:min(len(all_matches), n_propose_sample)]
        return [candidate + _ + '\n' for _ in proposals]


class AgentRAFA_plan_evaluate(Agent):
    @staticmethod
    def get_current_numbers(y: str) -> str:
        last_line = y.strip().split('\n')[-1]
        return last_line.split('left: ')[-1].split(')')[0]

    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        last_line = y.strip().split('\n')[-1]
        if 'left: ' not in last_line:  # last step
            ans = last_line.lower().replace('answer: ', '')
            return prompts.evaluate_answer.format(input=x, answer=ans)
        current_numbers = AgentRAFA_plan_evaluate.get_current_numbers(y)
        return prompts.value_prompt.format(input=current_numbers)

    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        if len(y.strip().split('\n')) == 4 and 'answer' not in y.lower():
            return 0
        value_names = [_.split('\n')[-1].lower() for _ in value_outputs]
        value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}  # TODO: ad hoc
        value = sum(
            value * sum(name in value_name for value_name in value_names) for name, value in value_map.items())
        return value

    @staticmethod
    async def act(model: Model, state: StateGame24, **kwargs) -> Any:
        if "request_options" not in kwargs:
            raise ValueError("Missing required parameter: 'request_options'")
        if "new_output_candidates" not in kwargs:
            raise ValueError("Missing required parameter: 'new_output_candidates'")
        if "value_reflects" not in kwargs:
            raise ValueError("Missing required parameter: 'value_reflects'")

        if "n_evaluate_sample" not in kwargs:
            raise ValueError("Missing required parameter: 'n_evaluate_sample'")

        n_evaluate_sample = kwargs["n_evaluate_sample"]
        request_options = kwargs["request_options"]
        value_reflects = kwargs["value_reflects"]

        new_output_candidates = kwargs["new_output_candidates"]  # to sample etc

        prompt = "Now we would like to play a game of 24. That is, given 4 numbers, try to use "
        "them with arithmetic operations (+ - * /) to get 24. "
        history = [prompt,
                   dict(feedback="What you have learned about the puzzle are summarized below.\n" + "\n".join(
                       value_reflects))]
        values = []

        for candidate in new_output_candidates:  # each partial output

            value_prompt = AgentRAFA_plan_evaluate.value_prompt_wrap(state.puzzle, candidate)

            history_messages = RafaRequest.from_request_options(request_options=request_options,
                                                                n=n_evaluate_sample)
            for h in history:
                if 'answer' in h:
                    history_messages.add_assistant_message(h["answer"])
                if 'feedback' in h:
                    history_messages.add_user_message(h["feedback"])
            history_messages.add_user_message(value_prompt)  # todo confirm order of messages
            history_messages.request_id = f"step-{str(state.puzzle)}-{1}-{candidate}-{hash(1)}"  # todo this shpould be done properly at some point

            value_outputs = await model.request(prompt=history_messages.messages,
                                                n=history_messages.n,
                                                request_id=history_messages.request_id,
                                                namespace=history_messages.namespace,
                                                params=ModelRequestOptions(
                                                    max_completion_tokens=history_messages.max_completion_tokens,
                                                    temperature=history_messages.temperature,
                                                    top_p=history_messages.top_p,
                                                    stop=history_messages.stop_token,
                                                    logprobs=history_messages.logprobs,
                                                )
                                                )

            value = AgentRAFA_plan_evaluate.value_outputs_unwrap(state.puzzle, candidate,
                                                                 value_outputs)  # todo fix types

            values.append((candidate, value))

        return values
