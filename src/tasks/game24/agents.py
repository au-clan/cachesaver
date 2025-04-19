import asyncio
import itertools
import re
from dataclasses import replace
from typing import List, Any, cast

import sympy
from requests import Response

from . import prompts as prompts
from .state import StateGame24, GameState_rafa

from ...typedefs import Request, Agent, Model, DecodingParameters, State, ActKwargs_rafa, EvalKwargs_rafa


# Helper functions
def get_current_numbers(state: StateGame24) -> str:
    """
    Returns the current numbers in the state.
    """
    last_line = state.current_state.strip().split('\n')[-1]
    return last_line.split('left: ')[-1].split(')')[0]


def get_formula(state: StateGame24) -> str:
    formula = state.steps[-1].lower().replace("answer: ", "")
    return formula


class AgentActGame24(Agent):
    """
    """

    async def act(model: Model, state: StateGame24, n: int, namespace: str, request_id: str,
                  params: DecodingParameters) -> List[str]:
        # Format the prompt
        if state.current_state == "24":
            prompt = prompts.cot.format(input=state.puzzle) + "\nSteps:\n" + '\n'.join(state.steps) + "\nAnswer: "
        else:
            current_numbers = get_current_numbers(state)
            prompt = prompts.act.format(input=current_numbers)

        # Generate the response
        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params
        )

        # Parse the response
        proposals = [r.strip() for r in responses]
        return proposals


class AgentBfsGame24(Agent):

    @staticmethod
    async def act(model: Model, state: StateGame24, namespace: str, request_id: str, params: DecodingParameters) -> \
            List[str]:
        """
        Returns a list of actions for the Game of 24 task.
        """

        # Format the prompt
        if state.current_state == "24":
            prompt = prompts.cot.format(input=state.puzzle) + "\nSteps:\n" + '\n'.join(state.steps) + "\nAnswer: "
        else:
            current_numbers = get_current_numbers(state)
            prompt = prompts.bfs.format(input=current_numbers)

        # Generate the response
        response = await model.request(
            prompt=prompt,
            n=1,
            request_id=request_id,
            namespace=namespace,
            params=params
        )

        # Parse the response
        if state.current_state != "24":
            response = [response[0].rpartition(")")[0] + ")"]
        proposals = [r.strip() for r in response[0].split("\n")]
        return proposals


class AgentEvaluateGame24(Agent):

    @staticmethod
    async def act(model: Model, state: StateGame24, n: int, namespace: str, request_id: str, params: DecodingParameters,
                  cache: dict = None) -> float:
        """
        Returns a value for the given state
        """

        # Check if the state is already in the cache
        if cache is not None and state.current_state in cache:
            return cache[state.current_state]

        # Format the prompt
        if "left" not in state.steps[-1]:
            formula = get_formula(state)
            prompt = prompts.evaluate_answer.format(input=state.puzzle, answer=formula)
        else:
            prompt = prompts.evaluate.format(input=state.current_state)

        # Format the request
        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params
        )

        # Parse the response
        codes = [r.split('\n')[-1].lower() for r in responses]
        code_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}
        value = sum(value * codes.count(code) for code, value in code_map.items())

        # Cache the value
        if cache is not None:
            cache[state.current_state] = value
        return value


#####TODo
#####TODo
#####TODo
#####TODo


class AgentRafaGame24_act(Agent):
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
        current_numbers = AgentRafaGame24_act.get_current_numbers(y if y else x)
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
            prev_line = AgentRafaGame24_act.get_current_numbers(y.strip().split('\n')[-2])
        else:
            prev_line = x
        return prompts.validation_prompt.format(input=prev_line, formula=last_line)

    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        last_line = y.strip().split('\n')[-1]
        if 'left: ' not in last_line:  # last step
            ans = last_line.lower().replace('answer: ', '')
            return prompts.evaluate_answer.format(input=x, answer=ans)
        current_numbers = AgentRafaGame24_act.get_current_numbers(y)
        return prompts.value_prompt.format(input=current_numbers)

    @staticmethod
    def validation_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        validations = [_.split('\n')[-1] for _ in value_outputs]
        if "invalid" in validations:
            return 0
        return 1

    @staticmethod
    def reflect_prompt_wrap(x: str, y: str, feedback: str):  # todo removed type

        return (prompts.reflect_prompt.format(input=x,
                                              answer=y,
                                              feedback=feedback
                                              ),
                prompts.value_reflect_prompt.format(input=x,
                                                    answer=y,
                                                    feedback=feedback))

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
    def check_step_rafa(state: GameState_rafa, action):
        try:
            if "answer" in action.lower():
                correct, feedback = AgentRafaGame24_act.check_answer(state.puzzle, action)
                if not correct:
                    return f"Step {state.index} tries to give an answer but it is incorrect. {feedback}", 0
                return f"Step {state.index} is correct. {feedback}", 10
            else:
                # Check if the step is valid
                correct, feedback = AgentRafaGame24_act.check_valid_move_rafa(state, action)
                if not correct:
                    return f"Step {state.index} is illegal. {feedback}", 0

                formula = action.split('left:')[0].strip("()")
                correct, feedback = AgentRafaGame24_act.check_equation(formula)
                if not correct:
                    return f"Step {state.index} is not correctly calculated. {feedback}", 0

                correct, feedback = AgentRafaGame24_act.check_twentyfour(action)
                if not correct:
                    return f"Step {state.index} is impossible to lead to 24. {feedback}", 0

                return f"Step {state.index} is correct and can lead to 24.", 1

        except Exception as e:
            print(e)
            return f"Step {state.index} is invalid.", 0

    @staticmethod
    def generate_feedback_rafa(action, state: GameState_rafa, feedback_print: bool):
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
            if feedback_print:
                idx += 1
            feedback, reward = AgentRafaGame24_act.check_step_rafa(state=state, action=action)
            if feedback_print:
                state = replace(state, feedbacks=state.feedbacks.append(feedback))
                feedbacks.append(feedback)
            if reward > 0:
                if feedback_print:
                    state = replace(state, history=state.history.append(action))
                    # state.history.append(action)
                rewards += reward
            else:
                break

        total_feedback = " ".join(feedbacks) if feedback_print else None
        return state, total_feedback, rewards

    @staticmethod
    async def gpt_with_history(model: Model, request_params: Request, prompt, history, n, namespace,
                               request_id) -> Response:
        messages = []
        for h in history:
            if 'answer' in h:
                messages.extend([{"role": "assistant", "content": h["answer"]}])
            if 'feedback' in h:
                messages.extend([{"role": "user", "content": h["feedback"]}])
        messages.append({"role": "user", "content": prompt})

        response = await AgentRafaGame24_act.chatgpt(messages=messages, prompt=prompt, n=n, namespace=namespace,
                                                     request_id=request_id, request_params=request_params, model=model)
        return response

    @staticmethod
    async def gpt(prompt, n, namespace, request_id, model: Model, request_params: Request, ) -> Response:
        messages = [{"role": "user", "content": prompt}]
        return await AgentRafaGame24_act.chatgpt(messages=messages, prompt=prompt, n=n, namespace=namespace,
                                                 request_id=request_id, model=model, request_params=request_params)

    @staticmethod
    async def chatgpt(model: Model, request_params: Request, messages, prompt, n, namespace, request_id) -> Response:
        response = await model.request(Request(  # todo smth smart with the prompot/m,essage no neeed for double
            prompt=prompt,
            messages=messages,
            n=n,
            request_id=request_id,
            namespace=namespace,
            max_completion_tokens=request_params.max_completion_tokens,
            temperature=request_params.temperature,
            top_p=request_params.top_p,
            stop=request_params.stop,
            logprobs=request_params.logprobs
        )
        )
        return response

    @staticmethod
    async def get_value(model: Model, history, request_params: Request, state, y, namespace, request_id, cache_value,
                        value_cache,
                        n_evaluate_sample: int):
        value_prompt = AgentRafaGame24_act.value_prompt_wrap(state.puzzle, y)

        if cache_value and value_prompt in value_cache:
            return AgentRafaGame24_act.value_cache[value_prompt]  # todo cache values in future
        value_outputs = await AgentRafaGame24_act.gpt_with_history(prompt=value_prompt,
                                                                   history=history,
                                                                   model=model,
                                                                   request_params=request_params,
                                                                   n=n_evaluate_sample,
                                                                   namespace=namespace,
                                                                   request_id=request_id)  # todo could simply use the state.puzzle_index as the namespace as that is what it is...

        value = AgentRafaGame24_act.value_outputs_unwrap(state.puzzle, y, value_outputs)  # todo fix types
        if cache_value:
            # environment.Prompter.value_cache[value_prompt] = value
            # todo fix the caching
            print("cache not impl yet")
        return value

    @staticmethod
    async def get_values(state, ys, history, cache_value, request_id, namespace, model: Model,
                         request_params: Request, value_cache, n_evaluate_sample):
        values = []
        local_value_cache = {}
        for y in ys:  # each partial output
            if y in local_value_cache and cache_value:  # avoid duplicate candidates
                value = local_value_cache[y]
            else:
                # value = get_value(env, history, x, y, n_evaluate_sample, cache_value=cache_value)
                value = await AgentRafaGame24_act.get_value(state=state,
                                                            y=y,
                                                            history=history,
                                                            request_params=request_params,
                                                            model=model,
                                                            cache_value=cache_value,
                                                            value_cache=value_cache,
                                                            namespace=namespace,
                                                            request_id=request_id,
                                                            n_evaluate_sample=n_evaluate_sample)
                if cache_value:
                    local_value_cache[y] = value
            values.append(value)
        return values

    @staticmethod  # todo confirm this impl the gpt_with_history is both method and a global variable in their impl
    async def get_proposals(model: Model, puzzle, history, y, n_propose_sample, namespace, request_id,
                            request_params: Request, ):
        propose_prompt = AgentRafaGame24_act.propose_prompt_wrap(puzzle, y)
        result = await AgentRafaGame24_act.gpt_with_history(prompt=propose_prompt,
                                                            history=history,
                                                            n=1,
                                                            model=model,
                                                            namespace=namespace,
                                                            request_params=request_params,
                                                            request_id=request_id)  # todo stop token

        proposal_list = [x.split('\n') for x in result]  # todo the stop token
        proposals = []
        for p in proposal_list:
            proposals.extend(p)
        proposals = proposals[:min(len(proposals), n_propose_sample)]
        return [y + _ + '\n' for _ in proposals]

    @staticmethod
    async def plan_rafa(model: Model, state: GameState_rafa, n_propose_sample, n_select_sample, to_print, cache_value,
                        value_cache, n_evaluate_sample,
                        request_parameters):

        history = state.env_history
        ys = ["\n".join(history) + "\n"] if len(history) else [""]  # current output candidates
        infos = []
        prompt = "Now we would like to play a game of 24. That is, given 4 numbers, try to use "
        "them with arithmetic operations (+ - * /) to get 24. "
        obs = [{"feedback": prompt},
               {"feedback": "What you have learned about the puzzle are summarized below.\n" + "\n".join(
                   state.reflects)}]

        value_obs = [prompt,
                     dict(feedback="What you have learned about the puzzle are summarized below.\n" + "\n".join(
                         state.value_reflects))]
        for step in range(4 - len(history)):
            # generation
            coroutines = [
                AgentRafaGame24_act.get_proposals(puzzle=state.puzzle,
                                                  history=obs,
                                                  y=y,
                                                  n_propose_sample=n_propose_sample,
                                                  namespace=str(state.index),
                                                  request_id=f"step-{str(state.index)}-{step}-{y}-{hash(state)}",
                                                  model=model,
                                                  request_params=request_parameters)
                for y in ys]
            new_ys = await asyncio.gather(*coroutines)

            new_ys = list(itertools.chain(*new_ys))
            ids = list(range(len(new_ys)))
            # evaluation
            values = await AgentRafaGame24_act.get_values(state=state,
                                                          history=value_obs,
                                                          ys=new_ys,
                                                          namespace=str(state.index),
                                                          request_id=f"step-{str(state.index)}-{step}-{step}-{hash(state)}",
                                                          model=model,
                                                          request_params=request_parameters,
                                                          cache_value=cache_value,
                                                          value_cache=value_cache,
                                                          n_evaluate_sample=n_evaluate_sample
                                                          )  # todo step twice, not sure if safe

            # selection
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:n_select_sample]
            select_new_ys = [new_ys[select_id] for select_id in select_ids]

            # log
            if to_print:
                sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
                print(
                    f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')

            infos.append(
                {'step': step, 'x': state.puzzle, 'ys': ys, 'new_ys': new_ys, 'values': values,
                 'select_new_ys': select_new_ys})
            ys = select_new_ys

        if to_print:
            print(ys)

        ys_list = [y.split('\n')[len(history):] for y in ys]
        res_ys = ["\n".join(ys) for ys in ys_list][0]
        return state, res_ys, {'steps': infos}

    @staticmethod
    async def reflect_rafa(state: GameState_rafa, n_generate_sample, model: Model, request_params: Request, namespace,
                           request_id):
        y = state.obs_answer

        feedback = state.obs_feedback
        reflect_prompt, value_reflect_prompt = AgentRafaGame24_act.reflect_prompt_wrap(state.puzzle, y, feedback)
        reflects = await AgentRafaGame24_act.gpt(prompt=reflect_prompt,
                                                 n=n_generate_sample,
                                                 request_params=request_params,
                                                 model=model,
                                                 namespace=namespace,
                                                 request_id=request_id)

        value_reflects = await AgentRafaGame24_act.gpt(prompt=value_reflect_prompt,
                                                       n=n_generate_sample,
                                                       request_params=request_params,
                                                       model=model,
                                                       namespace=namespace,
                                                       request_id=request_id)
        # todo confirm the right types
        state = replace(state, reflects=reflects, value_reflects=value_reflects)

        return state

    @staticmethod
    async def act(model: Model, state: GameState_rafa, **kwargs) -> Any:
        required_keys = ["n_generate_sample", "request_params", "request_id", "namespace","n_propose_sample","n_select_sample","to_print","cache_value"]

        for key in required_keys:
            if key not in kwargs or kwargs[key] is None:
                raise ValueError(f"Missing required parameter: '{key}'")

        typed_kwargs = cast(ActKwargs_rafa, kwargs)

        n_generate_sample = typed_kwargs["n_generate_sample"]
        request_params = typed_kwargs["request_params"]
        request_id = typed_kwargs["request_id"]
        namespace = typed_kwargs["namespace"]
        n_propose_sample=typed_kwargs["n_propose_sample"]
        n_select_sample=typed_kwargs["n_select_sample"]
        to_print=typed_kwargs["to_print"]
        cache_value=typed_kwargs["cache_value"]
        puzzle=state.puzzle
        state = GameState_rafa()
        state= replace(state, puzzle=puzzle)
        if len(state.obs_feedback) >= 1:
            state = await AgentRafaGame24_act.reflect_rafa(state=state,
                                                           n_generate_sample=n_generate_sample,
                                                           model=model,
                                                           request_params=request_params,
                                                           namespace=namespace,
                                                           request_id=request_id)

            # model: Model, state: GameState_rafa, n_propose_sample, n_select_sample, to_print, cache_value,
            # value_cache, n_evaluate_sample,
            # request_parameters

        state, action, info = await AgentRafaGame24_act.plan_rafa(
            model=model,
            state=state,
            n_propose_sample=n_propose_sample,
            n_select_sample=n_select_sample,
            to_print=to_print,
            cache_value=cache_value,
            request_parameters=request_params
        )
        return state, action, info


class AgentRafaGame24_eval(Agent):
    @staticmethod
    def step_rafa(action, state: GameState_rafa, feedback_print: bool, max_step):
        state = replace(state, cur_step=state.cur_step + 1)
        prev_len = len(state.history)
        generated_state, feedback, reward = AgentRafaGame24_act.generate_feedback_rafa(action=action,
                                                                                       state=state,
                                                                                       feedback_print=feedback_print,
                                                                                       )
        new_len = len(state.history)
        delta = new_len - prev_len + 1 if new_len < 4 else new_len - prev_len
        assert delta > 0
        done = (reward >= 10) or (generated_state.cur_step > max_step)
        answer = [f"Step {i + 1}: {x}" for i, x in enumerate(action.split('\n')[:delta]) if x != ""]
        answer = "Attempt answer: " + "\n".join(answer)
        if generated_state.feedback:
            info = {'action': action, 'history': generated_state.history}
            obs = {'answer': answer, 'feedback': feedback}
        else:
            info = {'action': action, 'history': []}
            obs = {'answer': answer, 'feedback': []}
        return generated_state, obs, reward, done, info

    @staticmethod
    def act(model: Model, state: GameState_rafa, **kwargs: Any) -> Any:
        required_keys = ["feedback_print", "action"]

        for key in required_keys:
            if key not in kwargs or kwargs[key] is None:
                raise ValueError(f"Missing required parameter: '{key}'")

        typed_kwargs = cast(EvalKwargs_rafa, kwargs)

        feedback_print = typed_kwargs["feedback_print"]
        action = typed_kwargs["action"]

        return AgentRafaGame24_eval.step_rafa(action=action,
                                              state=state,
                                              feedback_print=feedback_print)
