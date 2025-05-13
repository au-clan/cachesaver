import random
import re
from typing import Any

from . import prompts as prompts, StateHotpotQA
from ...algorithm_options.rafa import RafaRequest
from ...typedefs import Agent, Model, DecodingParameters


class AgentRafaHotpotQA_eval(Agent):

    @staticmethod
    def act(model: Model, state: StateHotpotQA, **kwargs) -> Any:
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

        # todo this methods should should maybe be the reflextion agent else lets discuss later?
        raise NotImplemented
        action = kwargs["action"]
        puzzle = kwargs["puzzle"]
        max_steps = kwargs["max_steps"]
        cur_step = kwargs["cur_step"]
        history = kwargs["history"]
        self_feedbacks = kwargs["feedbacks"]
        cur_step += 1
        prev_len = len(history)
        # old signature and method for generating feedback look in AgentRafaGame24_eval for old impl
        feedback, reward, self_history, self_feedbacks = AgentRafaHotpotQA_eval.generate_feedback(action=action,
                                                                                                  history=history,
                                                                                                  puzzle=puzzle,
                                                                                                  feedbacks=self_feedbacks)

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
    async def act(model: Model, state: StateHotpotQA, **kwargs) -> Any:
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
            params=DecodingParameters(
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
    async def act(model: Model, state: StateHotpotQA, **kwargs) -> Any:
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
                                             params=DecodingParameters(
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
    async def act(model: Model, state: StateHotpotQA, **kwargs) -> Any:
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

        prompt = "Solve a question answering task with sequential Action steps. Action can be three types:\
                        (1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.\
                        (2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.\
                        (3) Finish[answer], which returns the answer and finishes the task.\
                        You may take as many steps as necessary."

        # have they forgotten to use answer? or not supported feature yet(from them??)
        history = [{"feedback": prompt},
                   {"feedback": "What you have learned is summarized below.\n" + "\n".join(
                       reflects_list)}]

        num_examples = 2
        examples = "(Example)\n" + "\n\n(Example)\n".join([example for example in prompts.examples_bfs[:num_examples]])
        propose_prompt = prompts.bfs.format(examples=examples, question=state.puzzle,
                                            current_state=state.current_state)  # here was the candidates used in the game of 24 rafa plan agent

        history_messages = RafaRequest.from_request_options(request_options=request_options,
                                                            n=n_generate_sample)

        for h in history:
            if 'answer' in h:
                history_messages.add_assistant_message(h["answer"])
            if 'feedback' in h:
                history_messages.add_user_message(h["feedback"])
        history_messages.add_user_message(propose_prompt)

        response = await model.request(prompt=history_messages.messages,
                                       n=history_messages.n,
                                       request_id=f"{history_messages.request_id}-randomnes-{random.randint(1, 10000)}",
                                       namespace=history_messages.namespace,
                                       params=DecodingParameters(
                                           max_completion_tokens=history_messages.max_completion_tokens,
                                           temperature=history_messages.temperature,
                                           top_p=history_messages.top_p,
                                           stop=history_messages.stop_token,
                                           logprobs=history_messages.logprobs,
                                       )
                                       )
        proposals = [r.strip() for r in response[0].split("\n")]
        return proposals[:min(len(proposals), n_propose_sample)]


class AgentRAFA_plan_evaluate(Agent):

    @staticmethod
    async def act(model: Model, state: StateHotpotQA, **kwargs) -> Any:
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

        prompt = "Solve a question answering task with sequential Action steps. Action can be three types:\
                                        (1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.\
                                        (2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.\
                                        (3) Finish[answer], which returns the answer and finishes the task.\
                                        You may take as many steps as necessary."
        history = [prompt,
                   dict(feedback="What you have learned is summarized below.\n" + "\n".join(
                       value_reflects))]
        values = []

        for candidate in new_output_candidates:  # each partial output

            num_examples = 2
            examples = "(Example)\n" + "\n\n(Example)\n".join(
                [example for example in prompts.examples_evaluate[:num_examples]])
            value_prompt = prompts.evaluate.format(examples=examples, question=state.puzzle,
                                                   current_state=state.current_state)

            # value_prompt = AgentRAFA_plan_evaluate.value_prompt_wrap(state.puzzle, candidate)

            history_messages = RafaRequest.from_request_options(request_options=request_options,
                                                                n=n_evaluate_sample)
            for h in history:
                if 'answer' in h:
                    history_messages.add_assistant_message(h["answer"])
                if 'feedback' in h:
                    history_messages.add_user_message(h["feedback"])
            history_messages.add_user_message(value_prompt)
            history_messages.request_id = f"step-{str(state.puzzle)}-{1}-{candidate}-{hash(1)}"

            responses = await model.request(prompt=history_messages.messages,
                                            n=history_messages.n,
                                            request_id=history_messages.request_id,
                                            namespace=history_messages.namespace,
                                            params=DecodingParameters(
                                                max_completion_tokens=history_messages.max_completion_tokens,
                                                temperature=history_messages.temperature,
                                                top_p=history_messages.top_p,
                                                stop=history_messages.stop_token,
                                                logprobs=history_messages.logprobs,
                                            )
                                            )

            # Parse the responses
            values = []
            pattern = r"\b(?:correctness[\s_]?score|score for correctness|correctness)\b(?:\s*(?:is|=|:|was|stands at|of))?\s*(-?\d+(?:\.\d+)?)"

            for response in responses:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    value = float(match.group(1))
                else:
                    # print(f"Unable to parse value from response : {response}")
                    value = 1
                values.append(value)
            value = sum(values)
            values.append((candidate, value))

        return values
