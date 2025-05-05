import asyncio
import itertools
from dataclasses import replace
from typing import TypedDict

from ..algorithm_options.rafa import RAFAOptions, RequestOptions
from ..typedefs import Algorithm, Model, Agent, Environment, Benchmark, State


class AgentDictRAFA(TypedDict):
    agent_reflect: Agent
    agent_reflect_value: Agent
    agent_plan: Agent
    agent_plan_evaluate: Agent
    agent_eval: Agent


class AlgorithmRAFA(Algorithm):

    def __init__(self,
                 model: Model,
                 agents: AgentDictRAFA,
                 env: Environment,
                 rafa_options: RAFAOptions,
                 use_local_cache: bool = False):

        super().__init__(model, agents, env)

        self.rafa_options = rafa_options

        # the agent to reflect: agent_reflect
        self.agent_reflect = agents['agent_reflect']

        # the agent to evaluate each reflect:agent_reflect_value
        self.agent_reflect_value = agents['agent_reflect_value']

        # the agent to plan:agent_plan
        self.agent_plan = agents['agent_plan']

        # the agent to evaluate the plan: agent_plan_evaluate
        self.agent_plan_evaluate = agents['agent_plan_evaluate']

        # the agent that evaluate after each loop in the while loop: agent_eval
        self.agent_eval = agents['agent_eval']

    async def solve(self, idx: int, state: State, namespace: str, value_cache: dict = None):

        # Initial state
        state = state
        request_options = RequestOptions(max_completion_tokens=200,
                                         temperature=1.0,
                                         top_p=1.0,
                                         logprobs=False,
                                         request_id=f"idx{idx}-step{0}-{hash(state)}-agent{0}",
                                         namespace=namespace)

        done = False
        ##-------These two methods are from the obs structure they had
        observations = {
            "action": "",
            "feedback": []
        }
        reflects_list = []
        value_reflects_list = []

        self_feedbacks = []
        self_history = []
        self_cur_step = 0

        ##-------
        i = 0
        while not done:
            request_options.request_id = f"idx{idx}-step{i}-{hash(state)}-agent{0}"
            i += 1

            if len(observations['feedback']) >= 1:
                request_options.request_id = f"idx{idx}-step{i}-{hash(state)}-reflect{i}"
                reflects = await self.agent_reflect.act(model=self.model,
                                                        state=state,
                                                        request_options=request_options,
                                                        n_propose_sample=self.rafa_options.n_propose_sample,
                                                        observations_answer=observations['answer'],
                                                        observations_feedback=observations['feedback']

                                                        )

                reflects_list.extend(reflects)

                # score reflects
                request_options.request_id = f"idx{idx}-step{i}-{hash(state)}-reflect_value{i}"
                value_reflects = await  self.agent_reflect_value.act(model=self.model,
                                                                     state=state,
                                                                     request_options=request_options,
                                                                     n_propose_sample=self.rafa_options.n_propose_sample,
                                                                     observations_answer=observations['answer'],
                                                                     observations_feedback=observations['feedback']

                                                                     )

                # update the value reflects
                value_reflects_list.extend(value_reflects)

            # -------------------------------------The plan begins
            current_output_candidates = ["\n".join(self_history) + "\n"] if len(self_history) else [
                ""]  # current output candidates
            infos = []
            for step in range(4 - len(self_history)):
                # get proposals (plan suggestions/generate)
                coroutines = []
                for output_candidate in current_output_candidates:
                    request_options.request_id = f"idx{idx}-step{i}-step-in-history{step}-{hash(state)}-plan-{output_candidate}"
                    coroutine = self.agent_plan.act(model=self.model,
                                                    state=state,
                                                    request_options=request_options,
                                                    candidate=output_candidate,
                                                    reflects_list=reflects_list,
                                                    n_propose_sample=self.rafa_options.n_propose_sample,
                                                    n_generate_sample=self.rafa_options.n_generate_sample
                                                    )
                    coroutines.append(coroutine)

                new_output_candidates = await asyncio.gather(*coroutines)
                new_output_candidates = list(itertools.chain(*new_output_candidates))

                # Evaluate proposals(evaluate plan suggestions/evaluate what has been generated)
                request_options.request_id = f"idx{idx}-step{i}-{hash(state)}-plan_evaluate"
                values = await self.agent_plan_evaluate.act(model=self.model,
                                                            state=state,
                                                            request_options=request_options,
                                                            new_output_candidates=new_output_candidates,
                                                            value_reflects=value_reflects_list,
                                                            n_evaluate_sample=self.rafa_options.n_evaluate_sample
                                                            )
                selected_top_candidates_with_score = sorted(values, key=lambda x: x[1], reverse=True)[
                                                     :self.rafa_options.n_select_sample]
                best_candidates_list = [candidate for candidate, _ in selected_top_candidates_with_score]

                # for logging i guess
                infos.append(
                    {'step': step,
                     'x': state.puzzle,
                     'ys(current output candidates)': current_output_candidates,
                     'new_ys(output candidates)': new_output_candidates,
                     'values': values,
                     'select_new_ys(best scored candidates)': best_candidates_list})

                current_output_candidates = best_candidates_list

            ys_list = [y.split('\n')[len(self_history):] for y in current_output_candidates]
            res_ys = ["\n".join(ys) for ys in ys_list][0]

            # generating feedback:
            obs, reward, done, env_info, self_history1, self_feedbacks1, self_curstep1 = self.agent_eval.act(
                model=self.model,
                state=state,
                puzzle=state.puzzle,
                action=res_ys,
                max_steps=self.rafa_options.max_step,
                cur_step=self_cur_step,
                history=self_history,
                feedbacks=self_feedbacks,
                max_step=self.rafa_options.max_step,
            )
            # update env (this should be done with a function at some point)
            # todo create env update function
            # self_feedbacks.append(self_feedbacks1)
            # if self_history1:
            #     self_history.extend(self_history1)
            self_feedbacks = self_feedbacks1
            self_history = self_history1
            self_cur_step = self_curstep1
            observations = obs
            state = replace(state, steps = self_history)

        return state

    async def benchmark(self, benchmark: Benchmark, share_ns: bool = False, cache: bool = True):
        cache = {} if cache else None
        solve_coroutines = [
            self.solve(
                idx=index,
                state=state,
                namespace="benchmark" if share_ns else f"benchmark-{index}",
                value_cache=cache
            )
            for index, state in benchmark
        ]
        results = await asyncio.gather(*solve_coroutines)
        return results
