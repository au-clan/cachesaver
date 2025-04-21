import asyncio
import itertools
from dataclasses import replace
from typing import TypedDict

from ..algorithm_options.rafa import RAFAOptions, RequestOptions, GameState_rafa
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

        # This value_cache should be used as a caching mechanism
        self.value_cache = {} if use_local_cache else None

    async def solve(self, idx: int, state: State, namespace: str, value_cache: dict = None):
        # Initial state

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
        # these two should be cleared after each puzzle

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
                reflects_list.append(reflects)

                # score reflects?
                request_options.request_id = f"idx{idx}-step{i}-{hash(state)}-reflect_value{i}"
                value_reflects = await  self.agent_reflect_value.act(model=self.model,
                                                                     state=state,
                                                                     request_options=request_options,
                                                                     n_propose_sample=self.rafa_options.n_propose_sample,
                                                                     observations_answer=observations['answer'],
                                                                     observations_feedback=observations['feedback']

                                                                     )
                # update the value reflects
                value_reflects_list.append(value_reflects)

            # -------------------------------------Now the plan begins
            ys = ["\n".join(state.env_history) + "\n"] if len(state.env_history) else [""]  # current output candidates
            infos = []
            for step in range(4 - len(state.env_history)):
                # get proposals (plan suggestions)
                coroutines = []
                for y in enumerate(ys):
                    request_options.request_id = f"idx{idx}-step{i}-{hash(state)}-plan-{y}"
                    coroutine = self.agent_plan.act(
                        state=state,
                        puzzle=state.puzzle,
                        y=y,
                        n_propose_sample=self.rafa_options.n_propose_sample,
                        n_generate_sample=self.rafa_options.n_generate_sample,
                        request_options=request_options,
                        model=self.model,
                    )
                    coroutines.append(coroutine)

                new_ys = await asyncio.gather(*coroutines)

                new_ys = list(itertools.chain(*new_ys))
                ids = list(range(len(new_ys)))
                # Evaluate proposals(evaluate plan suggestions)
                # todo for sure this isnot correct arguments but fix later
                request_options.request_id = f"idx{idx}-step{i}-{hash(state)}-plan_evaluate"
                values = await self.agent_plan_evaluate.act(puzzle=state.puzzle,
                                                            state=state,
                                                            new_ys=new_ys,
                                                            n_evaluate_sample=self.rafa_options.n_evaluate_sample,
                                                            request_options=request_options,
                                                            model=self.model)
                select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:self.rafa_options.n_select_sample]
                select_new_ys = [new_ys[select_id] for select_id in select_ids]
                infos.append(
                    {'step': step, 'x': state.puzzle, 'ys': ys, 'new_ys': new_ys, 'values': values,
                     'select_new_ys': select_new_ys})
                ys = select_new_ys

            ys_list = [y.split('\n')[len(state.history):] for y in ys]
            res_ys = ["\n".join(ys) for ys in ys_list][0]

            state = state
            res_ys = res_ys
            env_info = {'steps': infos}

            ##Generating feedback for the progress so far(last step in the old structure)

            state, obs, reward, done, env_info = self.agent_eval.act(state=state,
                                                                     model=self.model,
                                                                     action=res_ys,
                                                                     )

            if done:
                state = replace(state, reflects=[], value_reflects=[])
                i = 0

            print(obs)
            print(reward, done, env_info)

        return state  # todo refactor to old state type

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
