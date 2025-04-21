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
            output_candidates = ["\n".join(state.history) + "\n"] if len(state.history) else [
                ""]  # current output candidates
            infos = []
            for step in range(4 - len(state.history)):
                # get proposals (plan suggestions/generate)
                coroutines = []
                for output_candidate in enumerate(output_candidates):
                    request_options.request_id = f"idx{idx}-step{i}-{hash(state)}-plan-{output_candidate}"
                    coroutine = self.agent_plan.act(model=self.model,
                                                    state=state,
                                                    request_options=request_options,
                                                    candidate=output_candidate,
                                                    reflectsreflects=reflects_list,
                                                    n_propose_sample=self.rafa_options.n_propose_sample,
                                                    n_generate_sample=self.rafa_options.n_generate_sample
                                                    )
                    coroutines.append(coroutine)

                new_output_candidates = await asyncio.gather(*coroutines)

                new_output_candidates = list(itertools.chain(*new_output_candidates))
                ids = list(range(len(new_output_candidates)))
                # Evaluate proposals(evaluate plan suggestions/evaluate what has been generated)
                request_options.request_id = f"idx{idx}-step{i}-{hash(state)}-plan_evaluate"
                values = await self.agent_plan_evaluate.act(model=self.model,
                                                            state=state,
                                                            request_options=request_options,
                                                            new_output_candidates=new_output_candidates,
                                                            value_reflects=value_reflects_list,
                                                            n_evaluate_sample=self.rafa_options.n_evaluate_sample,

                                                            )
                select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:self.rafa_options.n_select_sample]
                select_new_ys = [new_output_candidates[select_id] for select_id in select_ids]
                infos.append(
                    {'step': step, 'x': state.puzzle, 'ys': output_candidates, 'new_ys': new_output_candidates,
                     'values': values,
                     'select_new_ys': select_new_ys})
                output_candidates = select_new_ys

            ys_list = [y.split('\n')[len(state.history):] for y in output_candidates]
            res_ys = ["\n".join(ys) for ys in ys_list][0]

            state = state
            res_ys = res_ys
            env_info = {'steps': infos}

            ##Generating feedback for the progress so far(last step in the old structure)

            state, obs, reward, done, env_info = self.agent_eval.act(model=self.model,
                                                                     state=state,
                                                                     action=res_ys,
                                                                     )
            # todo i think this is where we update with a step if types match, to be checked
            #comment for self, I think this is where the env should be updated...
            # action = agent.act(state)
            # new_state = environment.step(state, action)
            # if done:
            #     state = replace(state, reflects=[], value_reflects=[])
            #     i = 0

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
