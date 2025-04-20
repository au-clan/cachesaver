import asyncio
import itertools
from dataclasses import replace
from typing import TypedDict

from ..algorithm_options.rafa import RAFAOptions, RequestOptions, GameState_rafa
from ..typedefs import Algorithm, Model, Agent, Environment, Benchmark, State


class AgentDictRAFA_tot(TypedDict):
    agent_reflect: Agent
    agent_reflect_value: Agent
    agent_plan: Agent
    agent_plan_evaluate: Agent
    agent_eval: Agent


class AlgorithmRAFA_tot(Algorithm):

    def __init__(self,
                 model: Model,
                 agents: AgentDictRAFA_tot,
                 env: Environment,
                 rafa_options: RAFAOptions,
                 value_cache: dict = None):

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
        self.value_cache = value_cache  # todo utilize if it is None it means we dont want to cache values

    async def solve(self, idx: int, state: State, namespace: str, value_cache: dict = None):
        # Initial state
        # initial_state = self.reset_rafa(state.puzzle)  # todo verify puzzle is accessible

        request_options = RequestOptions(max_completion_tokens=200,
                                         temperature=1.0,
                                         top_p=1.0,
                                         logprobs=False,
                                         request_id=f"idx{idx}-step{0}-{hash(state)}-agent{0}",
                                         namespace=namespace)

        done = False
        i = 0
        while not done:
            request_options.request_id = f"idx{idx}-step{i}-{hash(state)}-agent{0}"
            i += 1
            # todo this should return the cache_value dict if it should be stored across puzzles...
            # state, action, agent_info = await self.agent_act.act(state=state,
            #                                                      model=self.model,
            #                                                      request_options=request_options,
            #                                                      value_cache=self.value_cache,
            #                                                      rafa_options=self.rafa_options
            #                                                      )
            ##inside these brackets is the new agent structure[[

            puzzle = state.puzzle
            state = GameState_rafa()
            state = replace(state, puzzle=puzzle)

            # reflect
            if len(state.obs_feedback) >= 1:
                reflects = await self.agent_reflect.act(state=state,
                                                        model=self.model,
                                                        request_options=request_options,
                                                        value_cache=self.value_cache,
                                                        rafa_options=self.rafa_options)
                value_reflects = await  self.agent_reflect_value.act(state=state,
                                                                     model=self.model,
                                                                     request_options=request_options,
                                                                     value_cache=self.value_cache,
                                                                     rafa_options=self.rafa_options)
                # collet results in state
                state = replace(state, reflects=reflects, value_reflects=value_reflects)

            # plan and eval plan
            ys = ["\n".join(state.env_history) + "\n"] if len(state.env_history) else [""]  # current output candidates
            infos = []
            for step in range(4 - len(state.env_history)):
                # get proposals (plan suggestions)
                coroutines = [
                    # todo confirm the right attributes passed cache prob missing
                    self.agent_plan.act(puzzle=state.puzzle,
                                        y=y,
                                        rafa_options=self.rafa_options,
                                        request_options=request_options,
                                        model=self.model,
                                        )
                    for y in ys]
                new_ys = await asyncio.gather(*coroutines)

                new_ys = list(itertools.chain(*new_ys))
                ids = list(range(len(new_ys)))
                # Evaluate proposals(evaluate plan suggestions)
                # todo for sure this isnot correct arguments but fix later
                values = await self.agent_plan_evaluate.act(puzzle=state.puzzle,
                                                            new_ys=new_ys,
                                                            rafa_options=self.rafa_options,
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
            # return state, res_ys, {'steps': infos}#todo this is not correct format
            state = state
            res_ys = res_ys
            env_info = {'steps': infos}

            ##Generating feedback for the progress so far(last step in the old structure)
            # todo do we want it to be agent if it doesnt prompt? it is somewhat task specific
            state, obs, reward, done, env_info = self.agent_eval.act(state=state,
                                                                     model=self.model,
                                                                     action=res_ys,
                                                                     )

            ##----------------------------------------------------------
            # 2: reflect

            if done:
                state = replace(state, reflects=[], value_reflects=[])
                i = 0

            print(obs)
            print(reward, done, env_info)

            # logs = logs + [log]
        # return logs

        # correct = 0
        # for i in range(len(logs)):
        #     # is_correct = self.verification_helper(logs[i]['obs_step'][-1]['answer'])
        #     # if is_correct:
        #     #     correct += 1
        # # verifications = [self.environment.verify(state) for state in states]
        return state  # todo refactor to old state type
        # return correct  ##todo we should return the correct format and not just 0 aorn

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
