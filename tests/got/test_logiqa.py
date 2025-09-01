import pytest
import tempfile
import asyncio
import itertools
from diskcache import Cache
from cachesaver.pipelines import OnlineAPI

from src.methods import AgentDictGOT, MethodGOT
from src.models import API
from src.typedefs import DecodingParameters
from src.tasks.logiqa import (
    EnvironmentLogiQA,
    AgentActLogiQA,
    AgentAggregateLogiQA,
    AgentEvaluateLogiQA,
    StateLogiQA,
    BenchmarkLogiQA,
)


class TestGoTLogiQA:
    params = DecodingParameters(
        temperature=0.7,
        max_completion_tokens=100,
        top_p=1.0,
        stop=None,
        logprobs=False,
    )
    benchmark = BenchmarkLogiQA("datasets/dataset_logiqa.csv.gz", "mini")
    env = EnvironmentLogiQA()

    @pytest.mark.asyncio()
    async def test_act_agent(self, offline_model_openai) -> None:
        _, state = self.benchmark[0]

        act = AgentActLogiQA()

        actions = await act.act(
            model=offline_model_openai,
            state=state,
            n=3,
            namespace="test_small",
            request_id=0,
            params=self.params,
        )

        assert len(actions) == 3
        assert all(a in "abcd" for a in actions)

    @pytest.mark.asyncio()
    async def test_aggregate_agent(self, offline_model_openai) -> None:
        _, state = self.benchmark[0]

        act = AgentActLogiQA()

        actions = await act.act(
            model=offline_model_openai,
            state=state,
            n=5,
            namespace="test_small",
            request_id=0,
            params=self.params,
        )

        aggregate = AgentAggregateLogiQA()

        selected_actions = await aggregate.act(
            model=offline_model_openai,
            state=state,
            actions=actions,
            k=3,
            namespace="test_small",
            request_id=0,
            params=self.params,
        )

        assert len(selected_actions) == 3
        assert all(agg_action in actions for agg_action in selected_actions)

    @pytest.mark.asyncio()
    async def test_evaluate_agent(self, offline_model_openai) -> None:
        _, state = self.benchmark[0]

        act = AgentActLogiQA()
        actions = await act.act(
            model=offline_model_openai,
            state=state,
            n=3,
            namespace="test_small",
            request_id=0,
            params=self.params,
        )

        proposed_states = []
        for action in actions:
            proposed_states.append(self.env.step(state, action))

        evaluate = AgentEvaluateLogiQA()
        evaluation = await evaluate.act(
            model=offline_model_openai,
            state=state,
            n=2,
            namespace="test_small",
            request_id=0,
            params=self.params,
            cache={},
        )

        assert isinstance(evaluation, float)
        assert evaluation > 0

    @pytest.mark.asyncio()
    async def test_got_logiqa(self, offline_model_openai) -> None:
        benchmark_small = [(idx, state) for idx, state in self.benchmark][:5]

        agents = AgentDictGOT(
            step=AgentActLogiQA,
            aggregate=AgentAggregateLogiQA,
            evaluate=AgentEvaluateLogiQA,
            step_params=self.params,
            aggregate_params=self.params,
            evaluate_params=self.params,
        )
        method = MethodGOT(
            model=offline_model_openai,
            agents=agents,
            env=self.env,
            num_selections=3,
            num_steps=1,
            num_generate=5,
            num_best=2,
            num_evaluations=3,
        )

        result_coroutine = [
            method.solve(
                idx=i,
                state=state,
                namespace="test_small",
                value_cache=None,
            )
            for i, state in benchmark_small
        ]
        results = await asyncio.gather(*result_coroutine)

        finished = []
        for result in results:
            for r in result:
                finish, _ = self.env.evaluate(r)
                finished.append(finish)

        assert all(finished)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_got_logiqa_with_cachesaver(self, cache, online_model_openai) -> None:
        benchmark_small = [(idx, state) for idx, state in self.benchmark][:5]

        agents = AgentDictGOT(
            step=AgentActLogiQA,
            aggregate=AgentAggregateLogiQA,
            evaluate=AgentEvaluateLogiQA,
            step_params=self.params,
            aggregate_params=self.params,
            evaluate_params=self.params,
        )

        async with OnlineAPI(
            model=online_model_openai,
            cache=cache,
            batch_size=10,
            timeout=0.1,
        ) as pipeline:
            api = API(
                pipeline=pipeline,
                model="gpt-4.1-nano"
            )

            method = MethodGOT(
                model=api,
                agents=agents,
                env=self.env,
                num_selections=3,
                num_steps=1,
                num_generate=5,
                num_best=2,
                num_evaluations=3,
            )

            result_coroutine = [
                method.solve(
                    idx=i,
                    state=state,
                    namespace="test_small",
                    value_cache={},
                )
                for i, state in benchmark_small
            ]

            results = await asyncio.gather(*result_coroutine)

            finished, correct = [], []

            for result in results:
                for r in result:
                    finish, value = self.env.evaluate(r)
                    finished.append(finish)
                    correct.append(value)

            assert sum(correct) > 0
            assert all(finished)