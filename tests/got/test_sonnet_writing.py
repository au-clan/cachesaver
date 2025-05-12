from os import name
import sys
import asyncio
import pytest
import tempfile
from diskcache import Cache
from cachesaver.pipelines import OnlineAPI
import itertools

from src.algorithms import AgentDictGOT, AlgorithmGOT
from src.models import API
from src.tasks.sonnetwriting.benchmark import BenchmarkSonnetWriting
from src.typedefs import Algorithm, DecodingParameters
from src.tasks.sonnetwriting import (
    EnvironmentSonnetWriting,
    AgentActSonnetWriting,
    AgentAggregateSonnetWriting,
    AgentEvaluateSonnetWriting,
    StateSonnetWriting,
)


class TestGoTSonnetWriting:
    llm = "llama-3.3-70b-versatile"
    params = DecodingParameters(
        temperature=0.7,
        max_completion_tokens=4000,
        top_p=1.0,
        stop=None,
        logprobs=False,
    )
    env = EnvironmentSonnetWriting()

    @pytest.fixture(scope="function")
    def cache(self):
        """Provide a temporary cache for each test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with Cache(tmpdir) as cache:
                yield cache

    @pytest.mark.asyncio()
    async def test_act_sonnet_writing(self, offline_model) -> None:
        state = StateSonnetWriting(
            puzzle='Write a sonnet with strict rhyme scheme ABAB CDCD EFEF GG, containing each of the following words verbatim: "grass", "value", and "jail".',
            current_state='Write a sonnet with strict rhyme scheme ABAB CDCD EFEF GG, containing each of the following words verbatim: "grass", "value", and "jail".',
            steps=[],
            target="ABAB CDCD EFEF GG, grass value jail",
            randomness=1234,
        )

        act = AgentActSonnetWriting()
        sonnet = await act.act(
            offline_model,
            state,
            n=1,
            namespace="test_small",
            request_id=0,
            params=self.params,
        )
        assert len(sonnet[0].split("\n")) == len(state.target.split(", ")[0])

    @pytest.mark.asyncio()
    async def test_aggregate_sonnet_writing(self, offline_model) -> None:
        state = StateSonnetWriting(
            puzzle='Write a sonnet with strict rhyme scheme ABAB CDCD EFEF GG, containing each of the following words verbatim: "grass", "value", and "jail".',
            current_state='Write a sonnet with strict rhyme scheme ABAB CDCD EFEF GG, containing each of the following words verbatim: "grass", "value", and "jail".',
            steps=[],
            target="ABAB CDCD EFEF GG, grass value jail",
            randomness=1234,
        )

        act = AgentActSonnetWriting()
        act_coroutine = [
            act.act(
                offline_model,
                state,
                n=1,
                namespace="test_small",
                request_id=0,
                params=self.params,
            )
            for _ in range(5)
        ]
        act_actions = await asyncio.gather(*act_coroutine)
        act_actions = list(itertools.chain(*act_actions))
        assert all(len(action.split('\n')) == len(state.target.split(', ')[0]) for action in act_actions)

        aggregate = AgentAggregateSonnetWriting()
        actions = await aggregate.act(
            model=offline_model,
            state=state,
            actions=act_actions,
            k=3,
            namespace="test_small",
            request_id=0,
            params=self.params,
        )

        assert len(actions) == 3
        assert all(agg_action in act_actions for agg_action in actions)

    @pytest.mark.asyncio()
    async def test_evaluate_sonnet_writing(self, offline_model) -> None:
        state = StateSonnetWriting(
            puzzle='Write a sonnet with strict rhyme scheme ABAB CDCD EFEF GG, containing each of the following words verbatim: "grass", "value", and "jail".',
            current_state='Write a sonnet with strict rhyme scheme ABAB CDCD EFEF GG, containing each of the following words verbatim: "grass", "value", and "jail".',
            steps=[],
            target="ABAB CDCD EFEF GG, grass value jail",
            randomness=1234,
        )

        act = AgentActSonnetWriting()
        act_coroutine = [
            act.act(
                offline_model,
                state,
                n=1,
                namespace="test_small",
                request_id=0,
                params=self.params,
            )
            for _ in range(5)
        ]
        act_actions = await asyncio.gather(*act_coroutine)
        act_actions = list(itertools.chain(*act_actions))
        assert all(len(action.split('\n')) == len(state.target.split(', ')[0]) for action in act_actions)

        states = []
        for action in act_actions:
            states.append(self.env.step(state, action))

        evaluate = AgentEvaluateSonnetWriting()
        evaluation_coroutines = [
            evaluate.act(
                offline_model,
                state=state,
                n=1,
                namespace="test_small",
                request_id=0,
                params=self.params
            )
            for state in states
        ]
        evaluations = await asyncio.gather(*evaluation_coroutines)

        assert all(isinstance(value, float) for value in evaluations)

    @pytest.mark.asyncio()
    async def test_got_sonnet_writing(self, offline_model_openai) -> None:
        benchmark = BenchmarkSonnetWriting("datasets/dataset_sonnetwriting.jsonl.gz", "single")
        env = EnvironmentSonnetWriting()
        agents = AgentDictGOT(
            step=AgentActSonnetWriting,
            aggregate=AgentAggregateSonnetWriting,
            evaluate=AgentEvaluateSonnetWriting,
            step_params=self.params,
            aggregate_params=self.params,
            eval_params=self.params,
        )
        algorithm = AlgorithmGOT(
            model=offline_model_openai,
            agents=agents,
            env=env,
            num_selections=3,
            num_steps=1,
            num_generate=5,
            num_best=2,
            num_evaluations=3,
        )

        result_coroutine = [
            algorithm.solve(
                idx=i,
                state=state,
                namespace="test_small",
                value_cache=None,
            )
            for i, state in benchmark
        ]
        results = await asyncio.gather(*result_coroutine)

        finished = []
        for result in results:
            for r in result:
                finish, _ = env.evaluate(r)
                finished.append(finish)

        assert len(finished) > 0
        assert any(finished)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_got_sonnet_writing_cachesaver(self, cache, online_model_openai) -> None:
        benchmark = BenchmarkSonnetWriting("datasets/dataset_sonnetwriting.jsonl.gz", "mini")
        benchmark_small = [(idx, state) for idx, state in benchmark][:5]

        env = EnvironmentSonnetWriting()
        agents = AgentDictGOT(
            step=AgentActSonnetWriting,
            aggregate=AgentAggregateSonnetWriting,
            evaluate=AgentEvaluateSonnetWriting,
            step_params=self.params,
            aggregate_params=self.params,
            eval_params=self.params,
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

            algorithm = AlgorithmGOT(
                model=api,
                agents=agents,
                env=env,
                num_selections=3,
                num_steps=1,
                num_generate=5,
                num_best=2,
                num_evaluations=3,
            )

            results_coroutine = [
                algorithm.solve(
                    idx=i,
                    state=state,
                    namespace="test_small",
                    value_cache={},
                )
                for i, state in benchmark_small
            ]

            results = await asyncio.gather(*results_coroutine)

            finished = []
            for result in results:
                for r in result:
                    finish, _ = env.evaluate(r)
                    finished.append(finish)

            assert any(finished)