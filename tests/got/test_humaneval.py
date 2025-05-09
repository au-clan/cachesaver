import pytest
import tempfile
import asyncio
import itertools
from diskcache import Cache
from cachesaver.pipelines import OnlineAPI

from src.algorithms import AgentDictGOT, AlgorithmGOT
from src.models import API
from src.typedefs import DecodingParameters
from src.tasks.humaneval import (
    EnvironmentHumanEval,
    AgentActHumanEval,
    AgentAggregateHumanEval,
    AgentEvaluateHumanEval,
    StateHumanEval,
    BenchmarkHumanEval,
)


class TestGoTHumanEval:
    llm = "llama-3.3-70b-versatile"
    params = DecodingParameters(
        temperature=0.7,
        max_completion_tokens=8192,
        top_p=1.0,
        stop=None,
        logprobs=False,
    )

    @pytest.fixture(scope="function")
    def cache(self):
        """Provide a temporary cache for each test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with Cache(tmpdir) as cache:
                yield cache

    @pytest.mark.asyncio()
    async def test_aggregate_humaneval(self, offline_model) -> None:
        benchmark = BenchmarkHumanEval("datasets/humaneval-py-sorted.csv.gz", "mini")
        generate = AgentActHumanEval()
        aggregate = AgentAggregateHumanEval
        _, state = benchmark[0]
        states = [state.clone(randomness=1235)]

        # Groq does not support n>1, so we just simulate it
        generate_coroutines = [
            generate.act(
                model=offline_model,
                state=state,
                n=1,
                namespace="test_small",
                request_id=0,
                params=self.params,
            )
            for state in states
            for _ in range(3)
        ]
        actions = await asyncio.gather(*generate_coroutines)
        actions = list(itertools.chain(*actions))

        aggregate_coroutines = [
            aggregate.act(
                model=offline_model,
                state=state,
                actions=action,
                k=2,
                namespace="test_small",
                request_id=0,
                params=self.params,
            )
            for state, action in zip(states, actions)
        ]
        agg_actions = await asyncio.gather(*aggregate_coroutines)
        agg_actions = list(itertools.chain(*agg_actions))

        assert len(agg_actions) == 2

    @pytest.mark.asyncio()
    async def test_got_humaneval(self, offline_model) -> None:
        benchmark = BenchmarkHumanEval("datasets/humaneval-py-sorted.csv.gz", "mini")
        benchsmark_small = [(idx, state) for idx, state in benchmark][:3]

        env = EnvironmentHumanEval()
        agents = AgentDictGOT(
            step=AgentActHumanEval,
            aggregate=AgentAggregateHumanEval,
            evaluate=AgentEvaluateHumanEval,
            step_params=self.params,
            aggregate_params=self.params,
            eval_params=self.params,
        )
        algorithm = AlgorithmGOT(
            model=offline_model,
            agents=agents,
            env=env,
            num_selections=2,
            num_steps=1,
            num_generate=5,
            num_best=2,
            num_evaluations=1,
        )

        result_coroutine = [
            algorithm.solve(
                idx=i,
                state=state,
                namespace="test_small",
                value_cache=None,
            )
            for i, state in benchsmark_small
        ]
        results = await asyncio.gather(*result_coroutine)

        finished = []
        for result in results:
            for r in result:
                finish, _ = env.evaluate(r)
                finished.append(finish)

        assert any(finished)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_got_humaneval_with_cachesaver(self, cache, online_model) -> None:
        benchmark = BenchmarkHumanEval("datasets/humaneval-py-sorted.csv.gz", "mini")
        benchsmark_small = [(idx, state) for idx, state in benchmark][:3]

        env = EnvironmentHumanEval()
        agents = AgentDictGOT(
            step=AgentActHumanEval,
            aggregate=AgentAggregateHumanEval,
            evaluate=AgentEvaluateHumanEval,
            step_params=self.params,
            aggregate_params=self.params,
            eval_params=self.params,
        )

        async with OnlineAPI(
            model=online_model,
            cache=cache,
            batch_size=2,
            timeout=0.1,
        ) as pipeline:
            api = API(
                pipeline=pipeline,
                model=self.llm,
            )

            algorithm = AlgorithmGOT(
                model=api,
                agents=agents,
                env=env,
                num_selections=1,
                num_steps=1,
                num_generate=1,
                num_best=1,
                num_evaluations=1,
            )

            results_coroutine = [
                algorithm.solve(
                    idx=i,
                    state=state,
                    namespace="test_small",
                    value_cache={},
                )
                for i, state in benchsmark_small
            ]

            results = await asyncio.gather(*results_coroutine)

            finished = []
            for result in results:
                for r in result:
                    finish, _ = env.evaluate(r)
                    finished.append(finish)
            
            assert any(finished)