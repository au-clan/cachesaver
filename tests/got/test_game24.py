# imports
import sys
import asyncio
import pytest
import tempfile
from diskcache import Cache
from cachesaver.pipelines import OnlineAPI

from src.algorithms import AgentDictGOT, AlgorithmGOT
from src.models import API
from src.typedefs import ModelRequestOptions
from src.tasks.game24 import (
    EnvironmentGame24,
    AgentBfsGame24,
    AgentAggregateGame24,
    AgentEvaluateGame24,
    StateGame24,
)


if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class TestGoTOffline:
    params = ModelRequestOptions(
        temperature=0.7,
        max_completion_tokens=100,
        top_p=1.0,
        stop=None,
        logprobs=False,
    )

    @pytest.mark.asyncio()
    async def test_aggregate_game24(self, offline_model) -> None:
        state = StateGame24(
            puzzle="10 10 1 4",
            current_state="10 10 1 4",
            steps=[],
            randomness=0,
        )
        generate = AgentBfsGame24()
        aggregate = AgentAggregateGame24()

        generate_results = await generate.act(
            model=offline_model,
            state=state,
            namespace="test_small",
            request_id=0,
            params=self.params,
        )
        aggregate_results = await aggregate.act(
            model=offline_model,
            state=state,
            actions=generate_results,
            k=3,
            namespace="test_small",
            request_id=0,
            params=self.params,
        )
        assert len(aggregate_results) == 3

        aggregate_results = await aggregate.act(
            model=offline_model,
            state=state,
            actions=generate_results,
            k=2,
            namespace="test_small",
            request_id=1,
            params=self.params,
        )
        assert len(aggregate_results) == 2
        assert all(agg_step in generate_results for agg_step in aggregate_results)

    @pytest.mark.asyncio()
    async def test_got_game24(self, offline_model) -> None:
        state = StateGame24(
            puzzle="10 10 1 4",
            current_state="10 10 1 4",
            steps=[],
            randomness=0,
        )
        env = EnvironmentGame24()

        agents = AgentDictGOT(
            step=AgentBfsGame24,
            aggregate=AgentAggregateGame24,
            evaluate=AgentEvaluateGame24,
            step_params=self.params,
            aggregate_params=self.params,
            eval_params=self.params,
        )
        method = AlgorithmGOT(
            model=offline_model,
            agents=agents,
            env=env,
            num_selections=3,
            num_steps=4,
            num_best=2,
            num_evaluations=1,
        )

        result = await method.solve(
            idx=0,
            state=state,
            namespace="test_small",
            value_cache=None,
        )

        finished, _ = env.evaluate(result[0])

        assert finished


class TestGoTOnline:
    TEST_TIMEOUT = 0.1

    llm = "llama-3.3-70b-versatile"
    params = ModelRequestOptions(
        temperature=0.7,
        max_completion_tokens=100,
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

    @pytest.mark.asyncio(loop_scope="function")
    async def test_got_game24_with_cache(self, cache, online_model) -> None:
        state = StateGame24(
            puzzle="10 10 1 4",
            current_state="10 10 1 4",
            steps=[],
            randomness=0,
        )
        env = EnvironmentGame24()

        async with OnlineAPI(
            model=online_model,
            cache=cache,
            batch_size=1,
            timeout=self.TEST_TIMEOUT,
        ) as pipeline:
            api = API(
                pipeline=pipeline,
                model=self.llm,
            )

            agents = AgentDictGOT(
                step=AgentBfsGame24,
                aggregate=AgentAggregateGame24,
                evaluate=AgentEvaluateGame24,
                step_params=self.params,
                aggregate_params=self.params,
                eval_params=self.params,
            )
            method = AlgorithmGOT(
                model=api,
                agents=agents,
                env=env,
                num_selections=3,
                num_steps=4,
                num_best=2,
                num_evaluations=1,
            )

            result = await method.solve(
                idx=0, state=state, namespace="test_small", value_cache={}
            )

            finished, _ = env.evaluate(result[0])
            assert finished
