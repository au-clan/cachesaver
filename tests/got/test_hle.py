from os import name
import sys
import asyncio
import pytest
import tempfile
from diskcache import Cache
from cachesaver.pipelines import OnlineAPI

from src.algorithms import AgentDictGOT, AlgorithmGOT
from src.models import API
from src.typedefs import Algorithm, DecodingParameters
from src.tasks.hle import (
    EnvironmentHLE,
    AgentActHLE,
    AgentAggregateHLE,
    AgentEvaluateHLE,
    StateHLE,
    BenchmarkHLE
)

class TestGoTHLE:
    params = DecodingParameters(
        temperature=0.7,
        max_completion_tokens=4000,
        top_p=1.0,
        stop=None,
        logprobs=False,
    )
    env = EnvironmentHLE()
    benchmark = BenchmarkHLE("datasets/dataset_hle_sample_without_images.jsonl.gz", "mini")

    @pytest.mark.asyncio()
    async def test_aggregate_hle(self, offline_model_openai) -> None:
        _, state = self.benchmark[0]

        act = AgentActHLE()
        actions = await act.act(
            offline_model_openai,
            state,
            n=5,
            namespace="test_small",
            request_id=0,
            params=self.params,
        )
        assert len(actions) == 5
        
        aggregate = AgentAggregateHLE()
        selected_actions = await aggregate.act(
            offline_model_openai,
            state,
            actions,
            k=3,
            namespace="test_small",
            request_id=0,
            params=self.params,
        )
        
        assert len(selected_actions) == 3
        assert all(action in actions for action in selected_actions)

    @pytest.mark.asyncio()
    async def test_got_hle(self, offline_model_openai) -> None:
        states = [(i, state) for i, state in self.benchmark][:5]

        agents = AgentDictGOT(
            step=AgentActHLE(),
            aggregate=AgentAggregateHLE(),
            evaluate=AgentEvaluateHLE(),
            step_params=self.params,
            aggregate_params=self.params,
            eval_params=self.params,
        )
        algorithm = AlgorithmGOT(
            model=offline_model_openai,
            agents=agents,
            env=self.env,
            num_selections=3,
            num_steps=5,
            num_generate=5,
            num_best=2,
            num_evaluations=3,
        )

        results_coroutine = [
            algorithm.solve(
                idx=i,
                state=state,
                namespace="test_small",
                value_cache=None,
            )
            for i, state in states 
        ]
        results = await asyncio.gather(*results_coroutine)

        finished, corrects = [], []
        for result in results:
            for r in result:
                finish, correct = self.env.evaluate(r)
                finished.append(finish)
                corrects.append(correct)

        assert any(finished)
        assert sum(corrects) > 0

    @pytest.mark.asyncio(loop_scope="function")
    async def test_got_hle_with_cachesaver(self, cache, online_model_openai) -> None:
        states = [(i, state) for i, state in self.benchmark][:5]

        agents = AgentDictGOT(
            step=AgentActHLE(),
            aggregate=AgentAggregateHLE(),
            evaluate=AgentEvaluateHLE(),
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
                model="gpt-4.1-nano",
            )

            algorithm = AlgorithmGOT(
                model=api,
                agents=agents,
                env=self.env,
                num_selections=3,
                num_steps=5,
                num_generate=5,
                num_best=2,
                num_evaluations=3,
            )

            results_coroutine = [
                algorithm.solve(
                    idx=i,
                    state=state,
                    namespace="test_small",
                    value_cache=None,
                )
                for i, state in states
            ]

            results = await asyncio.gather(*results_coroutine)

            finished, corrects = [], []
            for result in results:
                for r in result:
                    finish, correct = self.env.evaluate(r)
                    finished.append(finish)
                    corrects.append(correct)
            
            assert any(finished)
            assert sum(corrects) > 0