# imports
import sys
import asyncio
import pytest
import tempfile
from diskcache import Cache
from groq import AsyncGroq
from groq import RateLimitError
from lazykey import AsyncKeyHandler
from omegaconf import OmegaConf
from cachesaver.pipelines import OnlineAPI
from cachesaver.typedefs import Response, Batch, Request
from typing import Any, List
import secret

from src.algorithms import AgentDictGOT, AlgorithmGOT
from src.models import OnlineLLM, API
from src.typedefs import DecodingParameters, Model
from src.tasks.game24 import (
    EnvironmentGame24,
    AgentBfsGame24,
    AgentAggregateGame24,
    AgentEvaluateGame24,
    StateGame24,
)

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class MockLLM(Model):
    def __init__(self, model: str) -> None:
        self.client = AsyncKeyHandler(secret.GROQ_API_KEYS, AsyncGroq)
        self.model = model

    async def request(
        self,
        prompt: str,
        n: int,
        request_id: int,
        namespace: str,
        params: DecodingParameters,
    ):
        sleep = 1
        while True:
            try:
                completion = await self.client.request(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model,
                    n=1,
                    max_tokens=params.max_completion_tokens
                    or None,  # or None not needed but just to be explicit
                    temperature=params.temperature or 1,
                    stop=params.stop or None,
                    top_p=params.top_p or 1,
                    seed=1234,
                    logprobs=params.logprobs or False,
                    top_logprobs=None,
                )
                break
            except RateLimitError as e:
                await asyncio.sleep(max(sleep, 90))
                sleep *= 2
            except Exception as e:
                print(f"Error {e}")
                raise e
        input_tokens = completion.usage.prompt_tokens
        completion_tokens = completion.usage.completion_tokens
        response = [choice.message.content for choice in completion.choices]
        return response

    async def batch_request(self, batch: Batch) -> List[Response]:
        requests = [self.request(request) for request in batch.requests]
        completions = await asyncio.gather(*requests)
        return completions


class LazyMockOnlineLLM(Model):
    def __init__(self) -> None:
        self.lazy_client = AsyncKeyHandler(secret.GROQ_API_KEYS, AsyncGroq)

    async def request(self, request: Request) -> Response:
        sleep = 1
        while True:
            try:
                completion = await self.lazy_client.request(
                    messages=[{"role": "user", "content": request.prompt}],
                    model=request.model,
                    n=1,
                    max_tokens=request.max_completion_tokens or None,
                    temperature=request.temperature or 1,
                    stop=request.stop or None,
                    top_p=request.top_p or 1,
                    seed=1234,
                    logprobs=request.logprobs or False,
                    top_logprobs=request.top_logprobs or None,
                )
                break
            except RateLimitError as e:
                await asyncio.sleep(max(sleep, 90))
                sleep *= 2
            except Exception as e:
                print(f"Error {e}")
                raise e
        input_tokens = completion.usage.prompt_tokens
        completion_tokens = completion.usage.completion_tokens
        response = Response(
            data=[
                (choice.message.content, input_tokens, completion_tokens / 1)
                for choice in completion.choices
            ]
        )
        return response

    async def batch_request(self, batch: Batch) -> List[Response]:
        requests = [self.request(request) for request in batch.requests]
        completions = await asyncio.gather(*requests)
        return completions


class TestGoTOffline:
    model = MockLLM(model="llama-3.3-70b-versatile")
    env = EnvironmentGame24()
    params = DecodingParameters(
        temperature=0.7,
        max_completion_tokens=100,
        top_p=1.0,
        stop=None,
        logprobs=False,
    )

    @pytest.mark.asyncio()
    async def test_aggregate(self) -> None:
        state = StateGame24(
            puzzle="10 10 1 4",
            current_state="10 10 1 4",
            steps=[],
            randomness=0,
        )
        generate = AgentBfsGame24()
        aggregate = AgentAggregateGame24()

        generate_results = await generate.act(
            model=self.model,
            state=state,
            namespace="test_small",
            request_id=0,
            params=self.params,
        )
        aggregate_results = await aggregate.act(
            model=self.model,
            state=state,
            actions=generate_results,
            k=3,
            namespace="test_small",
            request_id=0,
            params=self.params,
        )
        assert len(aggregate_results) == 3

        aggregate_results = await aggregate.act(
            model=self.model,
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
    async def test_got(self) -> None:
        state = StateGame24(
            puzzle="10 10 1 4",
            current_state="10 10 1 4",
            steps=[],
            randomness=0,
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
            model=self.model,
            agents=agents,
            env=EnvironmentGame24,
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

        finished, _ = self.env.evaluate(result[0])

        assert finished


class TestGoTOnline:
    TEST_TIMEOUT = 0.1

    llm = "llama-3.3-70b-versatile"
    model = LazyMockOnlineLLM()
    env = EnvironmentGame24()
    params = DecodingParameters(
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
    async def test_got_with_cache(self, cache):
        state = StateGame24(
            puzzle="10 10 1 4",
            current_state="10 10 1 4",
            steps=[],
            randomness=0,
        )

        async with OnlineAPI(
            model=self.model,
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
                env=EnvironmentGame24,
                num_selections=3,
                num_steps=4,
                num_best=2,
                num_evaluations=1,
            )

            result = await method.solve(
                idx=0, state=state, namespace="test_small", value_cache={}
            )

            finished, _ = self.env.evaluate(result[0])
            assert finished
