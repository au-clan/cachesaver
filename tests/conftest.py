from groq import AsyncGroq
from openai import AsyncOpenAI
from groq import RateLimitError
from lazykey import AsyncKeyHandler
from cachesaver.typedefs import Response, Batch, Request
from typing import List
import secret
import asyncio
import pytest
import tempfile
from diskcache import Cache

from src.typedefs import Model, DecodingParameters


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
                    messages=[{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt,
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
                    messages=[{"role": "user", "content": request.prompt}] if isinstance(request.prompt, str) else request.prompt,
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

class MockOnlineLLMOpenAI(Model):
    def __init__(self, model: str) -> None:
        self.client = AsyncOpenAI()
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
                completion = await self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt,
                    model=self.model,
                    n=n,
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

class LazyMockOnlineLLMOpenAI(Model):
    def __init__(self) -> None:
        self.client = AsyncOpenAI()

    async def request(self, request: Request) -> Response:
        sleep = 1
        while True:
            try:
                completion = await self.client.chat.completions.create(
                    messages=[{"role": "user", "content": request.prompt}] if isinstance(request.prompt, str) else request.prompt,
                    model=request.model,
                    n=request.n or 1,
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

@pytest.fixture(scope="function")
def cache():
    """Provide a temporary cache for each test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with Cache(tmpdir) as cache:
            yield cache

@pytest.fixture
def offline_model() -> Model:
    """
    Fixtur for initializing the offline model, that does not use cachesaver API.
    """
    return MockLLM(model="llama-3.3-70b-versatile")

@pytest.fixture
def online_model() -> Model:
    """
    Fixture for initializing the online model, that uses cachesaver API.
    """
    return LazyMockOnlineLLM()

@pytest.fixture
def offline_model_openai() -> Model:
    """
    Fixture for initializing the openai model, that uses cachesaver API.
    """
    return MockOnlineLLMOpenAI('gpt-4.1-nano')

@pytest.fixture
def online_model_openai() -> Model:
    """
    Fixture for initializing the openai model, that uses cachesaver API.
    """
    return LazyMockOnlineLLMOpenAI()