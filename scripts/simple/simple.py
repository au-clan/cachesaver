import os
import asyncio
import argparse

from diskcache import Cache
from omegaconf import OmegaConf

from openai import AsyncOpenAI

from cachesaver.pipelines import OnlineAPI

import sys
sys.path.append(os.getcwd())

from src import BenchmarkFactory, EnvironmentFactory, MethodFactory
from src.tasks import *
from src.methods import *
from src.models import OnlineLLM, API
from src.typedefs import DecodingParameters


async def run(args, trial, cache_path):
    
    # Cache directory
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache = Cache(cache_path)

    # LLM Provider
    client = AsyncOpenAI()

    # Model
    model = OnlineLLM(client=client)

    # Pipeline
    pipeline = OnlineAPI(
        model=model,
        cache=cache,
        batch_size=args.batch_size,
        timeout=args.timeout,
        allow_batch_overflow=args.allow_batch_overflow,
        correctness=bool(args.correctness)
    )

    # API
    api = API(
        pipeline=pipeline,
        model=args.model
    )

    # Decoding Parameters
    params = DecodingParameters(
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
        top_p=args.top_p,
        stop=args.stop,
        logprobs=args.logprobs
    )

    # Config for the framework hyperparameters
    config = OmegaConf.load(f"scripts/configs/{args.benchmark}.yaml")[args.method]

    # Environment
    environment = EnvironmentFactory.get(args.benchmark)

    # Method
    method = MethodFactory.get(
        method=args.method,
        benchmark=args.benchmark,
        params=params,
        model=model,
        env=environment,
        config=config)

    # Benchmark
    benchmark = BenchmarkFactory.get(args.benchmark)

    # # Run the method
    # results = await method.benchmark(
    #     benchmark=benchmark,
    #     share_ns=args.share_ns,
    #     cache=args.value_cache
    # )

    print("All set up!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Solve tasks with different methods.")
    #parser.add_argument("--provider", type=str, help="LLM provider")
    #parser.add_argument("--base_url", type=str, help="Base URL for the API")
    parser.add_argument("--benchmark", type=str, help="Benchmark to be solved")
    parser.add_argument("--method", type=str, help="Method to be used")
    parser.add_argument("--model", type=str, help="LLM model")
    parser.add_argument("--batch_size", type=int, help="CacheSaver's batch size")
    parser.add_argument("--timeout", type=float, help="CacheSaver's timeout")
    parser.add_argument("--temperature", type=float, help="Temperature for the model")
    parser.add_argument("--max_completion_tokens", type=int, help="Max completion tokens")
    parser.add_argument("--top_p", type=float, help="Top P for the model")
    parser.add_argument("--stop", type=str, nargs="+", help="Stop sequence for the model")
    parser.add_argument("--logprobs", action="store_true", help="Logprobs for the model")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset")
    parser.add_argument("--split", type=str, help="Split of the dataset")
    parser.add_argument("--value_cache", action="store_true", help="Use value cache")
    parser.add_argument("--correctness", type=int, help="Use original ('correct') implementation")
    parser.add_argument("--allow_batch_overflow", type=int, help="Allow batch overflow in CacheSaver")
    args = parser.parse_args()

    asyncio.run(run(args, trial=0, cache_path="caches/developping"))