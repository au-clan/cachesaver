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

    # Model
    model = OnlineLLM(provider=args.provider)

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
        model=api,
        env=environment,
        config=config)
    
    print(type(method.model))

    # Benchmark
    benchmark = BenchmarkFactory.get(args.benchmark, split=args.split)

    # Run the method
    results = await method.benchmark(
        benchmark=benchmark,
        ns_ratio=args.ns_ratio,
        cache=args.value_cache
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Solve tasks with different methods.")
    #parser.add_argument("--provider", type=str, help="LLM provider")
    #parser.add_argument("--base_url", type=str, help="Base URL for the API")
    parser.add_argument("--benchmark", type=str)
    parser.add_argument("--method", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max_completion_tokens", type=int)
    parser.add_argument("--top_p", type=float)
    parser.add_argument("--stop", type=str, nargs="+")
    parser.add_argument("--logprobs", action="store_true")
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--value_cache", action="store_true")
    parser.add_argument("--correctness", type=int)
    parser.add_argument("--allow_batch_overflow", type=int)
    parser.add_argument("--ns_ratio", type=float)
    parser.add_argument("--provider", type=str)

    args = parser.parse_args()

    if args.ns_ratio > 1.0 or args.ns_ratio < 0.0:
        raise ValueError("ns_ratio must be between 0.0 and 1.0")

    asyncio.run(run(args, trial=0, cache_path="caches/developping"))