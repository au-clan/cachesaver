import os
import asyncio
import logging
import argparse
from diskcache import Cache
from openai import AsyncOpenAI
from omegaconf import OmegaConf
from together import AsyncTogether
from cachesaver.pipelines import OnlineAPI
logger = logging.getLogger(__name__)

import sys
sys.path.append(os.getcwd())

from src.utils import tokens2cost
from src.algorithms import *
from src.models import OnlineLLM, API
from src.typedefs import DecodingParameters
from src.tasks.humaneval import EnvironmentHumanEval, BenchmarkHumanEval, AgentGenerateHumanEval, AgentAggregateHumanEval, AgentEvaluateHumanEval

cache = Cache(f"caches/humaneval")

async def run(args):
    if args.provider == "openai":
        client = AsyncOpenAI()
    elif args.provider == "together":
        client = AsyncTogether()
    elif args.provider == "local":
        raise NotImplementedError("Local client is not implement yet.")
    else:
        raise ValueError("Invalid provider. Choose 'openai', 'together', or 'local'")
    
    if args.provider in ["openai", "together"]:
        model = OnlineLLM(client=client)
    else:
        raise NotImplementedError("Local model is not implemented yet.")
    
    pipeline = OnlineAPI(
        model=model,
        cache=cache,
        batch_size=args.batch_size,
        timeout=args.timeout
    )

    api = API(
        pipeline=pipeline,
        model=args.model
    )

    params = DecodingParameters(
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
        top_p=args.top_p,
        stop=args.stop,
        logprobs=args.logprobs
    )
    
    config = OmegaConf.load(args.conf_path)

    if args.method == "got":
        agents = AgentDictGOT(
            step=AgentGenerateHumanEval,
            aggregate=AgentAggregateHumanEval,
            evaluate=AgentEvaluateHumanEval,
            step_params=params,
            aggregate_params=params,
            eval_params=params,
        )
        method = AlgorithmGOT(
            model=api,
            agents=agents,
            env=EnvironmentHumanEval,
            num_selections=config.got.num_selections,
            num_steps=config.got.num_steps,
            num_generate=config.got.num_generate,
            num_best=config.got.num_best,
            num_evaluations=config.got.num_evaluations,
        )
    else:
        raise NotImplementedError(f"Method {args.method} is not implemented yet.")
    
    benchmark = BenchmarkHumanEval(path=args.dataset_path, split=args.split)
    results = await method.benchmark(
        benchmark=benchmark,
        share_ns=args.share_ns,
        cache=args.value_cache,
    )
    finished = []
    correct = []
    for i, result in enumerate(results):
        logger.info(f"Result {i}:")
        for r in result:
            logger.info(f"\t{r}")
    for result in results:
        evaluations = sorted([EnvironmentHumanEval.evaluate(state) for state in result], key=lambda x: x[1])
        finished.append(evaluations[-1][0])
        correct.append(evaluations[-1][1])
    acc_finished = sum(finished) / len(finished)
    acc_correct = sum(correct) / len(correct)
    print(f"Method: {args.method}")
    print(f"Finished: {acc_finished}")
    print(f"Correct: {acc_correct}")

    costs = {key:tokens2cost(api.tokens[key], args.model) for key in api.tokens.keys()}
    for key, value in costs.items():
        print(f"\t{key}: {value['total']:.3f}$")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve humaneval using LLMs.")
    parser.add_argument("--provider", type=str, help="LLM provider", choices=["openai", "together", "local"], default="openai")
    parser.add_argument("--model", type=str, help="LLM model", default="gpt-4o-mini")
    parser.add_argument("--batch_size", type=int, help="CacheSaver's batch size", default=300)
    parser.add_argument("--timeout", type=float, help="CacheSaver's timeout", default=0.05)
    parser.add_argument("--temperature", type=float, help="Temperature for the model", default=1.0)
    parser.add_argument("--max_completion_tokens", type=int, help="Max completion tokens", default=100)
    parser.add_argument("--top_p", type=float, help="Top P for the model", default=1.0)
    parser.add_argument("--stop", type=str, nargs="+", help="Stop sequence for the model", default=None)
    parser.add_argument("--logprobs", action="store_true", help="Logprobs for the model")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset")
    parser.add_argument("--split", type=str, help="Split of the dataset", choices=["mini", "train", "validation", "test"], default="mini")
    parser.add_argument("--share_ns", action="store_true", help="Share namespace between puzzles")
    parser.add_argument("--method", type=str, help="Method to use", choices=["foa", "tot", "got"], default="foa")
    parser.add_argument("--conf_path", type=str, help="Path to corresponding config")
    parser.add_argument("--value_cache", action="store_true", help="Use value cache")
    args = parser.parse_args()

    if not os.path.exists("logs/humaneval"):
        os.makedirs("logs/humaneval")
    
    logging.basicConfig(level=logging.INFO, filename=f"logs/humaneval/{args.method}.log", filemode="w")

    asyncio.run(run(args))