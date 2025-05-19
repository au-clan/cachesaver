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
from src.models import OnlineLLM, API, GroqAPILLM
from src.typedefs import DecodingParameters
from src.tasks.game24 import *

cache = Cache(f"caches/game24")

async def run(args):
    # LLM Provider
    if args.provider == "openai":
        if args.base_url and "localhost" in args.base_url:
            # For local vLLM servers, use a dummy API key
            client = AsyncOpenAI(base_url=args.base_url, api_key="dummy-key")
        else:
            client = AsyncOpenAI(base_url=args.base_url) if args.base_url else AsyncOpenAI()
    elif args.provider == "together":
        client = AsyncTogether()
    elif args.provider == "local":
        raise NotImplementedError("Local client is not implemented yet.")
    elif args.provider == "groq":
        pass  # skip this check as groq model initializes its own client
    else:
        raise ValueError("Invalid provider. Choose 'openai', 'together', or 'local'.")
    # CacheSaver model layer
    if args.provider in ["openai", "together"]:
        model = OnlineLLM(client=client)
    elif args.provider == "groq":
        model = GroqAPILLM(use_multiple_keys=True)
    else:
        raise NotImplementedError("Local model is not implemented yet.")

    # CacheSaver Pipeline: Batcher -> Reorderer -> Deduplicator -> Cache -> Model
    pipeline = OnlineAPI(
                    model=model,
                    cache=cache,
                    batch_size=args.batch_size,
                    timeout=args.timeout
                    )

    # Cachesaver additional layer for wrapping: API -> Pipeline
    api = API(
        pipeline=pipeline,
        model=args.model
    )

    # Decoding parameters
    params = DecodingParameters(
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
        top_p=args.top_p,
        stop=args.stop,
        logprobs=args.logprobs,
        self_eval = args.self_eval
    )

    # Config
    config = OmegaConf.load(args.conf_path)

    # Setup the method
    if args.method == "foa":
        agents = AgentDictFOA(
            step=AgentActGame24,
            evaluate=AgentEvaluateGame24,
            step_params=params,
            eval_params=params,
        )
        method = AlgorithmFOA(
            model=api,
            agents=agents,
            env=EnvironmentGame24,
            num_agents=config.foa.num_agents,
            num_steps=config.foa.num_steps,
            k=config.foa.k,
            backtrack=config.foa.backtrack,
            resampling=config.foa.resampling,
            origin=config.foa.origin,
            min_steps=config.foa.min_steps,
            num_evaluations=config.foa.num_evaluations,
        )
    elif args.method == "tot":
        agents = AgentDictTOT(
            step=AgentBfsGame24,
            evaluate=AgentEvaluateGame24,
            step_params=params,
            eval_params=params,
        )
        method = AlgorithmTOT(
            model=api,
            agents=agents,
            env=EnvironmentGame24,
            num_selections=config.tot.num_selections,
            num_steps=config.tot.num_steps,
            num_evaluations=config.tot.num_evaluations,
        )
    elif args.method == "got":
        agents = AgentDictGOT(
            step=AgentActGame24,
            aggregate=AgentAggregateGame24,
            evaluate=AgentEvaluateGame24,
            step_params=params,
            aggregate_params=params,
            eval_params=params,
        )
        method = AlgorithmGOT(
            model=api,
            agents=agents,
            env=EnvironmentGame24,
            num_selections=config.got.num_selections,
            num_steps=config.got.num_steps,
            num_generate=config.got.num_generate,
            num_best=config.got.num_best,
            num_evaluations=config.got.num_evaluations,
        )
    elif args.method == "rap":
        agents = AgentDictRAP(
            step=AgentReactGame24,
            evaluate=AgentSelfEvaluateGame24,
            step_params=params,
            eval_params=params,
        )
        method = AlgorithmRAP(
            model=api,
            agents=agents,
            env=EnvironmentGame24,
            num_iterations=config.rap.num_iterations,
            num_samples=config.rap.num_samples,
            num_evaluations=config.rap.num_evaluations,
            exploration_constant=config.rap.exploration_constant,
        )
    elif args.method == "rap_er":
        step_params = DecodingParameters(
            temperature=args.temperature,
            max_completion_tokens=args.max_completion_tokens,
            top_p=args.top_p,
            stop=args.stop,
            logprobs=False,
            self_eval=args.self_eval
        )

        eval_params = DecodingParameters(
            temperature=args.temperature,
            max_completion_tokens=args.max_completion_tokens,
            top_p=args.top_p,
            stop=args.stop,
            logprobs=args.logprobs,
            self_eval=args.self_eval
        )
        agents = AgentDictRAP(
            step=AgentBfsGame24,
            evaluate=AgentEvaluateGame24,
            step_params=step_params,
            eval_params=eval_params,
        )
        method = AlgorithmRAP(
            model=api,
            agents=agents,
            env=EnvironmentGame24,
            num_evaluations=config.rap.num_evaluations,
            logprobs_model=api if (args.provider=="together" and args.logprobs) else None
        )
    else:
        raise NotImplementedError(f"Method {args.method} is not implemented yet.")
    
    benchmark = BenchmarkGame24(path=args.dataset_path, split=args.split)
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
        evaluations = sorted([EnvironmentGame24.evaluate(state) for state in result], key=lambda x: x[1])
        if evaluations: # check is needed as a result can have 0 evaluations
            finished.append(evaluations[-1][0])
            correct.append(evaluations[-1][1])
    acc_finished = sum(finished) / len(finished)
    acc_correct = sum(correct) / len(correct)
    if args.provider == "groq": # GroqAPI is free so no costs
        costs = {key: {"in": 0, "out": 0, "total": 0} for key in api.tokens.keys()}
    else:
        costs = {key:tokens2cost(api.tokens[key], args.model) for key in api.tokens.keys()}

    print(f"Method: {args.method}")
    print(f"Finished: {acc_finished}")
    print(f"Correct: {acc_correct}")
    for key, value in costs.items():
        print(f"\t{key}: {value['total']:.3f}$")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve Game 24 using LLMs.")
    parser.add_argument("--provider", type=str, help="LLM Provider", choices=["openai", "together", "local", "groq"], default="openai")
    parser.add_argument("--use_single_key", type=bool, help="Allows the usage of single key instead of multiple in groq")
    parser.add_argument("--model", type=str, help="LLM Model",  default="gpt-4o-mini")
    parser.add_argument("--base_url", type=str, help="Base URL for the API", default=None)
    parser.add_argument("--batch_size", type=int, help="CacheSaver's batch size", default=300)
    parser.add_argument("--timeout", type=float, help="CacheSaver's timeout", default=0.05)
    parser.add_argument("--temperature", type=float, help="Temperature for the model", default=1.0)
    parser.add_argument("--max_completion_tokens", type=int, help="Max completion tokens", default=100)
    parser.add_argument("--top_p", type=float, help="Top P for the model", default=1.0)
    parser.add_argument("--stop", type=str, nargs="+", help="Stop sequence for the model", default=None)
    parser.add_argument("--logprobs", action="store_true", help="Logprobs for the model")
    parser.add_argument("--self_eval", action="store_true", help="Self evaluation for the model", default=True)
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset")
    parser.add_argument("--split", type=str, help="Split of the dataset", choices=["mini", "train", "validation", "test"], default="mini")
    parser.add_argument("--share_ns", action="store_true", help="Share namespace between puzzles")
    parser.add_argument("--method", type=str, help="Method to use", choices=["foa", "tot", "got", "rap", "rap_er"], default="foa")
    parser.add_argument("--conf_path", type=str, help="Path to corresponding config")
    parser.add_argument("--value_cache", action="store_true", help="Use value cache")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, filename=f"logs/game24/{args.method}.log", filemode="w")

    asyncio.run(run(args))
