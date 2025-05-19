import os
import re
import time
import shutil
import asyncio
import itertools
import logging
import argparse
import numpy as np
from diskcache import Cache
from openai import AsyncOpenAI
from omegaconf import OmegaConf
from together import AsyncTogether
from cachesaver.pipelines import OnlineAPI
logger = logging.getLogger(__name__)

import sys
sys.path.append(os.getcwd())

from src.utils import tokens2cost, clean_log
from src.algorithms import *
from src.models import OnlineLLM, API
from src.typedefs import DecodingParameters
from src.tasks.game24 import *

def build_method(method_name: str,params: DecodingParameters, api: API, config: OmegaConf):
# Setup the method
    if method_name == "foa":
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
    elif method_name == "tot_bfs":
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
            num_selections=config.tot_bfs.num_selections,
            num_steps=config.tot_bfs.num_steps,
            num_evaluations=config.tot_bfs.num_evaluations,
        )
    elif method_name == "got":
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
    elif method_name == "rap":
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
    elif method_name == "react":
        agents = AgentDictReact(
            step=AgentReactGame24,
            step_params=params,
        )
        method = AlgorithmReact(
            model=api,
            agents=agents,
            env=EnvironmentGame24,
            num_steps=config.react.num_steps,
        )

    else:
        raise NotImplementedError(f"Method {method_name} is not implemented yet.")
    return method

async def run(args, ablation, trial, cache_path, method_name):
    # Cache to be used
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache = Cache(cache_path)

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
    else:
        raise ValueError("Invalid provider. Choose 'openai', 'together', or 'local'.")
    
    # CacheSaver model layer
    if args.provider in ["openai", "together"]:
        model = OnlineLLM(client=client)
    else:
        raise NotImplementedError("Local model is not implemented yet.")
    
    # CacheSaver Pipeline: Batcher -> Reorderer -> Deduplicator -> Cache -> Model
    pipeline = OnlineAPI(
                    model=model,
                    cache=cache,
                    batch_size=args.batch_size,
                    timeout=args.timeout,
                    allow_batch_overflow=True,
                    correctness=bool(args.correctness)
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
        logprobs=args.logprobs
    )

    # Config for framework hyperpaarameters
    config = OmegaConf.load(args.conf_path)
    if ablation == "resampling":
        config.foa.resampling = "max_unique"
    elif ablation == "backtracking":
        config.foa.backtrack = 0
    elif ablation == "selection":
        config.foa.k = 0

    # Build the method
    method = build_method(method_name, params, api, config)

    # Load the dataset
    benchmark = BenchmarkGame24(path=args.dataset_path, split=args.split)

    # Run the method
    start = time.time()
    results = await method.benchmark(
        benchmark=benchmark,
        share_ns=True,
        cache=args.value_cache,
    )
    end = time.time()

    finished = []
    correct = []
    for result in results:
        evaluations = sorted([EnvironmentGame24.evaluate(state) for state in result], key=lambda x: x[1])
        finished.append(False if len(evaluations) == 0 else evaluations[-1][0])
        correct.append(1.0 if len(evaluations) == 0 else evaluations[-1][1])
    perc_finished = sum(finished) / len(finished)
    perc_correct = sum(correct) / len(correct)
    costs = {key:tokens2cost(api.tokens[key], args.model)["total"] for key in api.tokens.keys()}
    latency = {
        "mean": np.mean(api.latencies), 
        "std": np.std(api.latencies),
        "max": np.max(api.latencies), 
        "min": np.min(api.latencies), 
        "total": np.sum(api.latencies)
        }
    reuse = {
        "mean": np.mean(list(api.reuse.values())),
        "std": np.std(list(api.reuse.values())),
        "max": np.max(list(api.reuse.values())),
        "min": np.min(list(api.reuse.values())),
        "median": np.median(list(api.reuse.values())),
        "total" : np.sum(list(api.reuse.values())),
        "num_unique": len(api.reuse),
    }
    run_time = end - start
    throughput = len(benchmark) / run_time

    logger.info(f"Finished: {perc_finished:.2f} (trial {trial})")
    logger.info(f"Correct: {perc_correct:.2f} (trial {trial})")
    logger.info(f"Costs: {costs} (trial {trial})")
    logger.info(f"Latency: {latency['mean']} (trial {trial})")
    logger.info(f"Run time: {run_time:.2f} seconds (trial {trial})")
    logger.info(f"Throughput: {throughput:.2f} puzzles/second (trial {trial})")

    logger.info(f"Correct (deailed): {correct} (trial {trial})")
    logger.info(f"Tokens (detailed): {api.tokens} (trial {trial})")
    logger.info(f"Calls (detailed): {api.calls} (trial {trial})")
    logger.info(f"Reuse (detailed): {reuse} (trial {trial})")
    cache.close()
    
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
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset")
    parser.add_argument("--split", type=str, help="Split of the dataset", choices=["mini", "train", "validation", "test"], default="mini")
    parser.add_argument("--share_ns", action="store_true", help="Share namespace between puzzles")
    parser.add_argument("--conf_path", type=str, help="Path to corresponding config")
    parser.add_argument("--value_cache", action="store_true", help="Use value cache")
    parser.add_argument("--correctness", type=int, help="Use original ('correct') implementation")
    args = parser.parse_args()

    filename = f"logs/perms/{args.model.split('/')[-1]}/game24.log"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename=filename, filemode="a")
    

    trial = 0
    ablations = ["resampling", "backtracking"]
    for ablation in ablations:
        
        #ablations_to_run = [a for a in ablations if a != ablation]

        cache_path = f"cache/l1o/{'_'.join(ablations)}"

        for a in ablations:
            trial += 1
            logger.info("#"*50)
            logger.info(f"Trial {trial}, ablation {a}, perm {cache_path.split('/')[-1]}")
            asyncio.run(run(args, method_name="foa", ablation=a, cache_path=cache_path, trial=trial))

        shutil.copytree(cache_path, cache_path + "_copy")

        
