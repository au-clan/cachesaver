import os
import re
import time
import asyncio
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
from src.tasks.hotpotqa import *

def build_method(method_name: str, params: DecodingParameters, api: API, config: OmegaConf):
# Setup the method
    if method_name == "foa":
        agents = AgentDictFOA(
            step=AgentActHotpotQA,
            evaluate=AgentEvaluateHotpotQA,
            step_params=params,
            eval_params=params,
        )
        method = AlgorithmFOA(
            model=api,
            agents=agents,
            env=EnvironmentHotpotQA,
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
            step=AgentBfsHotpotQA,
            evaluate=AgentEvaluateHotpotQA,
            step_params=params,
            eval_params=params,
        )
        method = AlgorithmTOT(
            model=api,
            agents=agents,
            env=EnvironmentHotpotQA,
            num_selections=config.tot_bfs.num_selections,
            num_steps=config.tot_bfs.num_steps,
            num_evaluations=config.tot_bfs.num_evaluations,
        )
    elif method_name == "got":
        agents = AgentDictGOT(
            step=AgentBfsHotpotQA,
            aggregate=AgentAggregateHotpotQA,
            evaluate=AgentEvaluateHotpotQA,
            step_params=params,
            aggregate_params=params,
            eval_params=params,
        )
        method = AlgorithmGOT(
            model=api,
            agents=agents,
            env=EnvironmentHotpotQA,
            num_selections=config.got.num_selections,
            num_steps=config.got.num_steps,
            num_best=config.got.num_best,
            num_evaluations=config.got.num_evaluations,
        )
    elif args.method == "rap":
        agents = AgentDictRAP(
            step=AgentReactHotpotQA,
            evaluate=AgentSelfEvaluateHotpotQA,
            step_params=params,
            eval_params=params,
        )
        method = AlgorithmRAP(
            model=api,
            agents=agents,
            env=EnvironmentHotpotQA,
            num_iterations=config.rap.num_iterations,
            num_samples=config.rap.num_samples,
            num_evaluations=config.rap.num_evaluations,
            exploration_constant=config.rap.exploration_constant,
        )
    elif args.method == "react":
        agents = AgentDictReact(
            step=AgentReactHotpotQA,
            step_params=params,
        )
        method = AlgorithmReact(
            model=api,
            agents=agents,
            env=EnvironmentHotpotQA,
            num_steps=config.react.num_steps,
        )

    else:
        raise NotImplementedError(f"Method {method_name} is not implemented yet.")
    return method

async def run(args, trial, cache_path):
    # Cache to be used
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache = Cache(cache_path)

    # LLM Provider
    if args.provider == "openai":
        if args.base_url and "v1" in args.base_url:
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

    # Build the method
    method = build_method(args.method, params, api, config)

    # Load the dataset
    benchmark = BenchmarkHotpotQA(path=args.dataset_path, split=args.split)

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
        evaluations = sorted([EnvironmentHotpotQA.evaluate(state) for state in result], key=lambda x: x[1])
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
    
    print("All good.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve HotpotQA using LLMs.")
    parser.add_argument("--provider", type=str, help="LLM provider")
    parser.add_argument("--base_url", type=str, help="Base URL for the API")
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
    parser.add_argument("--method", type=str, help="Method to use")
    parser.add_argument("--conf_path", type=str, help="Path to corresponding config")
    parser.add_argument("--value_cache", action="store_true", help="Use value cache")
    parser.add_argument("--correctness", type=int, help="Use original ('correct') implementation")
    args = parser.parse_args()

    filename = f"logs/correctness/hotpotqa/{args.method}.log"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename=filename, filemode="a")
    logger.info("#"*50)

    # Load previous content
    with open(filename, "r") as f:
        contents = f.read()
    
    if args.batch_size == 1:
        previous_trials = [int(num) for num in re.findall(r"Shared Namespace \(trial (\d+)\)", contents)]
        trial = max(previous_trials) + 1 if previous_trials else 1
        logger.info(f"Shared Namespace (trial {trial})")
        cache_path = f"caches/correctness/hotpotqa/{args.method}/sns_{trial}"
    else:
        previous_trials = [int(num) for num in re.findall(r"Shared Namespace and Batch \(trial (\d+)\)", contents)]
        trial = max(previous_trials) + 1 if previous_trials else 1
        logger.info(f"Shared Namespace and Batch (trial {trial})")
        cache_path = f"caches/correctness/hotpotqa/{args.method}/snsb_{trial}"

    asyncio.run(run(args, trial=trial, cache_path=cache_path))
    logger.info("\n"*3)
    clean_log(filename)
