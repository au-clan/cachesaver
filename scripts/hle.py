import os
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
import pdb

from src.utils import tokens2cost
from src.algorithms import *
from src.models import OnlineLLM, API
from src.typedefs import DecodingParameters
from src.tasks.hle.environment import EnvironmentHLE
from src.tasks.hle.benchmark import BenchmarkHLE
from src.tasks.hle.agents import AgentActHLE, AgentAggregateHLE, AgentEvaluateHLE, AgentBfsHLE

cache = Cache(f"caches/hle")

async def run(args):
    
    # LLM Provider
    if args.provider == "openai":
        client = AsyncOpenAI()
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
        logprobs=args.logprobs
    )

    # Config
    config = OmegaConf.load(args.conf_path)

    # Setup the method
    ## We can create a method factory for this
    if args.method == "foa":
        agents = AgentDictFOA(
            step=AgentActHLE,
            evaluate=AgentEvaluateHLE,
            step_params=params,
            eval_params=params,
        )
        method = AlgorithmFOA(
            model=api,
            agents = agents,
            env=EnvironmentHLE,
            num_agents=config.foa.num_agents,
            num_steps=config.foa.num_steps,
            k=config.foa.k,
            backtrack=config.foa.backtrack,
            resampling=config.foa.resampling,
            origin= config.foa.origin,
            min_steps=config.foa.min_steps,
            num_evaluations=config.foa.num_evaluations,
        )
    elif args.method == "tot":
        agents = AgentDictTOT(
            step=AgentBfsHLE,
            evaluate=AgentEvaluateHLE,
            step_params=params,
            eval_params=params,
        )
        method = AlgorithmTOT(
            model=api,
            agents = agents,
            env=EnvironmentHLE,
            num_selections=config.tot.num_selections,
            num_steps=config.tot.num_steps,
            num_evaluations=config.tot.num_evaluations,
        )
    elif args.method == "got":
        agents = AgentDictGOT(
            step=AgentActHLE,
            aggregate=AgentAggregateHLE,
            evaluate=AgentEvaluateHLE,
            step_params=params,
            aggregate_params=params,
            eval_params=params,
        )
        method = AlgorithmGOT(
            model=api,
            agents=agents,
            env=EnvironmentHLE,
            num_selections=config.got.num_selections,
            num_steps=config.got.num_steps,
            num_generate=config.got.num_generate,
            num_best=config.got.num_best,
            num_evaluations=config.got.num_evaluations,
        )
    else:
        raise NotImplementedError("Method not implemented yet.")
    
    benchmark = BenchmarkHLE(path=args.dataset_path, split=args.split)

    logger.debug(f"Benchmark details:")
    logger.debug(f"- Path: {args.dataset_path}")
    logger.debug(f"- Split: {args.split}")
    logger.debug(f"- Size: {len(benchmark)}")
    logger.debug(f"- First state: {benchmark[0] if len(benchmark) > 0 else 'Empty'}")

    logger.debug(f"Starting benchmark execution:")
    logger.debug(f"- Method: {args.method}")
    logger.debug(f"- Config: {config[args.method]}")
    logger.debug(f"- Share NS: {args.share_ns}")
    logger.debug(f"- Value Cache: {args.value_cache}")

    pdb.set_trace()
    results = await method.benchmark(
        benchmark=benchmark,
        share_ns=args.share_ns,
        cache=args.value_cache,
    )
    logger.info(f"Benchmark returned {len(results)} results")
    
    for i, result in enumerate(results):
        logger.debug(f"Processing result {i}/{len(results)}:")
        logger.debug(f"- States in result: {len(result)}")
        for j, r in enumerate(result):
            logger.debug(f"  - State {j}: {r}")
        
        evaluations = sorted([EnvironmentHLE.evaluate(state) for state in result], key=lambda x: x[1])
        logger.debug(f"- Evaluations: {evaluations}")
        finished.append(evaluations[-1][0])
        correct.append(evaluations[-1][1])
        logger.debug(f"- Finished: {evaluations[-1][0]}")
        logger.debug(f"- Correct: {evaluations[-1][1]}")
        
    #debug
    if not results:
        logger.error("Benchmark returned empty results")
        return
    #debug
    
    finished = []
    correct = []
    for i, result in enumerate(results):
        logger.debug(f"Result {i}:")
        for r in result:
            logger.debug(f"\t{r}")
    for result in results:
        for r in result:
            print(f"\t{r}")
        evaluations = sorted([EnvironmentHLE.evaluate(state) for state in result], key=lambda x: x[1])
        finished.append(evaluations[-1][0])
        correct.append(evaluations[-1][1])
    logging.info(f"Finished list: {finished}")
    if len(finished) == 0:
        logging.error("No tasks were finished. Exiting.")
        return
    
    pdb.set_trace()
    acc_finished = sum(finished) / len(finished)
    acc_correct = sum(correct) / len(correct)
    
    costs = {key:tokens2cost(api.tokens[key], args.model) for key in api.tokens.keys()}

    print("DEBUG: Final Results:")
    print(f"DEBUG: finished={finished}")
    print(f"DEBUG: correct={correct}")
    print(f"DEBUG: costs={costs}")
    print(f"Method: {args.method}")
    print(f"Finished: {acc_finished:.3f}%")
    print(f"Correct: {acc_correct:.3f}%")
    for key, value in costs.items():
        print(f"\t{key}: {value['total']:.3f}$")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve HLE using LLMs.")
    parser.add_argument("--provider", type=str, help="LLM Provider", choices=["openai", "together", "local"], default="openai")
    parser.add_argument("--model", type=str, help="LLM Model",  default="gpt-4o-mini")
    parser.add_argument("--batch_size", type=int, help="CacheSaver's batch size", default=300)
    parser.add_argument("--timeout", type=float, help="CacheSaver's timeout", default=0.05)
    parser.add_argument("--temperature", type=float, help="Temperature for the model", default=1.0)
    parser.add_argument("--max_completion_tokens", type=int, help="Max completion tokens", default=100)
    parser.add_argument("--top_p", type=float, help="Top P for the model", default=1.0)
    parser.add_argument("--stop", type=str, nargs="+", help="Stop sequence for the model", default=None)
    parser.add_argument("--logprobs", action="store_true", help="Logprobs for the model")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset", default="datasets/dataset_hle_sample_without_images.jsonl.gz")
    parser.add_argument("--split", type=str, help="Split of the dataset", choices=["mini", "train", "validation", "test"], default="mini")
    parser.add_argument("--share_ns", action="store_true", help="Share namespace between puzzles")
    parser.add_argument("--method", type=str, help="Method to use", choices=["foa", "tot", "got"], default="foa")
    parser.add_argument("--conf_path", type=str, help="Path to corresponding config", default="scripts/hle.yaml")
    parser.add_argument("--value_cache", action="store_true", help="Use value cache")
    args = parser.parse_args()


    print(f"Dataset path: {args.dataset_path}")
    print(args)
    file_path = f"logs/hle/{args.method}.log"
    # Add after dataset loading
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # logging.basicConfig(level=logging.INFO, filename=file_path, filemode="w")
    
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(file_path, mode='w'),
            logging.StreamHandler()  # Also print to console
        ]
    )
    logger.debug("Starting script execution")
    asyncio.run(run(args))

