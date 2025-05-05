import argparse
import asyncio
import logging
import os

from cachesaver.pipelines import OnlineAPI
from diskcache import Cache
from omegaconf import OmegaConf
from openai import AsyncOpenAI
from together import AsyncTogether

from src.algorithm_options.rafa import RAFAOptions
from src.algorithms.rafa_algo import AgentDictRAFA, AlgorithmRAFA
from src.models.groq_wrapper import GroqModel
from src.tasks.game24.rafa_agent import AgentRafaGame24_eval, AgentRAFA_reflect, \
    AgentRAFA_reflect_value, AgentRAFA_plan, AgentRAFA_plan_evaluate

logger = logging.getLogger(__name__)

import sys

sys.path.append(os.getcwd())

from src.utils import tokens2cost
from src.algorithms import *
from src.models import OnlineLLM, API
from src.typedefs import ModelRequestOptions
from src.tasks.game24 import EnvironmentGame24, BenchmarkGame24, AgentActGame24, AgentAggregateGame24, \
    AgentEvaluateGame24, AgentBfsGame24

cache = Cache(f"caches/game24")


async def run(args):
    # LLM Provider
    if args.provider == "openai":
        client = AsyncOpenAI()
    elif args.provider == "together":
        client = AsyncTogether()
    elif args.provider == "local":
        raise NotImplementedError("Local client is not implemented yet.")
    elif args.provider == "groq":
        pass
    else:
        raise ValueError("Invalid provider. Choose 'openai', 'together', or 'local'.")

    # CacheSaver model layer
    if args.provider in ["openai", "together"]:
        model = OnlineLLM(client=client)
    elif args.provider == "groq":
        model = GroqModel(api_key=os.getenv("GROQ_API_KEY"), model=args.model)
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
    modelRequestoptions = ModelRequestOptions(
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
        top_p=args.top_p,
        stop=args.stop,
        logprobs=args.logprobs
    )

    # Config
    config = OmegaConf.load("game24.yaml")

    # Setup the method
    if args.method == "foa":
        agents = AgentDictFOA(
            step=AgentActGame24(),
            evaluate=AgentEvaluateGame24,
            step_params=modelRequestoptions,
            eval_params=modelRequestoptions,
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
            step_params=modelRequestoptions,
            eval_params=modelRequestoptions,
        )
        method = AlgorithmTOT(
            model=api,
            agents=agents,
            env=EnvironmentGame24,
            num_selections=config.tot.num_selections,
            num_steps=config.tot.num_steps,
            num_evaluations=config.tot.num_evaluations,
        )
    elif args.method == "rafa":
        agents = AgentDictRAFA(
            agent_reflect=AgentRAFA_reflect(),
            agent_reflect_value=AgentRAFA_reflect_value(),
            agent_plan=AgentRAFA_plan(),
            agent_plan_evaluate=AgentRAFA_plan_evaluate(),
            agent_eval=AgentRafaGame24_eval(),

        )
        method = AlgorithmRAFA(
            model=api,  # todo lint complain about type... should be fixed
            agents=agents,
            env=EnvironmentGame24(),
            rafa_options=RAFAOptions(n_propose_sample=2,  # todo all of these configs shouldnt be hardcoded
                                     n_generate_sample=2,
                                     n_evaluate_sample=2,
                                     max_step=5,
                                     n_select_sample=1)

        )
    elif args.method == "got":
        agents = AgentDictGOT(
            step=AgentBfsGame24,
            aggregate=AgentAggregateGame24,
            evaluate=AgentEvaluateGame24,
            step_params=modelRequestoptions,
            aggregate_params=modelRequestoptions,
            eval_params=modelRequestoptions,
        )
        method = AlgorithmGOT(
            model=api,
            agents=agents,
            env=EnvironmentGame24,
            num_selections=config.got.num_selections,
            num_steps=config.got.num_steps,
            num_best=config.got.num_best,
            num_evaluations=config.got.num_evaluations,
        )
    else:
        raise NotImplementedError(f"Method {args.method} is not implemented yet.")

    benchmark = BenchmarkGame24(path=r"C:\Users\Oskar\PycharmProjects\AUCLAN\cachesaver\datasets\dataset_game24.csv.gz",
                                split=args.split)

    # benchmark = BenchmarkGame24(path=args.dataset_path, split=args.split)
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
        finished.append(evaluations[-1][0])
        correct.append(evaluations[-1][1])
    acc_finished = sum(finished) / len(finished)
    acc_correct = sum(correct) / len(correct)
    costs = {key: tokens2cost(api.tokens[key], args.model) for key in api.tokens.keys()}

    print(f"Method: {args.method}")
    print(f"Finished: {acc_finished}")
    print(f"Correct: {acc_correct}")
    # print(f"Finished: {acc_finished:.3f}%")
    print(f"Correct: {acc_correct:.3f}%")
    for key, value in costs.items():
        print(f"\t{key}: {value['total']:.3f}$")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve Game 24 using LLMs.")
    parser.add_argument("--provider", type=str, help="LLM Provider", choices=["openai", "together", "local", "groq"],
                        default="openai")
    parser.add_argument("--model", type=str, help="LLM Model", default="gpt-4.1-nano")
    parser.add_argument("--batch_size", type=int, help="CacheSaver's batch size", default=300)
    parser.add_argument("--timeout", type=float, help="CacheSaver's timeout", default=0.05)
    parser.add_argument("--temperature", type=float, help="Temperature for the model", default=1.0)
    parser.add_argument("--max_completion_tokens", type=int, help="Max completion tokens", default=100)
    parser.add_argument("--top_p", type=float, help="Top P for the model", default=1.0)
    parser.add_argument("--stop", type=str, nargs="+", help="Stop sequence for the model", default=None)
    parser.add_argument("--logprobs", action="store_true", help="Logprobs for the model")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset")
    parser.add_argument("--split", type=str, help="Split of the dataset",
                        choices=["mini", "train", "validation", "test", "single"], default="single")
    parser.add_argument("--share_ns", action="store_true", help="Share namespace between puzzles")
    parser.add_argument("--method", type=str, help="Method to use", choices=["foa", "tot", "rafa", "got"],
                        default="rafa")
    parser.add_argument("--conf_path", type=str, help="Path to corresponding config")
    parser.add_argument("--value_cache", action="store_true", help="Use value cache")
    args = parser.parse_args()

    # Create the directory if it doesn't exist
    log_dir = "logs/game24"
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO, filename=f"logs/game24/{args.method}.log", filemode="w")

    asyncio.run(run(args))
