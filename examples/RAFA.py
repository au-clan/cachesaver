import asyncio
import copy
import json
import os
import sys
from dataclasses import asdict

from diskcache import Cache
from omegaconf import OmegaConf

from cachesaver.pipelines import OnlineAPI
from client_wrapper.groq_wrapper import GroqModel1
from src.frameworks.framework_rafa import FrameworkRAFA
from src.tasks.game24.state_game24 import StateGame24

sys.path.append('..')

from src.tasks import EnvironmentBasic
from src.agents import AgentLLM
from src.utils import tokens2cost


async def main():
    # Choose task
    task = "game24"  # "hotpotqa" or "game24"

    # Config
    config = OmegaConf.load(f'../configs/{task}/config_rafa_{task}.yaml')

    # Environment
    env = EnvironmentBasic.create(task=task, data_path=f"../datasets/dataset_{task}.csv.gz")

    # Cache
    cache = Cache(f"../caches/{task}")

    # LLM Client and Model
    # client = AsyncTogether(api_key=os.environ.get('TOGETHER_API_KEY_PERS'))
    model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    model = GroqModel1(api_key=os.getenv("GROQ_API_KEY"), model="gemma2-9b-it")

    # CacheSaver API
    api = OnlineAPI(
        model=model,
        cache=cache,
        batch_size=30,
        timeout=0.5,
    )

    # Agent
    agent = AgentLLM(api=api)

    # log = {}
    # Complete Framework
    fw = FrameworkRAFA(config, agent, env)

    for seed in range(config.run.repeats):
        log = {}
        step_cache = None
        value_cache = {}

        # Get the puzzle indexes for the current set
        set = config.run.set
        puzzle_idxs = env.data.get_set_idxs(set)[:config.run.debugging]

        # Run the puzzles in parallel
        puzzle_coroutines = [
            fw.run(
                puzzle_idx=puzzle_idx,
                namespace=str(puzzle_idx),
                seed=seed,
                value_cache=value_cache,
                step_cache=step_cache,

            )
            for puzzle_idx in puzzle_idxs
        ]
        results = await asyncio.gather(*puzzle_coroutines)
        states, verifications = map(list, zip(*results))
        # print("results")
        # Compute quality
        # success = [any(v.finished and v.correct for v in vs) for vs in verifications]
        # accuracy = sum(success) / len(success)

        # Merge the logs
        # for l in logs:
        #     log.update(l)


        # Saving additional info in the log
        # log["results"] = results

        ##todo reduce logging in the following 2 lines
        json_ready_log = copy.deepcopy(states)

        # Convert all StateGame24 objects inside the log to dicts
        json_ready_log["Experiment_data"] = [
            asdict(item) if isinstance(item, StateGame24) else item
            for item in json_ready_log["Experiment_data"]
        ]
        log_copy= copy.deepcopy(states)
        log["Experiment_data"]=log_copy
        log["Info"] = {}

        log["Info"]["Cost"] = {
            "total": tokens2cost(agent.tokens["total"], model_name),
            "cached": tokens2cost(agent.tokens["cached"], model_name),
            "generated": tokens2cost(agent.tokens["generated"], model_name),
        }

        config = OmegaConf.to_container(config, resolve=True)
        log["Info"]["LLM"] = {"model": config["api"]["model"], "parameters": config["api"]["parameters"]}
        log["Info"]["Framework"] = config["framework"]
        log["Task"] = config["task"]
        log["Run"] = {"seed": seed, "set": config["run"]["set"], "debugging": config["run"]["debugging"]}
        log["Quality"] = {"accuracy": verifications, "success": verifications} #todo change to correct values later

        # Save the results of the trial
        log_dir = os.path.join(os.getcwd().split("cachesaver")[0] + "cachesaver", config["logs"]["log_dir"])
        os.makedirs(log_dir, exist_ok=True)
        # log_path = os.path.join(log_dir, f"{config.logs.log_name}_{seed}.json")
        log_path = os.path.join(log_dir, f"{config['logs']['log_name']}_{seed}.json")
        with open(log_path, "w") as f:
            json.dump(log, f, indent=4)
        #todo add the results to the json log file
        # with open(log_path, "w") as f:
        #     json.dump(results, f, indent=4)


if __name__ == "__main__":
    asyncio.run(main())
