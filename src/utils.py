import os
import re
import time
import random
import logging

import numpy as np 

from argparse import Namespace
from omegaconf import OmegaConf
from typing import List, Awaitable, Tuple, Any

def assign_ns(length: int, fraction: float) -> List[int]:
    """
    Assigns a list of integers of valuesfrom 0 to length-1, where a fraction of the list
    is the same (-1) and the rest are unique integers.
    """
    x_same = round(length * fraction)
    x_unique = length - x_same
    ns_same = [-1] * x_same
    ns_unique = list(range(x_unique))
    ns = ns_same + ns_unique
    random.seed(0)
    random.shuffle(ns)
    assert len(ns) == length, f"Expected output length {length}, but got {len(ns)}"
    return ns

import re

def clean_log(file_path: str):
    # Define all patterns you want to remove
    patterns = [
        re.compile(r'^INFO:httpx:HTTP Request: POST https://api\.openai\.com/v1/chat/completions "HTTP/1\.1 200 OK"$'),
        re.compile(r'^INFO:openai\._base_client:Retrying request to /chat/completions in \d+(\.\d+)? seconds$')
    ]

    # Read the file and filter out matching lines
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Write back only the lines that do not match the pattern
    with open(file_path, 'w') as file:
        for line in lines:
            line_stripped = line.strip()
            if any(pattern.match(line_stripped) for pattern in patterns):
                continue
            if line_stripped.startswith(("INFO:src.", "MCTS")):
                continue
            elif line_stripped.startswith("INFO:__main__:") or line_stripped=="":
                file.write(line)
            else:
                continue

def tokens2cost(tokens: dict, model_name: str) -> dict:
    catalog = {
        # Llama-3 models
        "meta-llama/Llama-3.3-70B-Instruct-Turbo" : {"in": 0.88, "out": 0.88},
        "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo" : {"in": 0.88, "out": 0.88},
        "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo" : {"in": 0.18, "out": 0.18},
        "meta-llama/Llama-3.1-8B-Instruct" : {"in": 0.05, "out": 0.05},  # guesstimated values
        
        # Llama-4 models
        "unsloth/Llama-4-Scout-17B-16E-Instruct" : {"in": 0.15, "out": 0.15},  # guesstimated values
        "unsloth/Llama-4-Maverick-17B-128E-Instruct-FP8" : {"in": 0.15, "out": 0.15},  # guesstimated values

        # Llama 4 models (Together AI)
        "meta-llama/Llama-4-Scout-17B-16E-Instruct" : {"in": 0.18, "out": 0.59},
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8" : {"in": 0.27, "out": 0.85},

        # GPT-4o models
        "gpt-4o": {"in": 2.50, "out": 10.00},
        "gpt-4o-mini": {"in": 0.15, "out": 0.60},

        # GPT-3.5 models
        "gpt-3.5-turbo": {"in": 0.50, "out": 1.50},
        
        # GPT-4.1 models
        "gpt-4.1-nano": {"in": 0.10, "out": 0.40},
        "gpt-4.1-mini": {"in": 0.40, "out": 1.60},
        "gpt-4.1": {"in": 2.00, "out": 8.00},

        # GPT-5 models
        "gpt-5-nano": {"in": 0.05, "out": 0.40},
        "gpt-5-mini": {"in": 0.25, "out": 2.00},
        "gpt-5": {"in": 1.25, "out": 10.00},


    }

    catalog["llama-3.3-70b-specdec"] = catalog["meta-llama/Llama-3.3-70B-Instruct-Turbo"]
    catalog["llama-3.2-90b-vision-preview"] = catalog["meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo"]
    catalog["llama-3.1-8b-instruct"] = catalog["meta-llama/Llama-3.1-8B-Instruct"]
    catalog["llama-4-scout-17b"] = catalog["unsloth/Llama-4-Scout-17B-16E-Instruct"]
    catalog["llama-4-maverick-17b"] = catalog["unsloth/Llama-4-Maverick-17B-128E-Instruct-FP8"]
    
    price_in = catalog[model_name]["in"] * tokens["in"] / 1e6
    price_out = catalog[model_name]["out"] * tokens["out"] / 1e6
    return {"in": price_in, "out": price_out, "total": price_in + price_out}

class Resampler: 
    def __init__(self, randomness: int):
        self.randomness = randomness 

    def resample(self, state_records, n_picks, resampling_method):
        """
        Resample states based on their values.

        Inputs:
            - state_records: List of tuples (state_identifier, state_value, state)
            - n_picks: Number of states to resample
            - resampling_method: Method to use for resampling
            - include_init: Whether to include the initial state in the resampling process
        
        Outputs:
            - resampled_states: List of resampled states
            - resampled_indices: List of indices of the resampled states in the original state_records
        """
        methods = {
            "linear": Resampler.linear,
            "linear_filtered": Resampler.linear_filtered,
            "max": Resampler.max,
            "max_unique": Resampler.max_unique, 
            "percentile": Resampler.percentile
        }

        if resampling_method not in methods:
            raise ValueError(f"Invalid resampling method: {resampling_method}\nValid methods: {methods.keys()}")
        
        if n_picks == 0 or len(state_records) == 0:
            return [], []

        # Get probabilities for each state based on values
        probabilities = methods[resampling_method]([value for _, value, _ in state_records])
        np.random.seed(self.randomness)
        resampled_indices = np.random.choice(range(len(state_records)), size=n_picks, p=probabilities, replace=True).tolist()
        
        # Resample states based on resampled_indices
        random.seed(self.randomness)
        new_randomness = [random.randint(1, 1000) for _ in range(n_picks)]
        self.randomness = new_randomness[-1]
        resampled_states = [state_records[i][2].clone(randomness) for i, randomness in zip(resampled_indices, new_randomness)]
        return resampled_states, resampled_indices
    
    @staticmethod
    def linear(values: List[float])-> List[float]:
        """
        Compute the linear probability of each value.
        """
        eps = 1e-6
        values = [value + eps for value in values]
        total = sum(values)
        return [value / total for value in values]
    
    @staticmethod
    def linear_filtered(values: List[float], threshold: float=0.5)-> List[float]:
        """
        Computes the linear probability of each value, but filters out values below a certain threshold.
        """
        max_value = np.max(values)
        values = [value if value>= max_value * threshold else 0 for value in values]
        return Resampler.linear(values)

    @staticmethod
    def max(values: List[float])-> List[float]:
        """
        Computes uniform probability of highest values solely.
        """
        max_value = max(values)
        values = [value if value==max_value else 0 for value in values]
        total = sum(values)
        if total == 0:
            return Resampler.linear(values)
        else:
            return [value / total for value in values]
    
    @staticmethod
    def max_unique(values: List[float])-> List[float]:
        """
        Computes uniform probability of highest values solely.
        """
        max_value = max(values)
        values = [1 if value==max_value else 0 for value in values]
        total = sum(values)
        if total == 0:
            return [1] + [0] * (len(values) - 1)
        else:
            first_one_index = values.index(1)
            values = [0] * len(values)
            values[first_one_index] = 1
            return values
    
    @staticmethod
    def percentile(values: List[float], percentile: float=0.75) -> List[float]:
        """
        Computes the linear probability considering only the highest percentile values.
        """
        threshold = np.percentile(values, percentile)
        values = [value if value >= threshold else 0 for value in values]
        return Resampler.linear(values)
    
async def timed(label: str, coroutine: Awaitable) -> Tuple[str, float, Any]:
    # Start timing
    start = time.perf_counter()
    
    # Await / Execute the coroutine
    result = await coroutine
    
    # End timing
    end = time.perf_counter()
    duration = end - start

    return label, duration, result
    
def initial_logging(
        logger: logging.Logger, 
        args:Namespace, 
        log_path: str
        ):
    
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    f_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    f_handler.setLevel(logging.INFO)
    logger.addHandler(f_handler)
    logger.setLevel(logging.INFO)

    logger.info("Script: simple.py")
    logger.info("Description: Solve a specified task using a specified method.\n")

    logger.info("General information:")
    logger.info("\tMethod: %s", args.method)
    logger.info("\tBenchmark: %s", args.benchmark)
    logger.info("\tSplit: %s", args.split)
    logger.info("\tDataset Path: %s", args.dataset_path)
    logger.info("\tUsing method's internal cache: %s\n", args.value_cache)

    logger.info("Method Configuration:")
    config = OmegaConf.load(f"scripts/configs/{args.benchmark}.yaml")[args.method]
    for key, value in config.items():
        logger.info(f"\t{key}: {value}")
    logger.info("\n")
    

    logger.info("LLM Information:")
    logger.info("\tProvider: %s", args.provider)
    logger.info("\tModel: %s", args.model)
    logger.info("\tTemperature: %f", args.temperature)
    logger.info("\tMax Completion Tokens: %d", args.max_completion_tokens)
    logger.info("\tTop-p: %f", args.top_p)
    logger.info("\tStop: %s", args.stop)
    logger.info("\tLogprobs: %s\n", args.logprobs)

    logger.info("CacheSaver Information:")
    logger.info("\tBatch Size: %d", args.batch_size)
    logger.info("\tTimeout: %f", args.timeout)
    logger.info("\tAllow Batch Overflow: %d", args.allow_batch_overflow)
    logger.info("\tNameSpace : %f\n", args.ns_ratio)

# TODO: Clarify types in definition: Importing creates circular import
def final_logging(
        logger: logging.Logger, 
        api: "src.models.API", 
        clocktime: float, 
        durations: List[float], 
        evaluations: List[Any]
        ):


    if len(api.tabs) > 1:
        logger.info("API Detailed Information (per tab)")
        for tab in api.tabs:
            logger.info(f"\tTab: {tab}")

            # Latency
            latencies = api.latencies[tab]
            logger.info("\t\tLatencies (in seconds): %s\n", latencies)

            # Reuse
            reuse = api.reuse[tab]
            logger.info("\t\tReuse (number of uses): %s\n", reuse.values())

            # Calls
            calls = api.calls[tab]
            logger.info("\t\tCalls (total): %s", calls["total"])
            logger.info("\t\tCalls (saved by cacher): %s", calls["cacher"])
            logger.info("\t\tCalls (saved by deduplicator): %s\n", calls["deduplicator"])

            # Tokens
            tokens = api.tokens[tab]
            logger.info("\t\tTokens (total): in %s, out %s", tokens["total"]["in"], tokens["total"]["out"])
            logger.info("\t\tTokens (saved by cacher): in %s, out %s", tokens["cacher"]["in"], tokens["cacher"]["out"])
            logger.info("\t\tTokens (saved by deduplicator): in %s, out %s\n", tokens["duplicator"]["in"], tokens["duplicator"]["out"])

            # Cost
            cost = {key: tokens2cost(tokens[key], api.model) for key in tokens.keys()}
            logger.info("\t\tCost (total): in $%f, out $%f, total $%f", cost["total"]["in"], cost["total"]["out"], cost["total"]["total"])
            logger.info("\t\tCost (saved by cacher): in $%f, out $%f, total $%f", cost["cacher"]["in"], cost["cacher"]["out"], cost["cacher"]["total"])
            logger.info("\t\tCost (saved by deduplicator): in $%f, out $%f, total $%f\n", cost["duplicator"]["in"], cost["duplicator"]["out"], cost["duplicator"]["total"])

        # Moving to API Summed Information
        logger.info("API Summed Information (all tabs)")
    else:
        logger.info("API Information")
    

    # Latency
    all_latencies = [lat for tab in api.tabs for lat in api.latencies[tab]]
    logger.info("\tSummed Latencies (in seconds): %s\n", all_latencies)

    # Reuse
    all_reuse = {key: sum(api.reuse[tab].get(key, 0) for tab in api.tabs) for tab in api.tabs for key in api.reuse[tab].keys()}
    logger.info("\tSummed Reuse (number of uses): %s\n", all_reuse.values())

    # Calls
    all_calls = {key: sum(api.calls[tab][key] for tab in api.tabs) for key in ["total", "cacher", "deduplicator"]}
    logger.info("\tSummed Calls (total): %s", all_calls["total"])
    logger.info("\tSummed Calls (saved by cacher): %s", all_calls["cacher"])
    logger.info("\tSummed Calls (saved by deduplicator): %s\n", all_calls["deduplicator"])

    # Tokens
    all_tokens = {key: {"in": sum(api.tokens[tab][key]["in"] for tab in api.tabs), "out": sum(api.tokens[tab][key]["out"] for tab in api.tabs)} for key in ["total", "cacher", "duplicator"]}
    logger.info("\tSummed Tokens (total): in %s, out %s", all_tokens["total"]["in"], all_tokens["total"]["out"])
    logger.info("\tSummed Tokens (saved by cacher): in %s, out %s", all_tokens["cacher"]["in"], all_tokens["cacher"]["out"])
    logger.info("\tSummed Tokens (saved by deduplicator): in %s, out %s\n", all_tokens["duplicator"]["in"], all_tokens["duplicator"]["out"])

    # Cost
    all_cost = {key: tokens2cost(all_tokens[key], api.model) for key in all_tokens.keys()}
    logger.info("\tSummed Cost (total): in $%f, out $%f, total $%f", all_cost["total"]["in"], all_cost["total"]["out"], all_cost["total"]["total"])
    logger.info("\tSummed Cost (saved by cacher): in $%f, out $%f, total $%f", all_cost["cacher"]["in"], all_cost["cacher"]["out"], all_cost["cacher"]["total"])
    logger.info("\tSummed Cost (saved by deduplicator): in $%f, out $%f, total $%f\n", all_cost["duplicator"]["in"], all_cost["duplicator"]["out"], all_cost["duplicator"]["total"])



    
    # Total duration
    logger.info("Total clocktime (in seconds): %f", clocktime)
    logger.info("Individual durations of each sample (in seconds): %s\n", list(durations))

    # Evaluations
    correct = [max(agent_result[1] for agent_result in e) for e in evaluations]
    logger.info("Correct: %s", correct)
    logger.info("Average correctness: %f", sum(correct) / len(correct))



    
    