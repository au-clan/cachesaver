import re
import random
from typing import List
import numpy as np 

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
    sps =  model_name.split('/')
    if len(sps) > 2:
        model_name = '/'.join(sps[-2:])

    catalog = {
        "meta-llama/Llama-3.3-70B-Instruct-Turbo" : {"in": 0.88, "out": 0.88},
        "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo" : {"in": 0.88, "out": 0.88},
        "meta-llama/Llama-3.1-8B-Instruct" : {"in": 0.05, "out": 0.05},  # guesstimated values
        "unsloth/Llama-4-Scout-17B-16E-Instruct" : {"in": 0.15, "out": 0.15},  # guesstimated values
        "Llama-4-Scout-17B-16E-Instruct/" : {"in": 0.15, "out": 0.15}, # duplicating
        "unsloth/Llama-4-Maverick-17B-128E-Instruct-FP8" : {"in": 0.15, "out": 0.15},  # guesstimated values
        "gpt-4o": {"in": 2.50, "out": 10.00},
        "gpt-4o-mini": {"in": 0.15, "out": 0.60},
        "gpt-3.5-turbo": {"in": 0.50, "out": 1.50},

        # GPT-4.1 models
        "gpt-4.1-nano": {"in": 0.10, "out": 0.40},
        "gpt-4.1-mini": {"in": 0.40, "out": 1.60},
        "gpt-4.1": {"in": 2.00, "out": 8.00},

        # LLama 4 models (Together AI)
        "meta-llama/Llama-4-Scout-17B-16E-Instruct" : {"in": 0.18, "out": 0.59},
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8" : {"in": 0.27, "out": 0.85},

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