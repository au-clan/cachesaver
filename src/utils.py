import yaml
import random
from typing import List
import numpy as np 
from collections import namedtuple
import hashlib

def sha256(string: str):
    return hashlib.sha256(string.encode()).hexdigest()

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
        resampled_states = [state_records[i][2].duplicate(randomness) for i, randomness in zip(resampled_indices, new_randomness)]
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