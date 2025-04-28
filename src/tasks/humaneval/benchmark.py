import pandas as pd
import random
from typing import Tuple

from .state import StateHumanEval
from ...typedefs import Benchmark

class BenchmarkHumanEval(Benchmark):
    def __init__(self, path: str, split: str = "mini") -> None:
        """
        Initializes the Benchmark for HumanEval dataset.
        """

        df = pd.read_csv(path, usecols=["prompt", "canonical_solution", "entry_point", "tests"], compression="gzip")
        df.reset_index(inplace=True)
        data = list(zip(df['index'], df['prompt'], df['canonical_solution'], df['entry_point'], df['tests']))

        if split == "mini":
            self.data = random.sample(data, 10)
        elif split == "train":
            self.data = random.sample(data, 50)
        elif split == "validation":
            self.data = random.sample(data[-100:], 50)
        elif split == "test":
            self.data = data[-50:] # <- Taken from reflexion
        elif split == "stratified":
            partition_samples = 50 // 3
            partition_size = len(self.data) // 3
            partitions = [self.data[i*partition_size:(i+1)*partition_size] for i in range(3)]
            self.data = [random.sample(partitions[i], partition_samples) for i in range(3)]
        else:
            raise ValueError("Invalid set name")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[int, StateHumanEval]:
        index = self.data[idx][0]
        signature, canonical_solution, entry_point, tests = self.data[idx][1:]

        state = StateHumanEval(
            puzzle=signature,
            current_state=signature,
            steps=[],
            canonical_solution=canonical_solution,
            entry_point=entry_point,
            tests=tests,
            randomness=None
        )
        return index, state