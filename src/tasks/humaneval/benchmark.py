import pandas as pd
import random
from typing import Tuple

from .state import StateHumanEval
from ... import BenchmarkFactory
from ...typedefs import Benchmark

@BenchmarkFactory.register
class BenchmarkHumanEval(Benchmark):
    def __init__(self, path: str, split: str = "mini") -> None:
        """
        Initializes the Benchmark for HumanEval dataset.
        """

        self.name = "humaneval"

        df = pd.read_csv(path, usecols=["prompt", "entry_point", "test"], compression="gzip")
        df.reset_index(inplace=True)
        data = list(zip(df['index'], df['prompt'], df['entry_point'], df['test']))

        if split == "mini":
            self.data = random.sample(data, 10)
        elif split == "train":
            self.data = random.sample(data, 50)
        elif split == "validation":
            self.data = random.sample(data[-100:], 50)
        elif split == "test":
            self.data = data[-50:] # <- Taken from reflexion
        elif split == "single":
            self.data = data[:1]
        else:
            raise ValueError("Invalid set name")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[int, StateHumanEval]:
        index = self.data[idx][0]
        signature, entry_point, test = self.data[idx][1:]

        state = StateHumanEval(
            puzzle=signature,
            current_state=signature,
            steps=[],
            entry_point=entry_point,
            test=test,
            randomness=None
        )
        return index, state
