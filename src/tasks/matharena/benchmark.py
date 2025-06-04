from typing import Tuple
import pandas as pd

from .state import StateMathArena
from ...typedefs import Benchmark
class BenchmarkMathArena(Benchmark):
    def __init__(self, path: str, split: str = "mini"):
        # Read dataset
        df = pd.read_json(path, lines=True, compression="gzip")
        df.reset_index(inplace=True)
        data = list(zip(df['problem'], df['answer']))

        # Split data
        if split == "mini":
            self.data = data[:3]
        elif split == "train":
            self.data = data[3:8]
        elif split == "validation":
            self.data = data[8:11]
        elif split == "test":
            self.data = data[11:15]
        else:
            raise ValueError("Invalid split name")

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx) -> Tuple[int, StateMathArena]:
        problem = self.data[idx][0]
        answer = self.data[idx][1]
        
        state = StateMathArena(
            problem=problem,
            answer=answer,
            current_state="",
            steps=[],
            randomness=0
        )
        return idx, state