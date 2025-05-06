import pandas as pd
from typing import Tuple

from .state import StateSciBench
from ...typedefs import Benchmark

class BenchmarkSciBench(Benchmark):
    def __init__(self, path: str, split: str = "mini", task: str = "class"):
        """
        Initializes the benchmark with the dataset.

        Args:
            data_path (str): Path to the dataset.
            split (str): Name of the dataset split (e.g., "mini", "train", "validation", "test").
            task (str): Name of the task to be performed.
        """
        df = pd.read_csv(path, compression="gzip")
        df = df[df.task == task].copy()
        df.reset_index(inplace=True)
        data = list(zip(df['index'], df['content'], df['answer']))
        
        # Calculate split sizes
        single_instance = data.pop(0)
        total = len(data)
        mini = round(total * 0.05)
        train = round(total * 0.10)
        valid = round(total * 0.15)
        
        
        if split == "single":
            self.data = [single_instance]
        elif split == "mini":
            self.data = data[:mini]
        elif split == "train":
            self.data = data[mini: mini + train]
        elif split == "validation":
            self.data = data[mini + train: mini + train + valid]
        elif split == "test":
            self.data = data[mini + train + valid:]
        else:
            raise ValueError("Invalid set name")
        
    def __len__(self) -> int:
        """
        Returns the length of the benchmark dataset.
        """
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[int, StateSciBench]:
        """
        Returns the index and the state for the given index.

        Args:
            idx (int): Index of the data point.

        Returns:
            Tuple[int, StateSciBench]: Index and the corresponding state.
        """
        index = self.data[idx][0]
        x = self.data[idx][1]
        y = self.data[idx][2]

        # Create a state object
        # Note: Left None for randomness, which enforces a state.clone() call in the algorithm
        state = StateSciBench(
            puzzle=x,
            current_state="",
            steps=[],
            step_n=0,
            answer=y,
            randomness=None
        )
        return index, state






     
