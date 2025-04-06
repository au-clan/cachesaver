import pandas as pd

from .state_game24 import StateGame24
from ...typedefs import Benchmark

class BenchmarkGame24(Benchmark):
    def __init__(self, data_path: str, set_name: str = "mini"):

        data = pd.read_csv(data_path, usecols=["Puzzles"], compression="gzip")["Puzzles"].tolist()

        if  set_name == "mini":
            self.data = data[:10]
        elif set_name == "train":
            self.data = data[850:875] + data[1025:1050]
        elif set_name == "validation":
            self.data = data[875:900] + data[1000:1025]
        elif set_name == "test":
            self.data = data[900:1000]
        else:
            raise ValueError("Invalid set name")
        
        # No labels for this dataset
        self.labels = [None] * len(self.data)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        # Create a state object
        # Note: Left None for randomness, to enforce a state.seed() call later on
        state = StateGame24(
            puzzle=x,
            current_state=x,
            steps=[],
            randomness=None
        )
        return state, y
    
    