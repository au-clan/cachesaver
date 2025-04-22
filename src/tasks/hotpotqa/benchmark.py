import random
import pandas as pd
from typing import Tuple
from langchain import Wikipedia
from langchain.agents.react.base import DocstoreExplorer

from .state import StateHotpotQA
from ...typedefs import Benchmark

class BenchmarkHotpotQA(Benchmark):
    def __init__(self, path: str, split: str = "mini"):
        """
        Initializes the benchmark with the dataset.

        Args:
            data_path (str): Path to the dataset.
            split (str): Name of the dataset split (e.g., "mini", "train", "validation", "test").
        """
        df = pd.read_csv(path, usecols=["question", "answer"], compression="gzip")
        df.reset_index(inplace=True)
        data = list(zip(df['index'], df['question'], df['answer']))

        # Compute the idxs for each subset
        valid_idxs = set(range(len(data)))

        # Taken from reflexion
        test_set_idxs = [4367, 1867, 3504, 5015,  554, 6979, 2197, 2942, 6378,  263, 5283,
                            6994, 5777, 7323, 6266, 7168, 6429, 5542, 6226, 5464, 4078, 2820,
                            251, 6593, 4690, 3149, 4309, 7158,  503, 2402, 3197, 6754, 3159,
                            3349, 2850, 5641, 2879, 3540, 1061, 5664, 4617, 4597, 7187, 3309,
                            2287,  230,  318, 1042, 5608, 7322, 4526, 4734, 1941, 4756, 1078,
                            3977, 1511, 3608, 5950,  169, 2922, 6864, 1790, 2569, 1608, 4240,
                            132, 1566, 2183, 5212, 1737, 1543, 5865, 5785, 5976, 2692, 4563,
                            468, 6210, 2399, 6733, 4159, 7315, 6109, 7031, 4099, 4094, 5926,
                            4545,  996,   37,   61,  472,  101, 3340, 4205, 6446, 3450, 3734,
                            3317]
        valid_idxs = valid_idxs - set(test_set_idxs)

        random.seed(0)
        mini_set_idxs = random.sample(list(valid_idxs), 10)
        valid_idxs = valid_idxs - set(mini_set_idxs)

        train_set_idxs = random.sample(list(valid_idxs), 50)
        valid_idxs = valid_idxs - set(train_set_idxs)

        validation_set_idxs = random.sample(list(valid_idxs), 50)
        valid_idxs = valid_idxs - set(validation_set_idxs)

        if split == "mini":
            self.data = [data[i] for i in mini_set_idxs]
        if split == "single":
            self.data = mini_set_idxs[0]
        elif split == "train":
            self.data = [data[i] for i in train_set_idxs]
        elif split == "validation":
            self.data = [data[i] for i in validation_set_idxs]
        elif split == "test":
            self.data = [data[i] for i in test_set_idxs]
        else:
            raise ValueError("Invalid set name")
        
    def __len__(self) -> int:
        """
        Returns the length of the benchmark dataset.
        """
        return len(self.data)
    
    def __getitem__(self, idx) -> Tuple[int, StateHotpotQA]:
        """
        Returns the index and the state for the given index.

        Args:
            idx (int): Index of the data point.

        Returns:
            Tuple[int, StateHotpotQA]: Index and the corresponding state.
        """
        index = self.data[idx][0]
        x = self.data[idx][1]
        y = self.data[idx][2]

        # Create a state object
        # Note: Left None for randomness, which enforces a state.clone() call in the algorithm
        state = StateHotpotQA(
            puzzle=x,
            current_state="",
            steps=[],
            answer=y,
            docstore=DocstoreExplorer(Wikipedia()),
            randomness=None
        )
        return index, state

