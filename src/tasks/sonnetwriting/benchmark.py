import random
from typing import Tuple

import pandas as pd

from .state import StateSonnetWriting
from ... import BenchmarkFactory
from ...typedefs import Benchmark


@BenchmarkFactory.register
class BenchmarkSonnetWriting(Benchmark):
    def __init__(self, path: str, split: str = "mini"):
        """
        Initializes the benchmark with the dataset.

        Args:
            data_path (str): Path to the dataset.
            split (str): Name of the dataset split (e.g., "mini", "train", "validation", "test").
        """

        self.name = "sonnetwriting"
        
        df = pd.read_json(path.replace("csv", "jsonl"), lines=True,
                          compression='gzip')
        df.reset_index(inplace=True)
        # todo 200 entires in this dataset
        data = list(
            zip(df['index'], df['input'], df['target']))

        # Compute the idxs for each subset
        valid_idxs = set(range(len(data)))

        random.seed(0)
        mini_set_idxs = random.sample(list(valid_idxs), 10)
        valid_idxs = valid_idxs - set(mini_set_idxs)

        train_set_idxs = random.sample(list(valid_idxs), 50)
        valid_idxs = valid_idxs - set(train_set_idxs)

        validation_set_idxs = random.sample(list(valid_idxs), 50)
        valid_idxs = valid_idxs - set(validation_set_idxs)

        test_set_idxs = random.sample(list(valid_idxs), 50)
        valid_idxs = valid_idxs - set(validation_set_idxs)

        if split == "single":
            self.data = data[:1]
        elif split == "mini":
            self.data = [data[i] for i in mini_set_idxs]
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

    def __getitem__(self, idx) -> Tuple[int, StateSonnetWriting]:
        """
        Returns the index and the state for the given index.

        Args:
            idx (int): Index of the data point.

        Returns:
            Tuple[int, StateHotpotQA]: Index and the corresponding state.
        """
        index = self.data[idx][0]
        input = self.data[idx][1]
        target = self.data[idx][2]

        # Create a state object
        # Note: Left None for randomness, which enforces a state.clone() call in the method
        state = StateSonnetWriting(
            puzzle=input,
            current_state=input,
            steps=[],
            target=target,
            randomness=None
        )
        return index, state
