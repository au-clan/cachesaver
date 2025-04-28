import random
from typing import Tuple

import pandas as pd

from .state import StateHumanEval
from ...typedefs import Benchmark


class BenchmarkHumanEval(Benchmark):
    def __init__(self, path: str, split: str = "mini"):
        """
        Initializes the benchmark with the dataset.

        Args:
            data_path (str): Path to the dataset.
            split (str): Name of the dataset split (e.g., "mini", "train", "validation", "test").
        """

        df = pd.read_json(path, lines=True,
                          compression='gzip')
        df.reset_index(inplace=True)
        data = list(
            zip(df['index'], df['task_id'], df['prompt'], df['entry_point'], df['canonical_solution'], df['test']))

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
        if split == "mini":
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

    def __getitem__(self, idx) -> Tuple[int, StateHumanEval]:
        """
        Returns the index and the state for the given index.

        Args:
            idx (int): Index of the data point.

        Returns:
            Tuple[int, StateHumanEval]: Index and the corresponding state.
        """
        index = self.data[idx][0]
        task_id = self.data[idx][1]
        prompt = self.data[idx][2]
        entry_point = self.data[idx][3]
        canonical_solution = self.data[idx][4]
        test = self.data[idx][5]

        # Create a state object

        state = StateHumanEval(
            prompt=prompt,
            canonical_solution=canonical_solution,
            entry_point=entry_point,
            test=test,
        )
        return task_id, state
