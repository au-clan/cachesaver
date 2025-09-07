import random
from typing import Tuple

import pandas as pd

from .state import StateMathArena
from ... import BenchmarkFactory
from ...typedefs import Benchmark

import sys

from .matharena_ethz.src.matharena.parser import parse_grading
def get_split_sizes(total_size, proportions=(0.1, 0.4, 0.25, 0.25)):
        """Calculate split sizes based on total dataset size"""
        sizes = [max(1, int(total_size * p)) for p in proportions]
        return sizes

@BenchmarkFactory.register
class BenchmarkMathArena(Benchmark):
    

    def __init__(self, path: str, split: str = "mini"):
        """
        Initializes the benchmark with the dataset.

        Args:
            data_path (str): Path to the dataset.
            split (str): Name of the dataset split (e.g., "mini", "train", "validation", "test").
        """

        self.name = "matharena"
        
        df = pd.read_json(path, lines=True,
                          compression='gzip')
        df.reset_index(inplace=True)
        
        # Parse the problems using the MathArena parser
        df['parsed_problem'] = df['problem'].apply(parse_grading)

        # Prepare the dataset
        data = list(zip(df['problem_idx'], df['parsed_problem'], df['answer']))

        # Compute the idxs for each subset
        valid_idxs = set(range(len(data)))

        total_samples = len(data)
        mini, train, val, test = get_split_sizes(total_samples)

        random.seed(0)
        
        mini_set_idxs = random.sample(list(valid_idxs), mini)
        valid_idxs = valid_idxs - set(mini_set_idxs)

        train_set_idxs = random.sample(list(valid_idxs), min(train, len(valid_idxs)))
        valid_idxs = valid_idxs - set(train_set_idxs)

        validation_set_idxs = random.sample(list(valid_idxs), min(val, len(valid_idxs)))
        valid_idxs = valid_idxs - set(validation_set_idxs)

        test_set_idxs = random.sample(list(valid_idxs), min(test, len(valid_idxs)))
        # valid_idxs = valid_idxs - set(validation_set_idxs)

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

    def __getitem__(self, idx) -> Tuple[int, StateMathArena]:
        """
        Returns the index and the state for the given index.

        Args:
            idx (int): Index of the data point.

        Returns:
            Tuple[int, StateMathArena]: Index and the corresponding state.
        """
        index = self.data[idx][0]
        parsed_problem = self.data[idx][1]
        answer = self.data[idx][2]

        # Create a state object
        state = StateMathArena(
            problem_idx=index,
            problem=parsed_problem,
            current_state="",
            steps=[],
            answer=answer
        )
        return index, state

