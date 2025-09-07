import random

import pandas as pd
from typing import Tuple

from .state import StateLogiQA
from ... import BenchmarkFactory
from ...typedefs import Benchmark


@BenchmarkFactory.register
class BenchmarkLogiQA(Benchmark):
    def __init__(self, path: str, split: str = "mini"):

        self.name = "logiqa"

        df = pd.read_csv(path, usecols=["right_choice", "context", "question", "option_a", "option_b", "option_c",
                                        "option_d"], compression="gzip")
        df.reset_index(inplace=True)
        data = list(zip(df['index'], df['right_choice'], df['context'], df['question'], df['option_a'], df['option_b'],
                        df['option_c'], df['option_d']))

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
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[int, StateLogiQA]:
        index = self.data[idx][0]
        right_choice = self.data[idx][1]
        context = self.data[idx][2]
        question = self.data[idx][3]
        option_a = self.data[idx][4]
        option_b = self.data[idx][5]
        option_c = self.data[idx][6]
        option_d = self.data[idx][7]

        # Create a state object
        # Note: Left None for randomness, which enforces a state.clone() call in the method
        state = StateLogiQA(
            context=context,
            question=question,
            option_a=option_a,
            option_b=option_b,
            option_c=option_c,
            option_d=option_d,
            current_state=question,
            steps=[],
            correct_option=right_choice,
            randomness=None
            )

        return index, state