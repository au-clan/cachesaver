import pandas as pd
from typing import Tuple

from .state import StateLogiQA
from ...typedefs import Benchmark


class BenchmarkLogiQA(Benchmark):
    def __init__(self, path: str, split: str = "mini"):

        df = pd.read_csv(path, usecols=["right_choice", "context", "question", "option_a", "option_b", "option_c",
                                        "option_d"], compression="gzip")
        df.reset_index(inplace=True)
        data = list(zip(df['index'], df['right_choice'], df['context'], df['question'], df['option_a'], df['option_b'],
                        df['option_c'], df['option_d']))

        if split == "mini":
            self.data = data[:10]
        if split == "single":
            self.data = data[:1]
        elif split == "train":
            self.data = data[850:875] + data[1025:1050]
        elif split == "validation":
            self.data = data[875:900] + data[1000:1025]
        elif split == "test":
            self.data = data[900:1000]
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
        # Note: Left None for randomness, which enforces a state.clone() call in the algorithm
        state = StateLogiQA(
            right_choice=right_choice,
            context=context,
            question=question,
            option_a=option_a,
            option_b=option_b,
            option_c=option_c,
            option_d=option_d)

        return index, state
