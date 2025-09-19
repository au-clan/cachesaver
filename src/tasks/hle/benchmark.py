import random
from typing import Tuple

import pandas as pd

from .state import StateHLE

from ... import BenchmarkFactory

from ...typedefs import Benchmark


@BenchmarkFactory.register
class BenchmarkHLE(Benchmark):
    def __init__(self, path: str, split: str = "mini", max_len: int = None):
        """
        Initializes the benchmark with the dataset.

        Args:
            data_path (str): Path to the dataset.
            split (str): Name of the dataset split (e.g., "mini", "train", "validation", "test").
        """

        self.name = "hle"
        df = pd.read_json(path.replace("csv", "jsonl"), lines=True,
                          compression='gzip')
        df.reset_index(inplace=True)

        # Check if 'image_preview' exists
        if 'image_preview' not in df.columns:
            print("Warning: 'image_preview' column is missing from the dataset.")
            df['image_preview'] = None  # Add a placeholder column if missing

        if 'rationale_image' not in df.columns:
            print("Warning: 'rationale_image' column is missing from the dataset.")
            df['rationale_image'] = None 

        # Existing code to process the dataset
        data = list(
            zip(df["index"], df['id'], df['question'], df['image'], df['image_preview'], df['answer'], df['answer_type'],
                df['author_name'], df['rationale'], df['rationale_image'], df['raw_subject'], df['category'], df['canary'])
        )

        valid_idxs = set(range(len(data)))
    
        # Ensure integer seed
        random.seed(0)  # Use explicit integer
        
        # Fixed size sampling with integer types
        mini_size = 10
        other_size = 50  # for train, validation, test
        
        mini_set_idxs = random.sample(list(valid_idxs), mini_size)
        valid_idxs = valid_idxs - set(mini_set_idxs)

        train_set_idxs = random.sample(list(valid_idxs), other_size)
        valid_idxs = valid_idxs - set(train_set_idxs)

        validation_set_idxs = random.sample(list(valid_idxs), other_size)
        valid_idxs = valid_idxs - set(validation_set_idxs)

        test_set_idxs = random.sample(list(valid_idxs), other_size)
        valid_idxs = valid_idxs - set(test_set_idxs)
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
        
        if max_len:
            self.data = self.data[:max_len]
        
    def __len__(self) -> int:
        """
        Returns the length of the benchmark dataset.
        """
        return len(self.data)
    
    def __getitem__(self, idx) -> Tuple[int, StateHLE]:
        """
        Returns the index and the state for the given index.

        Args:
            idx (int): Index of the data point.

        Returns:
            Tuple[int, StateHotpotQA]: Index and the corresponding state.
        """
        index = self.data[idx][0]
        id = self.data[idx][1]
        question = self.data[idx][2]
        image = self.data[idx][3]
        image_preview =self.data[idx][4]
        answer = self.data[idx][5]
        answer_type = self.data[idx][6]
        author_name = self.data[idx][7]
        rationale = self.data[idx][8]
        rationale_image = self.data[idx][9]
        raw_subject = self.data[idx][10]
        category = self.data[idx][11]
        canary = self.data[idx][10]

        # Create a state object
        # Note: Left None for randomness, which enforces a state.clone() call in the method
        state = StateHLE(
            id=id,
            question=question,
            image=image,
            image_preview=image_preview,
            answer=answer,
            answer_type=answer_type,
            author_name=author_name,
            rationale=rationale,
            rationale_image=rationale_image,
            raw_subject=raw_subject,
            category=category,
            canary=canary,
            randomness=None
        )
        return index, state

