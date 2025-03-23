import pandas as pd
from typing import List

class DataGame24:
    def __init__(self, path: str):
        assert path.endswith(".csv.gz"), "Assumes the data is in a gzip compressed CSV file"
        self.path = path

    def get_set_idxs(self, set:str) -> List[int]:
        if set == "mini":
            indices = list(range(0,10))
        elif set == "train":
            indices = list(range(850,875)) + list(range(1025,1050))
        elif set == "validation":
            indices = list(range(875,900)) + list(range(1000,1025))
        elif set == "test":
            indices = list(range(900,1000))
        else:
            raise ValueError("Invalid set name")

        return indices
    
    def read(self, idx: int):
        puzzle = pd.read_csv(self.path, usecols=["Puzzles"], compression="gzip").iloc[idx]["Puzzles"]
        return puzzle