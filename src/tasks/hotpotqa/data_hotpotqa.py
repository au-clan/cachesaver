import pandas as pd
from typing import List, Tuple

class DataHotpotQA:
    def __init__(self, path: str):
        assert path.endswith(".csv.gz"), "Assumes the data is in a gzip compressed CSV file"
        self.path = path

    def get_set_idxs(self, set:str) -> List[int]:
        """
        Test set indices were picked based on the reflexion paper.
        Mini, train and validation sets were picked randomly (no intersection with test).

        """
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
        
        if set == "mini":
            indices = list(range(0,10))
        elif set == "train":
            indices = list(range(600, 650))
        elif set == "validation":
            list(range(650, 700))
        elif set == "test":
            indices = test_set_idxs
        else:
            raise ValueError("Invalid set name")
        
        return indices
    def read(self, idx: int) -> Tuple[str, str]:
        question = pd.read_csv(self.path, usecols=["question"], compression="gzip").iloc[idx]["question"]
        answer = pd.read_csv(self.path, usecols=["answer"], compression="gzip").iloc[idx]["answer"]
        return question, answer