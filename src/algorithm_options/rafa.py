from dataclasses import dataclass


@dataclass(frozen=True)
class RAFAOptions:
    max_step:int
    n_generate_sample: int
    n_evaluate_sample:int
    n_select_sample:int
    n_propose_sample:int