from typing import NamedTuple

class Verification(NamedTuple):
    finished: bool
    correct: bool
    message: str

class Inference(NamedTuple):
    action: str = None
    thought: str = None
    value: int = None