from dataclasses import dataclass, field
from typing import Dict, List
from ...typedefs import State


@dataclass(frozen=True)
class StateHLE(State):
    # Unique ID and metadata
    id: str
    category: str
    raw_subject: str

     # Core QA pair
    question: str
    answer: str

    # Current state of reasoning
    current_state: str
    
    # Steps taken in solution
    steps: List[str]
    randomness: int = 0

    # The number of steps taken so far
    step_n: int
    
    # The value that the state had at its last evaluation
    values: Dict = field(default_factory=dict)


    def serialize(self) -> dict:
        """
        Returns a dictionary representation of the state.
        """
        return {
            "current_state": self.current_state,
            "steps": " -> ".join(self.steps),
        }

    def clone(self, randomness: int = None) -> "StateHLE":
        """
        Returns a new instance of StateHLE with an optional new randomness value.
        """
        return StateHLE(
            id=self.id,
            question=self.question,
            image=self.image,
            image_preview=self.image_preview,
            answer=self.answer,
            answer_type=self.answer_type,
            author_name=self.author_name,
            rationale=self.rationale,
            rationale_image=self.rationale_image,
            raw_subject=self.raw_subject,
            category=self.category,
            canary=self.canary,
            steps=self.steps,
            current_state=self.current_state,
            steps=self.steps,
            step_n=self.step_n,
            values=self.values,
            randomness=randomness or self.randomness
        )

    def get_seed(self) -> int:
        """
        Returns the randomness value associated with the state.
        """
        return self.randomness

    def __hash__(self) -> int:
        """
        Returns a hash of the current state.
        """
        return hash(str(self.serialize()))
