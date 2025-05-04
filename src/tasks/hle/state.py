from dataclasses import dataclass

from ...typedefs import State


@dataclass(frozen=True)
class StateHLE(State):
    id: str
    question: str
    image: str
    image_preview: str
    answer: str
    answer_type: str
    author_name: str
    rationale: str
    rationale_image: str
    raw_subject: str
    category: str
    canary: str

    def serialize(self) -> dict:
        """
        Returns a dictionary representation of the state.
        """
        return {
            "id": self.id,
            "question": self.question,
            "image": self.image,
            "image_preview": self.image_preview,
            "answer": self.answer,
            "answer_type": self.answer_type,
            "author_name": self.author_name,
            "rationale": self.rationale,
            "rationale_image": self.rationale_image,
            "raw_subject": self.raw_subject,
            "category": self.category,
            "canary": self.canary
        }

    def clone(self, randomness: int = None) -> "StateHLE":
        """
        Returns a new instance of StateHotpotQA with an optional new randomness value.
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
            canary=self.canary
        )

    def get_seed(self) -> int:
        """
        Returns the randomness value associated with the state.
        """
        pass
        return self.randomness

    def __hash__(self) -> int:
        """
        Returns a hash of the current state.
        """
        return hash(str(self.serialize()))
