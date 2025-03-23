from dataclasses import dataclass
from typing import List
from langchain.agents.react.base import DocstoreExplorer

from ..basic import StateBasic

@dataclass(frozen=True)
class StateHotpotQA(StateBasic):
    # The question to answer
    puzzle: str

    # Current state (progress) towards answering the question
    current_state: str

    # Thoughts taken towards answering the question
    thoughts: List[str]
    
    # Actions taken towards answering the question
    actions: List[str]

    # Observations seen towards answering the question
    observations: List[str]

    # Steps (can be thoughs, actions, observations, etc.) taken towards answering the question
    steps: List[str]

    # The true answer to the question
    answer: str
    
    # Docstore the current state is looking at
    docstore: DocstoreExplorer

    # A random number associated with the state
    randomness: int


    def __hash__(self):
        return hash((self.puzzle, self.current_state, " -> ".join(self.steps)))
    
    def items(self):
        return self.puzzle, self.current_state, self.steps, self.randomness
    
    def duplicate(self, randomness: int=None) -> "StateHotpotQA":
        """
        Returns a new instance of StateHotpotQA with an optional new randomness value.
        """
        return StateHotpotQA(
            puzzle=self.puzzle,
            current_state=self.current_state,
            thoughts=self.thoughts,
            actions=self.actions,
            observations=self.observations,
            steps=self.steps,
            answer=self.answer,
            docstore=self.docstore,
            randomness=randomness if randomness else self.randomness)