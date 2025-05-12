from dataclasses import dataclass, field
from typing import List, Optional
from ...typedefs import State
from matharena_ethz.src.matharena.parser import parse_grading

@dataclass(frozen=True)
class StateMathArena(State):
    """State representation for MathArena task."""
    
    problem_idx: str
    problem: str
    answer: str
    parsed_problem: Optional[dict] = None
    steps: List[str] = field(default_factory=list)
    current_state: Optional[dict] = None

    def __post_init__(self):
        """Parse problem after initialization if not already parsed."""
        if self.parsed_problem is None:
            # Use object.__setattr__ since this is a frozen dataclass
            object.__setattr__(self, 'parsed_problem', parse_grading(self.problem))

    def copy(self) -> 'StateMathArena':
        """Create a deep copy of the current state."""
        return StateMathArena(
            problem_idx=self.problem_idx,
            problem=self.problem,
            answer=self.answer,
            parsed_problem=self.parsed_problem,
            steps=self.steps.copy()
        )

    @classmethod
    def from_dict(cls, data: dict) -> 'StateMathArena':
        """Create state from dictionary representation."""
        # Parse problem during initialization
        parsed_problem = parse_grading(data['problem'])
        return cls(
            problem_idx=data['problem_idx'],
            problem=data['problem'],
            answer=data['answer'],
            parsed_problem=parsed_problem
        )

    def to_dict(self) -> dict:
        """Convert state to dictionary representation."""
        return {
            'problem_idx': self.problem_idx,
            'problem': self.problem,
            'answer': self.answer,
            'parsed_problem': self.parsed_problem,
            'steps': self.steps
        }

    # Required abstract methods from State class
    def clone(self):
        """Required by State: Returns a deep copy"""
        return self.copy()

    def serialize(self):
        """Required by State: Returns serialized form"""
        return self.to_dict()

    def get_seed(self):
        """Required by State: Returns unique seed for this state"""
        return hash((self.problem_idx, self.problem, str(self.parsed_problem)))