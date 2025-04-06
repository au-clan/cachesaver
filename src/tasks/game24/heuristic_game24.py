from .state_game24 import StateGame24
from ...typedefs import Heuristic

class HeuristicGame24(Heuristic):
    def __init__(self):
        super().__init__()
        self.name = "Game24 Heuristic"
    
    @staticmethod
    def get_current_numbers(state: StateGame24) -> str:
        """
        Returns the current numbers in the state.
        """
        last_line = state.current_state.strip().split('\n')[-1]
        return last_line.split('left: ')[-1].split(')')[0]
    
    @staticmethod
    def get_formula(state: StateGame24) -> str:
        formula = state.steps[-1].lower().replace("answer: ", "")
        return formula