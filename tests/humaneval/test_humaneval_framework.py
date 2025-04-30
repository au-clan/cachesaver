from src.tasks.humaneval import BenchmarkHumanEval, EnvironmentHumanEval
from src.tasks.humaneval.state import StateHumanEval

import pytest

class TestHumanEvalSetup:
    def test_benchmark_humaneval_init(self):
        # Test initialization with different splits
        benchmark = BenchmarkHumanEval(path="datasets/humaneval-py-sorted.csv.gz", split="mini")
        assert len(benchmark) == 10

        benchmark = BenchmarkHumanEval(path="datasets/humaneval-py-sorted.csv.gz", split="train")
        assert len(benchmark) == 50

        benchmark = BenchmarkHumanEval(path="datasets/humaneval-py-sorted.csv.gz", split="validation")
        assert len(benchmark) == 50

        benchmark = BenchmarkHumanEval(path="datasets/humaneval-py-sorted.csv.gz", split="test")
        assert len(benchmark) == 50

    def test_benchmark_humaneval_invalid_split(self):
        # Test initialization with an invalid split
        with pytest.raises(ValueError) as e_info:
            BenchmarkHumanEval(path="datasets/humaneval-py-sorted.csv.gz", split="invalid_split")

    def test_benchmark_humaneval_getitem(self):
        # Test __getitem__ method
        benchmark = BenchmarkHumanEval(path="datasets/humaneval-py-sorted.csv.gz", split="mini")
        index, state = benchmark[0]
        assert state.puzzle == state.current_state
        assert isinstance(index, int)
        assert state.randomness is None


class TestHumanEvalEnvironment:
    def test_step(self):
        # Test step method
        benchmark = BenchmarkHumanEval(path="datasets/humaneval-py-sorted.csv.gz", split="mini")
        _, state = benchmark[0]
        action = "```python\nprint('Hello, World!')\n```"
        new_state = EnvironmentHumanEval.step(state, action)
        assert new_state.current_state == action
        assert new_state.steps == [action]
        assert new_state.randomness is not None

    def test_evaluate_py_failed_solve(self):
        # Test evaluate_code_python method
        benchmark = BenchmarkHumanEval(path="datasets/humaneval-py-sorted.csv.gz", split="mini")
        _, state = benchmark[0]
        solved, score = EnvironmentHumanEval.evaluate(state)
        assert not solved
        assert isinstance(score, float)

    def test_evaluate_py_solved(self):
        # Test evaluate_code_python method
        # Simulate a solved state
        solution = "def function():\n    return True\n"
        test_function = "def check(candidate):\n    assert candidate()\n" 
        new_state = StateHumanEval(
            puzzle="",
            current_state=solution,
            steps=[solution],
            entry_point="function",
            test=test_function,
            randomness=None
        )

        solved, _ = EnvironmentHumanEval.evaluate(new_state)
        assert solved

    def test_evaluate_rust_solved(self):
        # Test evaluate_code_rust method
        # Simulate a solved state
        solution = "fn function() -> i32 {\n    return 1;\n}\n"
        test_function = "fn main() {\n    let candidate = function;\n    assert_eq!(candidate(), 1);\n}\n"
        new_state = StateHumanEval(
            puzzle="",
            current_state=solution,
            steps=[solution],
            entry_point="function",
            test=test_function,
            randomness=None
        )
        solved, _ = EnvironmentHumanEval.evaluate(new_state)
        assert solved

    def test_evaluate_rust_failed_solve(self):
        # Test evaluate_code_rust method
        # Simulate a failed state
        solution = "fn function() -> i32 {\n    return 42;\n}\n"
        test_function = "fn main() {\n    let candidate = function;\n    assert_eq!(candidate(), 0);\n}\n"
        new_state = StateHumanEval(
            puzzle="",
            current_state=solution,
            steps=[solution],
            entry_point="function",
            test=test_function,
            randomness=None
        )
        solved, _ = EnvironmentHumanEval.evaluate(new_state)
        assert not solved