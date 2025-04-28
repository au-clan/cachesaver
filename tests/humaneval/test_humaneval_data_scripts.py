from src.tasks.humaneval import BenchmarkHumanEval

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
        assert index == benchmark.data[0][0]
        assert state.puzzle == benchmark.data[0][1]
        assert state.current_state == benchmark.data[0][1]
        assert state.canonical_solution == benchmark.data[0][2]
        assert state.entry_point == benchmark.data[0][3]
        assert state.test == benchmark.data[0][4]
        assert state.steps == []
        assert state.randomness is None