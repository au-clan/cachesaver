from src.tasks.logiqa import BenchmarkLogiQA, EnvironmentLogiQA
from src.tasks.logiqa.state import StateLogiQA

import pytest

class TestLogiQAEnvironment:
    env = EnvironmentLogiQA()
    benchmark = BenchmarkLogiQA(path="datasets/dataset_logiqa.csv.gz", split="mini")

    def test_environment_step(self):
        for _, state in self.benchmark:
            action = state.option_a
            new_state = self.env.step(state, action)

            assert new_state.current_state != state.current_state
            assert new_state.current_state == 'a'

            action = state.option_b
            new_state = self.env.step(state, action)

            assert new_state.current_state != state.current_state
            assert new_state.current_state == 'b'

            action = state.option_c
            new_state = self.env.step(state, action)

            assert new_state.current_state != state.current_state
            assert new_state.current_state == 'c'

            action = state.option_d
            new_state = self.env.step(state, action)

            assert new_state.current_state != state.current_state
            assert new_state.current_state == 'd'

    def test_environment_step_int_given(self):
        _, state = self.benchmark[0]

        action = '4' # Should pick option d
        new_state = self.env.step(state, action)

        assert new_state.current_state == 'd'

    def test_environment_step_given_letter(self):
        _, state = self.benchmark[0]

        action = 'C'
        new_state = self.env.step(state, action)

        assert new_state.current_state == 'c'

    def test_environment_evaluate_correct(self):
        _, state = self.benchmark[0]

        action = state.correct_option
        new_state = self.env.step(state, action)
        print(action)
        finished, correct = self.env.evaluate(new_state)

        assert finished
        assert correct == 1.0

    def test_environment_evaluate_incorrect(self):
        _, state = self.benchmark[0]

        action = state.option_d
        new_state = self.env.step(state, action)

        finished, points = self.env.evaluate(new_state)

        assert finished
        assert points == 0.0

    def test_not_final(self):
        _, state = self.benchmark[0]

        finished, correct = self.env.evaluate(state)

        assert not finished
        assert correct == 0.0
