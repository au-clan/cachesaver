from os import name
import sys
import asyncio
import pytest
import tempfile
from diskcache import Cache
from cachesaver.pipelines import OnlineAPI
import itertools

from src.algorithms import AgentDictGOT, AlgorithmGOT
from src.models import API
from src.typedefs import Algorithm, DecodingParameters
from src.tasks.matharena import (
    EnvironmentMathArena,
    AgentActMathArena,
    AgentAggregateMathArena,
    AgentEvaluateMathArena,
    StateMathArena,
    BenchmarkMathArena,
)

class TestGoTMathArena:
    params = DecodingParameters(
        temperature=0.7,
        max_completion_tokens=4000,
        top_p=1.0,
        stop=None,
        logprobs=False,
    )
    env = EnvironmentMathArena()


    @pytest.mark.asyncio()
    async def test_aggregate_matharena(self, offline_model) -> None:
        state = StateMathArena(
            problem="Solve the equation 2x + 4 = 12",
            current_state="Solve the equation 2x + 4 = 12",
            steps=[],
            target="x = 4",
            randomness=1234,
        )

        mock_actions = ["Analyze[problem]", "Explain[math concepts]", "Finish[x = 4]", "Analyze[solution approach]", "Explain[solution steps]"]
        act = AgentAggregateMathArena()
        selected_actions = await act.act(
            offline_model,
            state,
            actions=mock_actions,
            k=3,
            namespace="test_small",
            request_id=0,
            params=self.params,
        )
        assert len(selected_actions) == 3
        assert all(action.startswith(("Analyze[", "Explain[", "Finish[")) for action in selected_actions)
        assert any(action == "Finish[x = 4]" for action in selected_actions)
        assert any(action in mock_actions for action in selected_actions)