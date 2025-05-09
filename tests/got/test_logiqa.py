import pytest
import tempfile
import asyncio
import itertools
from diskcache import Cache
from cachesaver.pipelines import OnlineAPI

from src.algorithms import AgentDictGOT, AlgorithmGOT
from src.models import API
from src.typedefs import DecodingParameters
from src.tasks.logiqa import (
    EnvironmentLogiQA,
    AgentActLogiQA,
    AgentAggregateLogiQA,
    AgentEvaluateLogiQA,
    StateLogiQA,
    BenchmarkLogiQA,
)


class TestGoTLogiQA:
    params = DecodingParameters(
        temperature=0.7,
        max_completion_tokens=100,
        top_p=1.0,
        stop=None,
        logprobs=False,
    )
    benchmark = BenchmarkLogiQA("datasets/dataset_logiqa.csv.gz", "mini")

    @pytest.mark.asyncio()
    async def test_act_agent(self, offline_model_openai) -> None:
        _, state = self.benchmark[0]

        act = AgentActLogiQA()

        actions = await act.act(
            model=offline_model_openai,
            state=state,
            n=3,
            namespace="test_small",
            request_id=0,
            params=self.params,
        )

        assert len(actions) == 3
