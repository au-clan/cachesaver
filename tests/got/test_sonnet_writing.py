import sys
import asyncio
import pytest
import tempfile
from diskcache import Cache
from cachesaver.pipelines import OnlineAPI
import itertools

from src.algorithms import AgentDictGOT, AlgorithmGOT
from src.models import API
from src.typedefs import DecodingParameters
from src.tasks.sonnetwriting import (
    EnvironmentSonnetWriting,
    AgentActSonnetWriting,
    AgentAggregateSonnetWriting,
    AgentEvaluateSonnetWriting,
    StateSonnetWriting,
)


class TestGoTSonnetWriting:
    llm = "llama-3.3-70b-versatile"
    params = DecodingParameters(
        temperature=0.7,
        max_completion_tokens=4000,
        top_p=1.0,
        stop=None,
        logprobs=False,
    )

    @pytest.fixture(scope="function")
    def cache(self):
        """Provide a temporary cache for each test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with Cache(tmpdir) as cache:
                yield cache

    @pytest.mark.asyncio()
    async def test_act_sonnet_writing(self, offline_model) -> None:
        state = StateSonnetWriting(
            puzzle='Write a sonnet with strict rhyme scheme ABAB CDCD EFEF GG, containing each of the following words verbatim: "grass", "value", and "jail".',
            current_state='Write a sonnet with strict rhyme scheme ABAB CDCD EFEF GG, containing each of the following words verbatim: "grass", "value", and "jail".',
            steps=[],
            target="ABAB CDCD EFEF GG, grass value jail",
            randomness=1234,
        )

        act = AgentActSonnetWriting()
        sonnet = await act.act(
            offline_model,
            state,
            n=1,
            namespace="test_small",
            request_id=0,
            params=self.params,
        )
        assert len(sonnet[0].split("\n")) == len(state.target.split(", ")[0])

    @pytest.mark.asyncio()
    async def test_aggregate_sonnet_writing(self, offline_model) -> None:
        state = StateSonnetWriting(
            puzzle='Write a sonnet with strict rhyme scheme ABAB CDCD EFEF GG, containing each of the following words verbatim: "grass", "value", and "jail".',
            current_state='Write a sonnet with strict rhyme scheme ABAB CDCD EFEF GG, containing each of the following words verbatim: "grass", "value", and "jail".',
            steps=[],
            target="ABAB CDCD EFEF GG, grass value jail",
            randomness=1234,
        )

        act = AgentActSonnetWriting()
        act_coroutine = [
            act.act(
                offline_model,
                state,
                n=1,
                namespace="test_small",
                request_id=0,
                params=self.params,
            )
            for _ in range(5)
        ]
        act_actions = await asyncio.gather(*act_coroutine)
        act_actions = list(itertools.chain(*act_actions))
        assert all(len(action.split('\n')) == len(state.target.split(', ')[0]) for action in act_actions)

        aggregate = AgentAggregateSonnetWriting()
        actions = await aggregate.act(
            model=offline_model,
            state=state,
            actions=act_actions,
            k=3,
            namespace="test_small",
            request_id=0,
            params=self.params,
        )

        assert len(actions) == 3
        assert all(agg_action in act_actions for agg_action in actions)

    @pytest.mark.asyncio()
    async def test_evaluate_sonnet_writing(self, offline_model) -> None:
        pass