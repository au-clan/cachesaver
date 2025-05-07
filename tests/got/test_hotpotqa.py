from langchain import Wikipedia
from langchain.agents.react.base import DocstoreExplorer

import pytest
import tempfile
from diskcache import Cache
from cachesaver.pipelines import OnlineAPI

from src.algorithms import AgentDictGOT, AlgorithmGOT
from src.models import API
from src.typedefs import DecodingParameters
from src.tasks.hotpotqa import (
    EnvironmentHotpotQA,
    AgentBfsHotpotQA,
    AgentAggregateHotpotQA,
    AgentEvaluateHotpotQA,
    StateHotpotQA,
)


class TestGoTHotpotQA:
    llm = "llama-3.3-70b-versatile"
    params = DecodingParameters(
        temperature=0.7,
        max_completion_tokens=100,
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
    async def test_aggregate_hotpotqa(self, offline_model) -> None:
        state = StateHotpotQA(
            puzzle="What is the capital of France?",
            current_state="",
            steps=[],
            answer="Paris",
            docstore=DocstoreExplorer(Wikipedia()),
            randomness=0,
        )
        generate = AgentBfsHotpotQA()
        aggregate = AgentAggregateHotpotQA()

        generate_results = await generate.act(
            model=offline_model,
            state=state,
            namespace="test_small",
            request_id=0,
            params=self.params,
        )
        aggregate_actions = await aggregate.act(
            model=offline_model,
            state=state,
            actions=generate_results,
            k=3,
            namespace="test_small",
            request_id=0,
            params=self.params,
        )

        assert len(aggregate_actions) == 3

        aggregate_actions = await aggregate.act(
            model=offline_model,
            state=state,
            actions=generate_results,
            k=2,
            namespace="test_small",
            request_id=1,
            params=self.params,
        )

        assert len(aggregate_actions) == 2
        assert all(agg_action in generate_results for agg_action in aggregate_actions)

    @pytest.mark.asyncio()
    async def test_got_hotpotqa(self, offline_model) -> None:

        # Setup
        state = StateHotpotQA(
            puzzle="What is the capital of France?",
            current_state="",
            steps=[],
            answer="Paris",
            docstore=DocstoreExplorer(Wikipedia()),
            randomness=0,
        )
        env = EnvironmentHotpotQA()
        agents = AgentDictGOT(
            step=AgentBfsHotpotQA,
            aggregate=AgentAggregateHotpotQA,
            evaluate=AgentEvaluateHotpotQA,
            step_params=self.params,
            aggregate_params=self.params,
            eval_params=self.params,
        )
        algorithm = AlgorithmGOT(
            model=offline_model,
            agents=agents,
            env=env,
            num_selections=3,
            num_steps=5,
            num_best=2,
            num_evaluations=1,
        )

        # Run the algorithm
        result = await algorithm.solve(
            idx=0,
            state=state,
            namespace="test_small",
            value_cache=None,
        )

        # Check the result
        finished, _ = env.evaluate(result[0])
        assert finished

    @pytest.mark.asyncio(loop_scope="function")
    async def test_got_hotpotqa_with_cachesaver(self, cache, online_model) -> None:

        # Setup
        state = StateHotpotQA(
            puzzle="What is the capital of France?",
            current_state="",
            steps=[],
            answer="Paris",
            docstore=DocstoreExplorer(Wikipedia()),
            randomness=0,
        )
        env = EnvironmentHotpotQA()
        agents = AgentDictGOT(
            step=AgentBfsHotpotQA,
            aggregate=AgentAggregateHotpotQA,
            evaluate=AgentEvaluateHotpotQA,
            step_params=self.params,
            aggregate_params=self.params,
            eval_params=self.params,
        )

        # Run the algorithm
        async with OnlineAPI(
            model=online_model,
            cache=cache,
            batch_size=1,
            timeout=0.1,
        ) as pipeline:
            api = API(
                pipeline=pipeline,
                model=self.llm,
            )

            algorithm = AlgorithmGOT(
                model=api,
                agents=agents,
                env=env,
                num_selections=3,
                num_steps=5,
                num_best=2,
                num_evaluations=1,  # Testing with Groq, so only 1 evaluation as groq does not support n>1
            )

            result = await algorithm.solve(
                idx=0,
                state=state,
                namespace="test_small",
                value_cache={},
            )

            # Check the result
            finished, _ = env.evaluate(result[0])
            assert finished
