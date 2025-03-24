import random
from typing import List, Any
from dataclasses import dataclass
from omegaconf.dictconfig import DictConfig

from cachesaver.typedefs import BatchRequestModel
from cachesaver.typedefs import Request, Response

from .agent_basic import AgentBasic
from ..tasks.basic import StateBasic, EnvironmentBasic

@dataclass(frozen=True)
class Request(Request):
    max_completion_tokens: int
    temperature: float
    top_p: float
    stop: str

class AgentLLM(AgentBasic):
    def __init__(self, api: BatchRequestModel):
        self.name = "LLM Agent"
        self.api = api
        self.calls = {"total": 0, "cached": 0, "duplicated": 0}
        self.tokens = {
            "total": {"in": 0, "out": 0},
            "cached": {"in": 0, "out": 0},
            "generated": {"in": 0, "out": 0}
        }

    async def request(self, prompt: str, n: int, request_id:str, namespace: str, config: DictConfig) -> List[Any]:
        """
        Makes a request to the api and tracks the number of calls.
        """
        request = Request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            max_completion_tokens=config.max_completion_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            stop=config.stop
        )
        response = await self.api.request(request)
        self.calls = {
            "total": self.calls["total"] + len(response.data),
            "cached": self.calls["cached"] + sum(response.cached),
            "duplicated": self.calls["duplicated"] + sum(response.duplicated)
        }
        
        messages, tokin, tokout = zip(*response.data)
        cached_tokin = [int(tokens * cached) for tokens, cached in zip(tokin, response.cached)]
        cached_tokout = [int(tokens * cached) for tokens, cached in zip(tokout, response.cached)]
        generated_tokin = [int(tokens * (not cached)) for tokens, cached in zip(tokin, response.cached)]
        generated_tokout = [int(tokens * (not cached)) for tokens, cached in zip(tokout, response.cached)]

        self.tokens["total"]["in"] += sum(tokin)
        self.tokens["total"]["out"] += sum(tokout)
        self.tokens["cached"]["in"] += sum(cached_tokin)
        self.tokens["cached"]["out"] += sum(cached_tokout)
        self.tokens["generated"]["in"] += sum(generated_tokin)
        self.tokens["generated"]["out"] += sum(generated_tokout)
        return messages

    async def act_step(self, state: StateBasic, environment: EnvironmentBasic, namespace: str, request_id: str, config: DictConfig, cache: dict=None) -> StateBasic:
        """
        Returns the next state after performing 1 action.
        """
        if cache is not None and state in cache:
            inference = cache[state]
        else:
            prompt = environment.prompter.act(state)
            response = await self.request(
                prompt=prompt,
                n=1,
                request_id=request_id,
                namespace=namespace,
                config=config
            )
            inference = environment.parser.act(response[0])

            if cache is not None:
                cache[state] = inference
        
        random.seed(state.randomness)
        randomness = random.randint(0, 1000)
        new_state = environment.get_next_state(
            inference=inference,
            state=state,
            randomness=randomness
        )
        return new_state
        
    
    async def react_step(self, state: StateBasic, environment: EnvironmentBasic, namespace: str, request_id: str, config: DictConfig, cache: dict=None) -> StateBasic:
        """
        Returns the next state after performing 1 thought and 1 action.
        """
        if cache is not None and state in cache:
            inference = cache[state]
        else:
            prompt = environment.prompter.react(state)
            response = await self.request(
                prompt=prompt,
                n=1,
                request_id=request_id,
                namespace=namespace,
                config=config
            )
            inference = environment.parser.react(response[0])

            if cache is not None:
                cache[state] = inference
        
        random.seed(state.randomness)
        randomness = random.randint(0, 1000)
        new_state = environment.get_next_state(
            inference=inference,
            state=state,
            randomness=randomness
        )
        return new_state
    
    async def foa_step(self, state: StateBasic, environment: EnvironmentBasic, namespace: str, request_id: str, config: DictConfig, cache: dict=None) -> StateBasic:
        """
        Returns a list of proposals for the given state.
        """
        if cache is not None and state in cache:
            inferences = cache[state]
        else:
            prompt = environment.prompter.bfs(state)
            response = await self.request(
                prompt=prompt,
                n=1,
                request_id=request_id,
                namespace=namespace,
                config=config
            )
            inferences = environment.parser.bfs(response[0])

            if cache is not None:
                cache[state] = inferences

        random.seed(state.randomness)
        select_inference = random.choice(inferences)
        randomness = random.randint(0, 1000)
        new_state = environment.get_next_state(
            inference=select_inference,
            state=state,
            randomness=randomness
        )
        return new_state
    
    async def tot_step(self, state: StateBasic, environment: EnvironmentBasic, namespace: str, request_id: str, config: DictConfig, cache: dict=None) -> List[StateBasic]:
        """
        Returns a list of proposals for the given state.
        """
        if cache is not None and state in cache:
            inferences = cache[state]
        else:
            prompt = environment.prompter.bfs(state)
            response = await self.request(
                prompt=prompt,
                n=1,
                request_id=request_id,
                namespace=namespace,
                config=config
            )
            inferences = environment.parser.bfs(response[0])

            if cache is not None:
                cache[state] = inferences
        randies = [random.randint(0, 1000) for _ in range(len(inferences))]
        new_states = [
            environment.get_next_state(
                inference=inference,
                state=state,
                randomness=randies[i]
            )
            for i, inference in enumerate(inferences)
        ]
        return new_states
    
    async def evaluate(self, state: StateBasic, environment: EnvironmentBasic, n: int, namespace: str, request_id: str, config: DictConfig, cache: dict=None) -> int:
        """
        Some tasks use exlucively deterministc methods for evaluation (humaneval), others exlucively llm-based methods (mini crosswords) and others a mixture of the two (game24).

        For this, even in this LLM-based agent, we employ a 2 step methods. First we employ the deterministic method, if it fails to find a value, we employ the llm-based method.
        """
        
        if cache is not None and state in cache:
            inference = cache[state]
        
        # Deterministic method
        inference = environment.get_value(state)
        
        # LLM-based method
        if inference.value is None:
            prompt = environment.prompter.evaluate(state)
            response = await self.request(
                prompt=prompt,
                n=n,
                request_id=request_id,
                namespace=namespace,
                config=config
            )
            inference = environment.parser.evaluate(response)
        
        if cache is not None:
            cache[state] = inference
        
        value = inference.value
        return value