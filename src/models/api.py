import time
from abc import ABC
from deepdiff import DeepHash
from collections import defaultdict

from ..typedefs import Model, SingleRequestModel, DecodingParameters, Request
from typing import List, Union

class API(ABC):
    """
    API class for the cachesaver API
    """

    def __init__(self, pipeline: Model, model: str):
        self.pipeline = pipeline
        self.model = model
        self.calls = defaultdict(lambda: {
            "total": 0,       # Total calls
            "cacher": 0,      # Calls saved by the cacher
            "deduplicator": 0 # Calls saved by the deduplicator
            })
        
        self.tokens = defaultdict(lambda: {
            "total": {"in": 0, "out": 0},      # Total tokens
            "cacher": {"in": 0, "out": 0},     # Tokens saved by the cacher
            "duplicator": {"in": 0, "out": 0}, # Tokens saved by the deduplicator
        })

        self.latencies = defaultdict(list)
        self.reuse = defaultdict(dict)
    
    async def request(self, prompt: Union[str, List[str]], n: int, request_id: str, namespace: str, params: DecodingParameters, tab: str="default") -> List[str]:
        """
        Send a request to the pipeline
        """
        request = Request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            model=self.model,
            max_completion_tokens=params.max_completion_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            stop=params.stop,
            logprobs=params.logprobs
        )

        start = time.perf_counter()
        response = await self.pipeline.request(request)
        end = time.perf_counter()

        # Measuring latency
        self.latencies[tab].append(end - start)

        # Measuring reuse
        hashed_prompt = DeepHash(prompt)[prompt]
        if hashed_prompt in self.reuse:
            self[tab].reuse[hashed_prompt] += n
        else:
            self[tab].reuse[hashed_prompt] = n

        # Measuring number of caslls
        self[tab].calls["total"] += len(response.data)
        self[tab].calls["cacher"] += sum(response.cached)
        self[tab].calls["deduplicator"] += sum(response.duplicated)

        # Measuring number of tokens
        messages, tokin, tokout = zip(*response.data)
        total_in = total_out = 0
        cached_in = cached_out = 0
        duplicated_in = 0 # deduplicator saves only on input

        for in_tok, out_tok, cached, duplicated in zip(tokin, tokout, response.cached, response.duplicated):

            total_in += in_tok
            total_out += out_tok
            
            # Amount of tokens saved by the cacher
            if cached:
                cached_in += in_tok
                cached_out += out_tok
            
            # Amount of tokens saved by the duplicator
            if duplicated and not cached:
                duplicated_in += in_tok


        self[tab].tokens["total"]["in"] += total_in
        self[tab].tokens["total"]["out"] += total_out
        self[tab].tokens["cacher"]["in"] += cached_in
        self[tab].tokens["cacher"]["out"] += cached_out
        self[tab].tokens["duplicator"]["in"] += duplicated_in

        return messages

        
    
    