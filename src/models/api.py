import os
import time
from abc import ABC
from deepdiff import DeepHash
from collections import defaultdict

import logging
logger = logging.getLogger(__name__)

from ..typedefs import Model, SingleRequestModel, DecodingParameters, Request
from ..utils import api_logging
from typing import List, Union 

class API(ABC):
    """
    API class for the cachesaver API
    """

    def __init__(self, pipeline: Model, model: str, log_path: str = None):
        self.pipeline = pipeline
        self.model = model

        self.tabs = set()
        self.calls = defaultdict(lambda: {
            "total": 0,       # Total calls
            "cacher": 0,      # Calls saved by the cacher
            "deduplicator": 0 # Calls saved by the deduplicator
            })
        
        self.tokens = defaultdict(lambda: {
            "total": {"in": 0, "out": 0, "cached": 0},      # Total tokens
            "cacher": {"in": 0, "out": 0},     # Tokens saved by the cacher
            "duplicator": {"in": 0, "out": 0}, # Tokens saved by the deduplicator
        })

        self.latencies = defaultdict(list)
        self.reuse = defaultdict(lambda: defaultdict(int))

        self.request_prompts = []

        self.log_path = log_path
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            f_handler = logging.FileHandler(self.log_path, mode="w", encoding="utf-8")
            f_handler.setLevel(logging.INFO)
            logger.addHandler(f_handler)
            logger.setLevel(logging.INFO)

    def _set_log_path(self, log_path: str):
        # Remove + close existing file handlers
        for h in logger.handlers[:]:
            if isinstance(h, logging.FileHandler):
                logger.removeHandler(h)
                h.close()

        # Ensure new directory exists
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # Create and add new handler
        f_handler = logging.FileHandler(self.log_path, mode="w", encoding="utf-8")
        f_handler.setLevel(logging.INFO)
        logger.addHandler(f_handler)
        logger.addHandler(f_handler)

        # Update log path
        self.log_path = log_path
    
    def update_log_path(self, log_path: str):
        """
        Update the log path for logging raw API calls
        """
        self._set_log_path(log_path)

    
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
        self.request_prompts.append(prompt)

        start = time.perf_counter()
        response = await self.pipeline.request(request)
        end = time.perf_counter()

        # Update tabs
        self.tabs.add(tab)
        
        # Measuring latency
        self.latencies[tab].append(end - start)

        # Measuring reuse
        hashed_prompt = DeepHash(prompt)[prompt]
        self.reuse[tab][hashed_prompt] += n

        # Measuring number of calls
        self.calls[tab]["total"] += len(response.data)
        self.calls[tab]["cacher"] += sum(response.cached)
        self.calls[tab]["deduplicator"] += sum(response.duplicated)

        # Measuring number of tokens
        normalized = [(*row, 0) if len(row) == 3 else row for row in response.data]
        messages, tokin, tokout, tokcached = zip(*normalized)
        total_in = total_out = total_cached = 0
        cached_in = cached_out = 0
        duplicated_in = 0 # deduplicator saves only on input

        for in_tok, out_tok, cached_tok, cached, duplicated in zip(tokin, tokout, tokcached, response.cached, response.duplicated):

            total_in += in_tok
            total_out += out_tok
            
            # Amount of tokens saved by the cacher
            if cached:
                cached_in += in_tok
                cached_out += out_tok
            
            # Amount of tokens saved by the duplicator
            if duplicated and not cached:
                duplicated_in += in_tok

            if not duplicated and not cached:
                total_cached += cached_tok

        self.tokens[tab]["total"]["in"] += total_in
        self.tokens[tab]["total"]["out"] += total_out
        self.tokens[tab]["total"]["cached"] += total_cached
        self.tokens[tab]["cacher"]["in"] += cached_in
        self.tokens[tab]["cacher"]["out"] += cached_out
        self.tokens[tab]["duplicator"]["in"] += duplicated_in

        # Logging
        if self.log_path:
            api_logging(
                logger=logger,
                prompt=prompt,
                n=n,
                response=messages,
            )


        return messages
    
    def clean(self):

        self.tabs = set()
        self.calls = defaultdict(lambda: {
            "total": 0,       # Total calls
            "cacher": 0,      # Calls saved by the cacher
            "deduplicator": 0 # Calls saved by the deduplicator
            })
        
        self.tokens = defaultdict(lambda: {
            "total": {"in": 0, "out": 0, "cached": 0},      # Total tokens
            "cacher": {"in": 0, "out": 0},     # Tokens saved by the cacher
            "duplicator": {"in": 0, "out": 0}, # Tokens saved by the deduplicator
        })

        self.latencies = defaultdict(list)
        self.reuse = defaultdict(lambda: defaultdict(int))