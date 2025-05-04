from abc import ABC
from typing import List, Union

from ..typedefs import Model, DecodingParameters, Request


class API(ABC):
    """
    API class for the cachesaver API
    """

    def __init__(self, pipeline: Model, model: str):
        self.pipeline = pipeline
        self.model = model
        self.calls = {"total": 0, "cached": 0, "duplicated": 0}
        self.tokens = {
            "total": {"in": 0, "out": 0},
            "cacher": {"in": 0, "out": 0},
            "duplicator": {"in": 0, "out": 0},
            "cacher_duplicator": {"in": 0, "out": 0},
        }

    async def request(self, prompt: Union[str, List[str]], n: int, request_id: str, namespace: str,
                      params: DecodingParameters) -> List[str]:
        """
        Send a request to the pipeline
        """
        request = Request(
            prompt=prompt,
            model=self.model,
            n=n,
            request_id=request_id,
            namespace=namespace,
            max_completion_tokens=params.max_completion_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            stop=params.stop,
            logprobs=params.logprobs
        )

        response = await self.pipeline.request(request)

        self.calls = {
            "total": self.calls["total"] + len(response.data),
            "cached": self.calls["cached"] + sum(response.cached),
            "duplicated": self.calls["duplicated"] + sum(response.duplicated)
        }

        messages, tokin, tokout = zip(*response.data)

        total_in = total_out = 0
        cached_in = cached_out = 0
        non_duplicated_in = non_duplicated_out = 0
        cacher_duplicator_in = cacher_duplicator_out = 0

        for in_tok, out_tok, cached, duplicated in zip(tokin, tokout, response.cached, response.duplicated):

            total_in += in_tok
            total_out += out_tok
            if cached:
                cached_in += in_tok
                cached_out += out_tok

            if not duplicated:
                non_duplicated_in += in_tok
                non_duplicated_out += out_tok
            else:
                non_duplicated_out += out_tok

            if cached and not duplicated:
                cacher_duplicator_in += in_tok
                cacher_duplicator_out += out_tok
            elif cached and duplicated:
                cacher_duplicator_out += out_tok

        self.tokens["total"]["in"] += total_in
        self.tokens["total"]["out"] += total_out
        self.tokens["cacher"]["in"] += cached_in
        self.tokens["cacher"]["out"] += cached_out
        self.tokens["duplicator"]["in"] += non_duplicated_in
        self.tokens["duplicator"]["out"] += non_duplicated_out
        self.tokens["cacher_duplicator"]["in"] += cacher_duplicator_in
        self.tokens["cacher_duplicator"]["out"] += cacher_duplicator_out

        return messages
