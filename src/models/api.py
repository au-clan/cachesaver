from ..typedefs import Model, SingleRequestModel, DecodingParameters, Request
from typing import List

class API(SingleRequestModel):
    """
    API class for the cachesaver API
    """

    def __init__(self, pipeline: Model, model: str):
        self.pipeline = pipeline
        self.model = model
        self.calls = {"total": 0, "cached": 0, "duplicated": 0}
        self.tokens = {
            "total": {"in": 0, "out": 0},
            "cached": {"in": 0, "out": 0},
            "generated": {"in": 0, "out": 0},
            "duplicated": {"in": 0, "out": 0}
        }
    
    async def request(self, prompt: str, n: int, request_id: str, namespace: str, params: DecodingParameters) -> List[str]:
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

        response = await self.pipeline.request(request)

        self.calls = {
            "total": self.calls["total"] + len(response.data),
            "cached": self.calls["cached"] + sum(response.cached),
            "duplicated": self.calls["duplicated"] + sum(response.duplicated)
        }

        messages, tokin, tokout = zip(*response.data)

        total_in = total_out = 0
        cached_in = cached_out = 0
        generated_in = generated_out = 0
        non_duplicated_in = non_duplicated_out = 0

        for in_tok, out_tok, cached, duplicated in zip(tokin, tokout, response.cached, response.duplicated):

            total_in += in_tok
            total_out += out_tok
            if cached:
                cached_in += in_tok
                cached_out += out_tok
            else:
                generated_in += in_tok
                generated_out += out_tok
            if not duplicated:
                non_duplicated_in += in_tok
                non_duplicated_out += out_tok
            else:
                non_duplicated_out += out_tok

        self.tokens["total"]["in"] += total_in
        self.tokens["total"]["out"] += total_out
        self.tokens["cached"]["in"] += cached_in
        self.tokens["cached"]["out"] += cached_out
        self.tokens["generated"]["in"] += generated_in
        self.tokens["generated"]["out"] += generated_out
        self.tokens["duplicated"]["in"] += non_duplicated_in
        self.tokens["duplicated"]["out"] += non_duplicated_out

        return messages

        
    
    