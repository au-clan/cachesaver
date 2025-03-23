import asyncio    
from typing import List, Any
from dataclasses import dataclass

from cachesaver.typedefs import Request, Batch, Response, SingleRequestModel, BatchRequestModel

class OnlineLLM(SingleRequestModel, BatchRequestModel):
    def __init__(self, client: Any, model: str):
        self.client = client
        self.model = model

    async def request(self, request: Request) -> Response:
        completion = await self.client.chat.completions.create(
            messages = [
                {
                    "role" : "user",
                    "content" : request.prompt
                }
            ],
            model = self.model,
            n = request.n,
            max_tokens= request.max_completion_tokens or None, # or None not needed but just to be explicit
            temperature = request.temperature or 1,
            stop = request.stop or None,
            top_p = request.top_p or 1,
            seed = request.seed or None,
            logprobs = request.logprobs or False,
            top_logprobs = request.top_logprobs or None,
        )
        response = Response(
            data = [choice.message.content for choice in completion.choices]
        )
        return response
    
    async def batch_request(self, batch: Batch) -> List[Response]:
        requests = [self.request(request) for request in batch.requests]
        completions = await asyncio.gather(*requests)
        return completions