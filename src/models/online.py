import asyncio
from typing import List, Any

from cachesaver.typedefs import Request, Batch, Response

from ..typedefs import Model


class OnlineLLM(Model):
    def __init__(self, client: Any):
        super().__init__()
        self.client = client

    async def request(self, request: Request) -> Response:
        sleep = 1
        while True:
            try:
                completion = await self.client.chat.completions.create(
                    
                    messages = [
                        {
                            "role" : "user",
                            "content" : request.prompt
                        }
                    ] if isinstance(request.prompt, str) else request.prompt,
                    model = request.model,
                    n = request.n,
                    max_tokens= request.max_completion_tokens or None, # or None not needed but just to be explicit
                    temperature = request.temperature or 1,
                    stop = request.stop or None,
                    top_p = request.top_p or 1,
                    seed = request.seed or None,
                    logprobs = request.logprobs or False,
                    top_logprobs = request.top_logprobs or None,
                )
                break
            except Exception as e:
                print(f"Error: {e}")
                await asyncio.sleep(max(sleep, 90))
                sleep *= 2
        input_tokens = completion.usage.prompt_tokens
        completion_tokens = completion.usage.completion_tokens
        response = Response(
            data = [(choice.message.content, input_tokens, completion_tokens/request.n) for choice in completion.choices]
        )
        return response
    
    async def batch_request(self, batch: Batch) -> List[Response]:
        requests = [self.request(request) for request in batch.requests]
        completions = await asyncio.gather(*requests)
        return completions