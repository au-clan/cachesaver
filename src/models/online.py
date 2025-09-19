import asyncio    
from typing import List, Any
from dataclasses import dataclass

from cachesaver.typedefs import Request, Batch, Response

from ..typedefs import Model

class OnlineLLM(Model):
    def __init__(self, provider: str, max_n: int = 128):
        self.client = client_init(provider)
        self.max_n = max_n

    async def request(self, request: Request) -> Response:
        total_n = request.n
        results = []
        input_tokens = 0
        completion_tokens = 0
        sleep = 1

        prompts = (
            [{"role": "user", "content": request.prompt}]
            if isinstance(request.prompt, str)
            else request.prompt
        )

        while total_n > 0:
            current_n = min(total_n, self.max_n)
            total_n -= current_n

            while True:
                try:
                    completion = await self.client.chat.completions.create(
                        messages=prompts,
                        model=request.model,
                        n=current_n,
                        max_completion_tokens=request.max_completion_tokens or None,
                        temperature=request.temperature or 1,
                        stop=request.stop or None,
                        top_p=request.top_p or 1,
                        seed=request.seed or None,
                        logprobs=request.logprobs or False,
                        top_logprobs=request.top_logprobs or None,
                    )
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    await asyncio.sleep(max(sleep, 90))
                    sleep *= 2

            input_tokens = completion.usage.prompt_tokens
            completion_tokens = completion.usage.completion_tokens
            if getattr(completion.usage, 'prompt_tokens_details', None):
                try:
                    cached_tokens = completion.usage.prompt_tokens_details.cached_tokens
                    
                except Exception as e:
                    print(f"Could not access cached tokens: {e}")
                    pass
            else:
                cached_tokens = 0
            
            print(f"Input tokens: {input_tokens}, Completion tokens: {completion_tokens}, Cached tokens: {cached_tokens}")
            

            results.extend(
                (choice.message.content, input_tokens, completion_tokens / current_n, cached_tokens)
                for choice in completion.choices
            )

        return Response(data=results)

    
    async def batch_request(self, batch: Batch) -> List[Response]:
        requests = [self.request(request) for request in batch.requests]
        completions = await asyncio.gather(*requests)
        return completions
    
def client_init(provider: str) -> Any:
    if provider == "openai":
        from openai import AsyncOpenAI
        return AsyncOpenAI()
    elif provider == "together":
        from together import AsyncTogether
        return AsyncTogether()
    else:
        raise ValueError(f"Unknown provider: {provider}")