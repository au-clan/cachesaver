import json
import os
import asyncio
from typing import List

from cachesaver.typedefs import Request, Batch, Response

from ..typedefs import Model
from groq import Groq
from lazykey import KeyHandler

from dotenv import load_dotenv

load_dotenv()
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_LLM_MODEL_NAME", None)
GROQ_API_KEY_LIST = os.getenv("GROQ_API_KEY_LIST", "") # GROQ_API_KEY_LIST='["Key1", "Key2"]'
GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)


def _extract_choice_and_usage(completion):
    if isinstance(completion, dict):
        choice = completion["choices"][0]
        usage = completion.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
    else:
        choice = completion.choices[0]
        usage = getattr(completion, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

    return choice, input_tokens, completion_tokens


class GroqAPILLM(Model):
    def __init__(self, use_multiple_keys: bool = True):
        if use_multiple_keys:
            self.client = KeyHandler(json.loads(GROQ_API_KEY_LIST),
                                     Groq)
            self.generation_function = self.client.request
        else:
            self.client = Groq(
                api_key=GROQ_API_KEY,
            )
            self.generation_function = self.client.chat.completions.create

    async def request(self, request: Request) -> Response:
        # TODO GroqAPI doesnt support n>1 so this needs to be done in a simple for loop for now
        # sleep = 1
        # while True:
        #     try:
        #         completion = self.generation_function(
        #
        #             messages=[
        #                 {
        #                     "role": "user",
        #                     "content": request.prompt
        #                 }
        #             ] if isinstance(request.prompt, str) else request.prompt,
        #             model=request.model,
        #             n=request.n,
        #             max_tokens=request.max_completion_tokens or None,  # or None not needed but just to be explicit
        #             temperature=request.temperature or 1,
        #             stop=request.stop or None,
        #             top_p=request.top_p or 1,
        #             seed=request.seed or None,
        #             logprobs=request.logprobs or False,
        #             top_logprobs=request.top_logprobs or None,
        #         )
        #         break
        #     except Exception as e:
        #         print(f"Error: {e}")
        #         await asyncio.sleep(max(sleep, 90))
        #         sleep *= 2
        # input_tokens = completion.usage.prompt_tokens
        # completion_tokens = completion.usage.completion_tokens
        # response = Response(
        #     data=[(choice.message.content, input_tokens, completion_tokens / request.n) for choice in
        #           completion.choices]
        # )
        # return response
        sleep = 1
        while True:
            try:
                completions = []
                total_input_tokens = 0
                total_completion_tokens = 0

                for _ in range(request.n):
                    completion = self.generation_function(
                        messages=[
                            {
                                "role": "user",
                                "content": request.prompt
                            }
                        ] if isinstance(request.prompt, str) else request.prompt,
                        model=request.model,
                        n=1,  # Force n=1 due to API limitation
                        max_tokens=request.max_completion_tokens or None,
                        temperature=request.temperature or 1,
                        stop=request.stop or None,
                        top_p=request.top_p or 1,
                        seed=request.seed or None,
                        logprobs=request.logprobs or False,
                        top_logprobs=request.top_logprobs or None,
                    )

                    choice, input_tokens, completion_tokens = _extract_choice_and_usage(completion)
                    completions.append(choice)
                    total_input_tokens += input_tokens
                    total_completion_tokens += completion_tokens

                break  # all successful, exit retry loop

            except Exception as e:
                print(f"Error: {e}")
                await asyncio.sleep(max(sleep, 90))
                sleep *= 2

        # Construct synthetic response
        response = Response(
            data=[
                (getattr(choice.message, "content", ""), total_input_tokens, total_completion_tokens / request.n)
                for choice in completions
            ]
        )
        return response

    async def batch_request(self, batch: Batch) -> List[Response]:
        requests = [self.request(request) for request in batch.requests]
        completions = await asyncio.gather(*requests)
        return completions
