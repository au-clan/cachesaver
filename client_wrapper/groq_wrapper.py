import asyncio
import re
import time
from typing import List

from cachesaver.typedefs import Request as BaseRequest, Batch
from cachesaver.typedefs import Response
from groq import AsyncGroq, RateLimitError

from src.algorithm_options.rafa import RafaRequest
from src.typedefs import Model as ModelBasic


class GroqModel(ModelBasic):
    def __init__(self, api_key: str, model: str):
        super().__init__()
        self.client = AsyncGroq(api_key=api_key)
        self.model = model

        self.model_limits = {
            "deepseek-r1-distill-llama-70b": (30, 1000, 6000, None),
            "deepseek-r1-distill-qwen-32b": (30, 1000, 6000, None),
            "distil-whisper-large-v3-en": (20, 2000, None, None),
            "gemma2-9b-it": (30, 14400, 15000, 500000),
            "llama-3.1-8b-instant": (30, 14400, 6000, 500000),
            "llama-3.2-1b-preview": (30, 7000, 7000, 500000),
            "llama-3.2-3b-preview": (30, 7000, 7000, 500000),
            "llama-3.2-11b-vision-preview": (30, 7000, 7000, 500000),
            "llama-3.2-90b-vision-preview": (15, 3500, 7000, 250000),
            "llama-3.3-70b-specdec": (30, 1000, 6000, 100000),
            "llama-3.3-70b-versatile": (30, 1000, 6000, 100000),
            "llama-guard-3-8b": (30, 14400, 15000, 500000),
            "llama3-8b-8192": (30, 14400, 6000, 500000),
            "llama3-70b-8192": (30, 14400, 6000, 500000),
            "mistral-saba-24b": (30, 1000, 6000, None),
            "qwen-2.5-32b": (30, 1000, 6000, None),
            "qwen-2.5-coder-32b": (30, 1000, 6000, None),
            "qwen-qwq-32b": (30, 1000, 6000, None),
            "whisper-large-v3": (20, 2000, None, None),
            "whisper-large-v3-turbo": (20, 2000, None, None),
        }

        # Get model limits or set defaults
        self.rpm, self.rpd, self.tpm, self.tpd = self.model_limits.get(model, (30, 1000, 6000, None))

        # Initialize rate counters
        self.rpm_remaining = self.rpm
        self.rpd_remaining = self.rpd
        self.tpm_remaining = self.tpm
        self.tpd_remaining = self.tpd

        # Reset times
        self.rpm_reset_time = time.time() + 60  # 1 minute
        self.rpd_reset_time = time.time() + 86400  # 24 hours
        self.tpm_reset_time = time.time() + 60  # 1 minute
        self.tpd_reset_time = time.time() + 86400  # 24 hours

        # todo locks, could also be semaphore
        self.rate_limit_lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(20)  # todo the same as the batch sizes or the number of requests pr minute

    async def batch_request(self, batch: Batch) -> List[Response]:
        """Handles a batch of requests"""

        responses = await asyncio.gather(*(self.request(req) for req in batch.requests))

        return responses

    async def request(self, request: RafaRequest) -> Response:

        responses = await asyncio.gather(*(self.single_request(request.prompt) for _ in range(request.n)))
        # todo format here, request count, input tokens and output tokens math should go here
        merged_data = [item for r in responses for item in r.data]
        merged_response = Response(
            data=merged_data
        )
        return merged_response

    async def single_request(self, request: RafaRequest) -> Response:
        await self.obey_rate_limits()
        async with self.rate_limit_lock:
            if self.rpm_remaining > 0:
                self.rpm_remaining -= 1
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] Making request")

        async with self.semaphore:
            try:
                completions = await self.client.with_raw_response.chat.completions.create(
                    messages=request.messages,
                    model=self.model,
                    n=1,
                    max_tokens=request.max_completion_tokens or None,  # or None not needed but just to be explicit
                    temperature=request.temperature or 1,
                    stop=request.stop_token or None,
                    top_p=request.top_p or 1,
                    seed=request.seed or None,
                    logprobs=request.logprobs or False,
                    top_logprobs=request.top_logprobs or None,
                )

                async with self.rate_limit_lock:

                    # todo update other fields if longer runs needed
                    self.rpd_remaining = int(completions.headers.get("x-ratelimit-remaining-requests"))
                    self.tpm_remaining = int(completions.headers.get("x-ratelimit-remaining-tokens"))

                    self.rpd_reset_time = time.time() + self.parse_reset_time(
                        (completions.headers.get("x-ratelimit-reset-requests", "86400s")))
                    self.tpm_reset_time = time.time() + self.parse_reset_time(
                        (completions.headers.get("x-ratelimit-reset-tokens", "60s")))

                retry_after = (completions.headers.get("retry-after"))

                # Extract reset times
                if retry_after:
                    await asyncio.sleep(float(retry_after))

                # Parse completions JSON pr documentation
                data = await completions.parse()
                input_tokens = data.usage.prompt_tokens
                completion_tokens = data.usage.completion_tokens
                response = Response(
                    data=[(data.choices[0].message.content, input_tokens, completion_tokens)]
                )

                return response
            except RateLimitError as e:
                match = re.search(r"Please try again in (\d+(\.\d+)?)s", str(e))
                retry_after = float(match.group(1)) if match else 5  # Default to 5s if missing

                print(f"Sleeping for {retry_after:.2f} seconds before retrying...")
                await asyncio.sleep(retry_after)
                return await self.single_request(request)

    async def obey_rate_limits(self):
        """Check and wait if rate limits are exceeded before allowing a request."""
        while True:
            async with self.rate_limit_lock:
                current_time = time.time()
                sleep_time = 0
                now = time.strftime("%Y-%m-%d %H:%M:%S")

                # Reset counters if necessary
                if current_time >= self.rpm_reset_time:
                    print(f"[{now}] Resetting RPM counter...")
                    self.rpm_remaining = self.rpm
                    self.rpm_reset_time = current_time + 60

                if current_time >= self.rpd_reset_time:
                    print("[{now}] Resetting RPD counter...")
                    self.rpd_remaining = self.rpd
                    self.rpd_reset_time = current_time + 86400

                if current_time >= self.tpm_reset_time:
                    print("[{now}] Resetting TPM counter...")
                    self.tpm_remaining = self.tpm
                    self.tpm_reset_time = current_time + 60

                if current_time >= self.tpd_reset_time:
                    print("[{now}] Resetting TPD counter...")
                    self.tpd_remaining = self.tpd
                    self.tpd_reset_time = current_time + 86400

                # Check if limits are exceeded
                if self.rpm_remaining <= 0:
                    sleep_time = max(sleep_time, self.rpm_reset_time - current_time)
                    print(f"[{now}] RPM Limit reached! Sleeping for {sleep_time:.2f} seconds.")

                if self.rpd_remaining <= 0:
                    sleep_time = max(sleep_time, self.rpd_reset_time - current_time)
                    print(f"[{now}] RPD Limit reached! Sleeping for {sleep_time:.2f} seconds.")

                if self.tpm is not None and self.tpm_remaining <= 0:
                    sleep_time = max(sleep_time, self.tpm_reset_time - current_time)
                    print(f"[{now}] TPM Limit reached! Sleeping for {sleep_time:.2f} seconds.")

                if self.tpd is not None and self.tpd_remaining <= 0:
                    sleep_time = max(sleep_time, self.tpd_reset_time - current_time)
                    print(f"[{now}] TPD Limit reached! Sleeping for {sleep_time:.2f} seconds.")

                if sleep_time == 0:
                    return

            await asyncio.sleep(sleep_time + 3)

    @staticmethod
    def parse_reset_time(reset_value: str) -> float:
        if "ms" in reset_value:
            milliseconds = int(re.sub(r"\D", "", reset_value))  # Extract numbers
            return max(1.0, milliseconds / 1000)

        """Convert reset time (like '2m59.56s' or '7.66s') to seconds"""
        if "m" in reset_value:
            minutes, seconds = reset_value.split("m")
            return float(minutes) * 60 + float(seconds.replace("s", ""))
        return float(reset_value.replace("s", ""))
