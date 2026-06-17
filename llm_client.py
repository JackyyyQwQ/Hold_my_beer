from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, TypeVar

from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from schema import ValidationError


T = TypeVar("T")


class LLMClient:
    """Small wrapper around the OpenAI SDK with JSON parsing and retries."""

    def __init__(
        self,
        model: str | None = None,
        max_retries: int = 3,
        retry_delay_seconds: float = 2.0,
        timeout_seconds: float = 30.0,
    ) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment.")

        self.client = OpenAI(api_key=api_key, timeout=timeout_seconds)
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        validator: Callable[[dict[str, Any]], T] | None = None,
    ) -> T | dict[str, Any]:
        """Call the LLM, parse JSON, and optionally validate the result."""
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                )

                content = response.choices[0].message.content
                if not content:
                    raise ValueError("The model returned an empty response.")

                data = json.loads(content)
                if not isinstance(data, dict):
                    raise ValueError("The model response was valid JSON, but not a JSON object.")

                return validator(data) if validator else data
            except (
                APIError,
                APITimeoutError,
                RateLimitError,
                ValidationError,
                json.JSONDecodeError,
                ValueError,
            ) as error:
                last_error = error
                if attempt == self.max_retries:
                    break

                sleep_seconds = self.retry_delay_seconds * attempt
                print(
                    f"LLM call failed on attempt {attempt}/{self.max_retries}: {error}. "
                    f"Retrying in {sleep_seconds:.1f}s..."
                )
                time.sleep(sleep_seconds)

        raise RuntimeError(
            f"Failed to get a valid JSON response from the LLM after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )
