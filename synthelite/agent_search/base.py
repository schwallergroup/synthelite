from abc import ABC, abstractmethod
import os
from typing import List

import litellm
from openai import OpenAI, AsyncOpenAI


class LLMBase(ABC):
    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.1,
        backend: str = "openrouter",
        verbose: bool = False,
    ):
        self.model = model
        self.temperature = temperature
        self.verbose = verbose
        self.backend = backend

        if self.verbose:
            print(f"Strategy Planner initialized with {model}")

        if backend == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenRouter API key not found in environment variables"
                )
            self.client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
        elif backend.startswith("http"):
            self.client = AsyncOpenAI(
                base_url=backend,
                api_key="EMPTY",
            )
        elif backend == "litellm":
            self.client = None

    async def _query_llm(
        self,
        messages: List[dict[str, str]],
    ) -> str:
        """Send prompt to LLM and return raw response text."""
        if self.verbose:
            print(f"Calling {self.model}...")

        if self.client:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )
        else:
            response = await litellm.acompletion(
                model=self.model, messages=messages, temperature=self.temperature
            )
        return response.choices[0].message.content
