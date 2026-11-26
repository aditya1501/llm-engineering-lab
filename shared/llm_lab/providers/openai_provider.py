"""OpenAI provider. Requires `pip install openai` and OPENAI_API_KEY."""

from __future__ import annotations

import os
import time

import numpy as np

from llm_lab.providers.base import EmbeddingProvider, LLMProvider, LLMResponse


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, model: str | None = None) -> None:
        super().__init__()
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("Install the 'live' extra: pip install -e '.[live]'") from e
        self.client = OpenAI()
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    def _complete(self, prompt: str, system: str | None, max_tokens: int) -> LLMResponse:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        start = time.perf_counter()
        r = self.client.chat.completions.create(model=self.model, messages=messages, max_tokens=max_tokens)
        return LLMResponse(
            text=r.choices[0].message.content or "",
            model=r.model,
            provider=self.name,
            input_tokens=r.usage.prompt_tokens if r.usage else 0,
            output_tokens=r.usage.completion_tokens if r.usage else 0,
            latency_ms=(time.perf_counter() - start) * 1000,
        )


class OpenAIEmbedder(EmbeddingProvider):
    name = "openai-embed"

    def __init__(self, model: str | None = None) -> None:
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model or os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    def embed(self, texts: list[str]) -> np.ndarray:
        r = self.client.embeddings.create(model=self.model, input=texts)
        arr = np.array([d.embedding for d in r.data], dtype=np.float64)
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)
