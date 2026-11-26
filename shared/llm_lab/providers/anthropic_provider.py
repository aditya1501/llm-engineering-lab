"""Anthropic provider. Requires `pip install anthropic` and ANTHROPIC_API_KEY."""

from __future__ import annotations

import os
import time

from llm_lab.providers.base import LLMProvider, LLMResponse


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, model: str | None = None) -> None:
        super().__init__()
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("Install the 'live' extra: pip install -e '.[live]'") from e
        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-opus-5-5")

    def _complete(self, prompt: str, system: str | None, max_tokens: int) -> LLMResponse:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        start = time.perf_counter()
        r = self.client.messages.create(**kwargs)
        if r.stop_reason == "refusal":
            text = ""
        else:
            text = "".join(b.text for b in r.content if b.type == "text")
        return LLMResponse(
            text=text,
            model=r.model,
            provider=self.name,
            input_tokens=r.usage.input_tokens,
            output_tokens=r.usage.output_tokens,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
