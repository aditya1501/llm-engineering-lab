"""Provider-agnostic interfaces.

Every challenge talks to models only through these two interfaces, so swapping
OpenAI for Anthropic (or for a deterministic mock in CI) is a config change,
not a code change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    raw: dict = field(default_factory=dict, compare=False, repr=False)


class LLMProvider(ABC):
    """Minimal text-in / text-out completion interface."""

    name: str = "base"

    def __init__(self) -> None:
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    @abstractmethod
    def _complete(self, prompt: str, system: str | None, max_tokens: int) -> LLMResponse: ...

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 1024) -> LLMResponse:
        resp = self._complete(prompt, system, max_tokens)
        self.calls += 1
        self.input_tokens += resp.input_tokens
        self.output_tokens += resp.output_tokens
        return resp

    def usage(self) -> dict:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


class EmbeddingProvider(ABC):
    name: str = "base"

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, d) array of L2-normalised embeddings."""
