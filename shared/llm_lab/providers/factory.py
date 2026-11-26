"""Build providers from a name, so experiments can switch via CLI flag or env."""

from __future__ import annotations

from llm_lab.providers.base import EmbeddingProvider, LLMProvider
from llm_lab.providers.mock import HashingEmbedder, MockLLMProvider, Responder


def get_llm(name: str = "mock", *, responder: Responder | None = None, model: str | None = None) -> LLMProvider:
    if name == "mock":
        if responder is None:
            raise ValueError("The mock provider needs a responder function")
        return MockLLMProvider(responder)
    if name == "openai":
        from llm_lab.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(model)
    if name == "anthropic":
        from llm_lab.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(model)
    raise ValueError(f"Unknown LLM provider: {name}")


def get_embedder(name: str = "hashing") -> EmbeddingProvider:
    if name == "hashing":
        return HashingEmbedder()
    if name == "openai":
        from llm_lab.providers.openai_provider import OpenAIEmbedder

        return OpenAIEmbedder()
    raise ValueError(f"Unknown embedding provider: {name}")
