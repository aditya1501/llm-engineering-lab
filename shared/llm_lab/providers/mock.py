"""Offline providers used for tests and for the default benchmark run.

Nothing here calls the network. Results produced with these providers are
"mock-run" results: they validate the pipeline, prompts, parsing and metrics,
not the quality of any real model.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from llm_lab.providers.base import EmbeddingProvider, LLMProvider, LLMResponse


def estimate_tokens(text: str) -> int:
    """Rough token estimate (about 4 characters per token). Mock-only."""
    return max(1, len(text) // 4)


Responder = Callable[[str, str | None], str]


class MockLLMProvider(LLMProvider):
    """Deterministic provider driven by a Python function.

    The responder receives (prompt, system) and returns the completion text.
    Challenges plug in a responder that simulates the behaviour they need.
    """

    name = "mock"

    def __init__(self, responder: Responder, model: str = "mock-1") -> None:
        super().__init__()
        self.responder = responder
        self.model = model

    def _complete(self, prompt: str, system: str | None, max_tokens: int) -> LLMResponse:
        start = time.perf_counter()
        text = self.responder(prompt, system)
        return LLMResponse(
            text=text,
            model=self.model,
            provider=self.name,
            input_tokens=estimate_tokens(prompt + (system or "")),
            output_tokens=estimate_tokens(text),
            latency_ms=(time.perf_counter() - start) * 1000,
        )


def prompt_key(prompt: str, system: str | None) -> str:
    return hashlib.sha256(f"{system or ''}\x00{prompt}".encode()).hexdigest()


class ReplayLLMProvider(LLMProvider):
    """Record/replay cassette.

    mode="record": forwards to `inner` and appends every response to a JSONL file.
    mode="replay": serves responses from the file and fails loudly on a miss.

    This is how a live run becomes reproducible offline: record once with real
    API keys, commit the cassette, and CI replays it byte for byte.
    """

    name = "replay"

    def __init__(self, path: str | Path, mode: str = "replay", inner: LLMProvider | None = None) -> None:
        super().__init__()
        if mode not in {"record", "replay"}:
            raise ValueError("mode must be 'record' or 'replay'")
        if mode == "record" and inner is None:
            raise ValueError("record mode needs an inner provider")
        self.path = Path(path)
        self.mode = mode
        self.inner = inner
        self._cache: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._cache[row["key"]] = row

    def _complete(self, prompt: str, system: str | None, max_tokens: int) -> LLMResponse:
        key = prompt_key(prompt, system)
        if key in self._cache:
            row = self._cache[key]
            return LLMResponse(row["text"], row["model"], f"replay:{row['provider']}",
                               row["input_tokens"], row["output_tokens"], 0.0)
        if self.mode == "replay":
            raise KeyError(f"No recorded response for prompt {key[:12]}; re-record the cassette.")
        assert self.inner is not None
        resp = self.inner.complete(prompt, system, max_tokens)
        row = {"key": key, "text": resp.text, "model": resp.model, "provider": resp.provider,
               "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens}
        self._cache[key] = row
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        return resp


_WORD = re.compile(r"[a-z0-9]+")


class HashingEmbedder(EmbeddingProvider):
    """Offline embedding: hashed word unigrams/bigrams + character trigrams.

    It is a lexical stand-in for a neural embedding model. It is useful for
    exercising the semantic-similarity pipeline without a network, but it does
    not understand paraphrase the way a real embedding model does. Mock-run
    numbers for "semantic similarity" should be read with that in mind.
    """

    name = "hashing"

    def __init__(self, dim: int = 2048) -> None:
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        words = _WORD.findall(text.lower())
        feats = [f"w:{w}" for w in words]
        feats += [f"b:{a}_{b}" for a, b in zip(words, words[1:])]
        for w in words:
            padded = f"#{w}#"
            feats += [f"c:{padded[i:i + 3]}" for i in range(len(padded) - 2)]
        return feats

    def _bucket(self, feat: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.md5(feat.encode()).digest()[:8], "little")
        return h % self.dim, (1.0 if (h >> 63) & 1 else -1.0)

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float64)
        for i, t in enumerate(texts):
            for feat in self._features(t):
                idx, sign = self._bucket(feat)
                out[i, idx] += sign
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms
