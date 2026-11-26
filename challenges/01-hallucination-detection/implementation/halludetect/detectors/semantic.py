"""Embedding similarity at sentence level.

For each answer sentence take its best cosine similarity to any context
sentence. The answer is as weak as its weakest sentence, so
score = 1 - min_sentence(max_context_sim).

Works with any EmbeddingProvider. Offline it uses the hashing embedder, which is
lexical; with a neural embedder it tolerates paraphrase, but it still cannot see
that "3.1%" and "1.2%" mean different things in otherwise identical sentences.
"""

from __future__ import annotations

import numpy as np

from halludetect.detectors.base import Detector
from halludetect.types import ClaimVerdict, Detection, Example
from llm_lab.providers.base import EmbeddingProvider
from llm_lab.text import split_sentences


class SemanticSimilarityDetector(Detector):
    name = "semantic_similarity"
    claim_level = True

    def __init__(self, embedder: EmbeddingProvider) -> None:
        self.embedder = embedder

    def detect(self, example: Example) -> Detection:
        ans = split_sentences(example.response)
        ctx = split_sentences(example.context)
        if not ans or not ctx:
            return Detection(score=0.0, claims=[])
        a = self.embedder.embed(ans)
        c = self.embedder.embed(ctx)
        sims = (a @ c.T).max(axis=1)
        verdicts = [
            ClaimVerdict(claim=s, verdict="similarity", score=float(1.0 - sim), reason=f"max_sim={sim:.3f}")
            for s, sim in zip(ans, sims)
        ]
        return Detection(score=float(1.0 - np.min(sims)), claims=verdicts)
