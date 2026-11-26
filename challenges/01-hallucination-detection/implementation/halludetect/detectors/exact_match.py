"""Baseline: lexical overlap between answer and context.

score = 1 - (share of the answer's content words that appear in the context)

This is ROUGE-1 precision with the context as the reference. It is free, fast
and explainable, and it is blind to meaning: a paraphrase looks like a
hallucination and a swapped entity that also appears in the context looks fine.
"""

from __future__ import annotations

from halludetect.detectors.base import Detector
from halludetect.types import Detection, Example
from llm_lab.text import content_tokens


class ExactMatchDetector(Detector):
    name = "exact_match"

    def detect(self, example: Example) -> Detection:
        ctx = set(content_tokens(example.context))
        toks = content_tokens(example.response)
        if not toks:
            return Detection(score=0.0)
        overlap = sum(t in ctx for t in toks) / len(toks)
        return Detection(score=1.0 - overlap, meta={"overlap": overlap})
