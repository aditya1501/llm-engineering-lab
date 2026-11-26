"""Build detectors for a given run mode."""

from __future__ import annotations

from halludetect.detectors import (
    ClaimVerificationDetector,
    Detector,
    ExactMatchDetector,
    LLMJudgeDetector,
    SemanticSimilarityDetector,
)
from halludetect.mock_responders import router
from llm_lab.providers import get_embedder, get_llm

METHODS = ("exact_match", "semantic_similarity", "llm_judge", "claim_verification")


def build_detectors(provider: str = "mock", embedder: str = "hashing", model: str | None = None) -> dict[str, Detector]:
    def llm():
        return get_llm(provider, responder=router if provider == "mock" else None, model=model)

    return {
        "exact_match": ExactMatchDetector(),
        "semantic_similarity": SemanticSimilarityDetector(get_embedder(embedder)),
        "llm_judge": LLMJudgeDetector(llm()),
        "claim_verification": ClaimVerificationDetector(llm()),
    }
