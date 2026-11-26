from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Claim:
    text: str
    supported: bool


@dataclass(frozen=True)
class Example:
    id: str
    question: str
    context: str
    response: str
    hallucinated: bool | None = None
    claims: tuple[Claim, ...] = ()
    hallucination_type: str | None = None
    split: str | None = None
    domain: str | None = None
    base_id: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> Example:
        return cls(
            id=d["id"],
            question=d["question"],
            context=d["context"],
            response=d["response"],
            hallucinated=d.get("hallucinated"),
            claims=tuple(Claim(c["text"], c["supported"]) for c in d.get("claims", [])),
            hallucination_type=d.get("hallucination_type"),
            split=d.get("split"),
            domain=d.get("domain"),
            base_id=d.get("base_id"),
        )


@dataclass
class ClaimVerdict:
    claim: str
    verdict: str  # "supported" | "unsupported" | "contradicted"
    score: float  # 0 = grounded, 1 = hallucinated
    reason: str = ""


@dataclass
class Detection:
    """Output of a detector.

    score: hallucination score in [0, 1]; higher means more likely unsupported.
    claims: per-claim verdicts when the method works at claim/sentence level.
    """

    score: float
    claims: list[ClaimVerdict] | None = None
    llm_calls: int = 0
    meta: dict = field(default_factory=dict)

    def flagged_claims(self, threshold: float) -> list[str]:
        return [c.claim for c in (self.claims or []) if c.score >= threshold]
