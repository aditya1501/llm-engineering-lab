"""Improved method: decompose the answer into atomic claims, verify each one.

1 call to decompose + 1 call per claim. The answer score is the worst claim's
score, and the unsupported claims themselves are returned, which is what a
product actually needs (highlight the bad sentence, not just "this answer is bad").
"""

from __future__ import annotations

from halludetect.detectors.base import Detector
from halludetect.parsing import ParseError, extract_json
from halludetect.prompts import DECOMPOSE_SYSTEM, DECOMPOSE_TEMPLATE, VERIFY_SYSTEM, VERIFY_TEMPLATE
from halludetect.types import ClaimVerdict, Detection, Example
from llm_lab.providers.base import LLMProvider
from llm_lab.text import split_sentences

VERDICT_SCORE = {"supported": 0.0, "unsupported": 1.0, "contradicted": 1.0}


class ClaimVerificationDetector(Detector):
    name = "claim_verification"
    claim_level = True

    def __init__(self, llm: LLMProvider, decomposer: LLMProvider | None = None) -> None:
        self.llm = llm
        self.decomposer = decomposer or llm

    def decompose(self, answer: str) -> list[str]:
        resp = self.decomposer.complete(DECOMPOSE_TEMPLATE.format(answer=answer), system=DECOMPOSE_SYSTEM)
        try:
            claims = [str(c).strip() for c in extract_json(resp.text)["claims"] if str(c).strip()]
        except (ParseError, KeyError, TypeError):
            claims = []
        # Fallback: sentence split. Never silently verify nothing.
        return claims or split_sentences(answer)

    def verify(self, claim: str, context: str) -> ClaimVerdict:
        resp = self.llm.complete(VERIFY_TEMPLATE.format(context=context, claim=claim), system=VERIFY_SYSTEM, max_tokens=256)
        try:
            data = extract_json(resp.text)
            verdict = str(data["verdict"]).lower().strip()
            if verdict not in VERDICT_SCORE:
                raise ValueError(verdict)
            reason = str(data.get("evidence", ""))
        except (ParseError, KeyError, TypeError, ValueError):
            verdict, reason = "unsupported", f"unparseable verifier output: {resp.text[:80]!r}"
        return ClaimVerdict(claim=claim, verdict=verdict, score=VERDICT_SCORE[verdict], reason=reason)

    def detect(self, example: Example) -> Detection:
        claims = self.decompose(example.response)
        verdicts = [self.verify(c, example.context) for c in claims]
        score = max((v.score for v in verdicts), default=0.0)
        return Detection(score=score, claims=verdicts, llm_calls=1 + len(claims))
