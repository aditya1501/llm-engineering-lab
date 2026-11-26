"""LLM-as-judge: one call that rates the whole answer's groundedness (1-5)."""

from __future__ import annotations

from halludetect.detectors.base import Detector
from halludetect.parsing import ParseError, extract_json
from halludetect.prompts import JUDGE_SYSTEM, JUDGE_TEMPLATE
from halludetect.types import Detection, Example
from llm_lab.providers.base import LLMProvider


class LLMJudgeDetector(Detector):
    name = "llm_judge"

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    def detect(self, example: Example) -> Detection:
        prompt = JUDGE_TEMPLATE.format(context=example.context, question=example.question, answer=example.response)
        resp = self.llm.complete(prompt, system=JUDGE_SYSTEM, max_tokens=256)
        try:
            data = extract_json(resp.text)
            g = int(data["groundedness"])
            g = min(5, max(1, g))
            reason = str(data.get("reason", ""))
        except (ParseError, KeyError, TypeError, ValueError):
            # Fail closed: an unparseable judgement is treated as "not verified".
            g, reason = 1, f"unparseable judge output: {resp.text[:80]!r}"
        return Detection(score=(5 - g) / 4, llm_calls=1, meta={"groundedness": g, "reason": reason})
