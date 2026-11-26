import json

import pytest

from halludetect.detectors import (
    ClaimVerificationDetector,
    ExactMatchDetector,
    LLMJudgeDetector,
    SemanticSimilarityDetector,
)
from halludetect.mock_responders import router
from halludetect.types import Example
from llm_lab.providers import HashingEmbedder, MockLLMProvider

CTX = "The Tejas-7 microcontroller runs at 133 MHz and has 264 KB of RAM. It supports USB 1.1 and has 26 GPIO pins."
GOOD = Example("g", "How much RAM?", CTX, "The Tejas-7 has 264 KB of RAM. It runs at 133 MHz.")
BAD = Example("b", "How much RAM?", CTX, "The Tejas-7 has 512 KB of RAM. It runs at 133 MHz.")


@pytest.fixture
def llm():
    return MockLLMProvider(router)


def test_exact_match_scores_copy_as_grounded():
    d = ExactMatchDetector().detect(Example("c", "q", CTX, CTX))
    assert d.score == 0.0


def test_exact_match_flags_new_words():
    d = ExactMatchDetector().detect(Example("c", "q", CTX, "It ships with a lithium battery and Bluetooth."))
    assert d.score > 0.5


def test_semantic_returns_sentence_scores():
    d = SemanticSimilarityDetector(HashingEmbedder()).detect(GOOD)
    assert len(d.claims) == 2
    assert 0.0 <= d.score <= 1.0


def test_judge_flags_changed_number(llm):
    det = LLMJudgeDetector(llm)
    assert det.detect(GOOD).score < det.detect(BAD).score
    assert llm.calls == 2


def test_claim_verification_points_at_bad_claim(llm):
    d = ClaimVerificationDetector(llm).detect(BAD)
    assert d.score == 1.0
    assert d.flagged_claims(1.0) == ["The Tejas-7 has 512 KB of RAM."]
    assert d.llm_calls == 3  # decompose + 2 claims


def test_judge_fails_closed_on_garbage():
    det = LLMJudgeDetector(MockLLMProvider(lambda p, s: "I think it's fine!"))
    d = det.detect(GOOD)
    assert d.score == 1.0 and "unparseable" in d.meta["reason"]


def test_claim_verification_survives_bad_decomposition():
    def responder(prompt, system):
        if "<claim>" in prompt:
            return json.dumps({"verdict": "supported", "evidence": ""})
        return "not json"

    d = ClaimVerificationDetector(MockLLMProvider(responder)).detect(GOOD)
    assert len(d.claims) == 2  # fell back to sentence split


def test_claim_verification_rejects_unknown_verdict():
    def responder(prompt, system):
        if "<claim>" in prompt:
            return json.dumps({"verdict": "probably"})
        return json.dumps({"claims": ["x"]})

    d = ClaimVerificationDetector(MockLLMProvider(responder)).detect(GOOD)
    assert d.claims[0].verdict == "unsupported"
