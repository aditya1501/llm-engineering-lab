import numpy as np
import pytest

from llm_lab.metrics import best_threshold, confusion, roc_auc
from llm_lab.providers import HashingEmbedder, MockLLMProvider, ReplayLLMProvider, get_llm
from llm_lab.text import content_tokens, numbers, split_sentences


def test_confusion_and_f1():
    m = confusion([True, True, False, False], [True, False, True, False])
    assert (m.tp, m.fp, m.tn, m.fn) == (1, 1, 1, 1)
    assert m.f1 == pytest.approx(0.5)


def test_roc_auc_perfect_and_random():
    assert roc_auc([True, False], [0.9, 0.1]) == 1.0
    assert roc_auc([True, False], [0.5, 0.5]) == 0.5


def test_best_threshold():
    assert best_threshold([False, False, True, True], [0.1, 0.2, 0.8, 0.9]) == 0.8


def test_text_helpers():
    assert split_sentences("A b. C d! E?") == ["A b.", "C d!", "E?"]
    assert numbers("Costs $5,000 or 3.1%") == {"5000", "3.1"}
    assert "the" not in content_tokens("the robots")


def test_hashing_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder()
    a, b = e.embed(["warehouse robots"]), e.embed(["warehouse robots"])
    assert np.allclose(a, b)
    assert np.isclose(np.linalg.norm(a), 1.0)
    sim_same = float((e.embed(["red apple"]) @ e.embed(["red apples"]).T)[0, 0])
    sim_diff = float((e.embed(["red apple"]) @ e.embed(["tax policy"]).T)[0, 0])
    assert sim_same > sim_diff


def test_mock_provider_counts_usage():
    p = MockLLMProvider(lambda prompt, system: "hello")
    p.complete("hi")
    assert p.usage()["calls"] == 1


def test_replay_roundtrip(tmp_path):
    path = tmp_path / "cassette.jsonl"
    rec = ReplayLLMProvider(path, mode="record", inner=MockLLMProvider(lambda p, s: p.upper()))
    assert rec.complete("abc").text == "ABC"
    rep = ReplayLLMProvider(path, mode="replay")
    assert rep.complete("abc").text == "ABC"
    with pytest.raises(KeyError):
        rep.complete("never recorded")


def test_factory_requires_responder_for_mock():
    with pytest.raises(ValueError):
        get_llm("mock")
