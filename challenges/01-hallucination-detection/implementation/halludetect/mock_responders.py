"""Deterministic stand-ins for the LLM calls, used when no API key is set.

IMPORTANT: these are heuristics, not models. They parse the same prompts a real
model would receive and answer in the same JSON format, so the full pipeline
(prompting, parsing, aggregation, metrics) runs offline and in CI. Numbers
produced with them are "mock-run" numbers. They say nothing about how GPT or
Claude would score; re-run with --provider openai|anthropic for that.

Heuristics used (chosen up front, not tuned on the test split):
- numbers in the text that do not appear in the context -> unsupported
- share of content words found in the context (lexical coverage)
- negation mismatch against the best-matching context sentence -> contradicted
The holistic judge looks at the whole answer at once; the claim verifier looks
at one claim at a time. That granularity difference is the thing this
challenge wants to make visible.
"""

from __future__ import annotations

import json
import re

from llm_lab.text import content_tokens, numbers, split_sentences

_TAG = re.compile(r"<(\w+)>\n?(.*?)\n?</\1>", re.DOTALL)
_PARTICLE = re.compile(r"^(yes|no)[,.;:]?\s+", re.IGNORECASE)
_NEGATION = re.compile(r"\b(?:not|no|never|none|cannot|without)\b|n't\b", re.IGNORECASE)

COVERAGE_SUPPORTED = 0.75


def _tags(prompt: str) -> dict[str, str]:
    return {k: v.strip() for k, v in _TAG.findall(prompt)}


def strip_particle(text: str) -> str:
    return _PARTICLE.sub("", text.strip())


def _negated(text: str) -> bool:
    return bool(_NEGATION.search(text))


def analyse(text: str, context: str) -> dict:
    text = strip_particle(text)
    ctx_tokens = set(content_tokens(context))
    toks = content_tokens(text)
    coverage = sum(t in ctx_tokens for t in toks) / len(toks) if toks else 1.0
    new_numbers = sorted(numbers(text) - numbers(context))

    best, best_overlap = "", 0.0
    tset = set(toks)
    for sent in split_sentences(context):
        s = set(content_tokens(sent))
        overlap = len(tset & s) / len(tset) if tset else 0.0
        if overlap > best_overlap:
            best, best_overlap = sent, overlap
    negation_mismatch = best_overlap >= 0.5 and _negated(text) != _negated(best)
    return {
        "coverage": coverage,
        "new_numbers": new_numbers,
        "negation_mismatch": negation_mismatch,
        "best_sentence": best,
    }


def judge_responder(prompt: str, system: str | None) -> str:
    """Holistic judge: one look at the whole answer."""
    t = _tags(prompt)
    a = analyse(t["answer"], t["context"])
    if a["new_numbers"]:
        g, reason = 2, f"Numbers not found in context: {', '.join(a['new_numbers'])}."
    elif a["negation_mismatch"]:
        g, reason = 2, "The answer's polarity conflicts with the context."
    elif a["coverage"] >= 0.9:
        g, reason = 5, "All statements are supported by the context."
    elif a["coverage"] >= COVERAGE_SUPPORTED:
        g, reason = 4, "Mostly supported, with some paraphrase."
    elif a["coverage"] >= 0.6:
        g, reason = 3, "Some details cannot be verified from the context."
    else:
        g, reason = 1, "Most of the answer is not in the context."
    return json.dumps({"groundedness": g, "reason": reason})


def decompose_responder(prompt: str, system: str | None) -> str:
    answer = _tags(prompt)["answer"]
    claims: list[str] = []
    for sent in split_sentences(answer):
        for part in re.split(r";\s*", sent):
            part = strip_particle(part).strip()
            if part:
                claims.append(part[0].upper() + part[1:])
    return json.dumps({"claims": claims})


def verify_responder(prompt: str, system: str | None) -> str:
    t = _tags(prompt)
    a = analyse(t["claim"], t["context"])
    if a["new_numbers"]:
        verdict, evidence = "unsupported", ""
    elif a["negation_mismatch"]:
        verdict, evidence = "contradicted", a["best_sentence"]
    elif a["coverage"] >= COVERAGE_SUPPORTED:
        verdict, evidence = "supported", a["best_sentence"]
    else:
        verdict, evidence = "unsupported", ""
    return json.dumps({"verdict": verdict, "evidence": evidence})


def router(prompt: str, system: str | None) -> str:
    """Single responder that dispatches on prompt shape, so one mock provider serves all detectors."""
    if "<claim>" in prompt:
        return verify_responder(prompt, system)
    if "atomic" in prompt and "<context>" not in prompt:
        return decompose_responder(prompt, system)
    if "groundedness" in prompt:
        return judge_responder(prompt, system)
    raise ValueError("Mock router: unrecognised prompt")
