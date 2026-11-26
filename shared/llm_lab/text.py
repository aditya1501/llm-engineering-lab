"""Small, dependency-free text helpers shared across challenges."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an and are as at be been by for from has have in is it its of on or that the
    their there these this to was were which with will would can could do does did
    what who when where why how than then so such into also about over after before
    up out if not no""".split()
)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


_ABBREVIATIONS = ("Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "St.", "e.g.", "i.e.", "vs.", "Inc.", "Ltd.")


def split_sentences(text: str) -> list[str]:
    protected = text.strip()
    for abbr in _ABBREVIATIONS:
        protected = protected.replace(abbr + " ", abbr.replace(".", "\x00") + " ")
    parts = _SENT_SPLIT.split(protected)
    return [p.replace("\x00", ".").strip() for p in parts if p.strip()]


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def normalize_token(tok: str) -> str:
    """Tiny suffix stripper. Good enough for overlap metrics, not linguistics."""
    tok = tok.replace(",", "")
    if tok.isdigit() or any(c.isdigit() for c in tok):
        return tok
    for suf in ("ing", "edly", "ed", "es", "s", "ly"):
        if len(tok) > len(suf) + 2 and tok.endswith(suf):
            return tok[: -len(suf)]
    return tok


def content_tokens(text: str) -> list[str]:
    return [normalize_token(t) for t in tokenize(text) if t not in STOPWORDS]


def numbers(text: str) -> set[str]:
    return {t.replace(",", "") for t in _TOKEN.findall(text.lower()) if any(c.isdigit() for c in t)}
