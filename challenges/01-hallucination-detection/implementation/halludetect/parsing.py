"""Robust JSON extraction from model output.

Real models wrap JSON in prose or code fences, or return slightly broken JSON.
A judge that crashes on one bad response is worse than one that degrades.
"""

from __future__ import annotations

import json
import re

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_OBJ = re.compile(r"\{.*\}", re.DOTALL)


class ParseError(ValueError):
    pass


def extract_json(text: str) -> dict:
    candidates = [m.group(1) for m in _FENCE.finditer(text)]
    m = _OBJ.search(text)
    if m:
        candidates.append(m.group(0))
    candidates.append(text)
    for c in candidates:
        try:
            obj = json.loads(c.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ParseError(f"No JSON object found in model output: {text[:120]!r}")
