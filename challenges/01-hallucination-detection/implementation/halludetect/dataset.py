from __future__ import annotations

import json
from pathlib import Path

from halludetect.types import Example

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATASET = REPO_ROOT / "datasets" / "hallucination_qa" / "v1.jsonl"


def load_dataset(path: str | Path = DEFAULT_DATASET, split: str | None = None) -> list[Example]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    examples = [Example.from_dict(r) for r in rows]
    if split:
        examples = [e for e in examples if e.split == split]
    return examples
