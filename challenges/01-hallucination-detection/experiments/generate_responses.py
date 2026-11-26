"""Generate fresh answers with a live LLM, then score them with every detector.

The benchmark (run_benchmark.py) uses the curated responses in the dataset,
because those carry gold labels. This script is the "generate responses using an
LLM" half of the experiment: it asks a real model to answer each question from
its context and reports how often each detector flags the answers. There are no
gold labels for generated answers, so the output is a flag rate plus the
flagged claims for manual review, not an accuracy number.

Usage (needs an API key):
    python .../generate_responses.py --provider anthropic --limit 20
Offline (--provider mock) it echoes the dataset's curated responses, which is
only useful as a smoke test of the script itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from halludetect.dataset import load_dataset
from halludetect.factory import METHODS, build_detectors
from halludetect.types import Example
from llm_lab.providers import get_llm

ANSWER_SYSTEM = "Answer the question using only the context. Be concise: two or three sentences."
ANSWER_TEMPLATE = "<context>\n{context}\n</context>\n\n<question>\n{question}\n</question>"
OUT = Path(__file__).resolve().parents[1] / "results"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="mock", choices=["mock", "openai", "anthropic"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args(argv)

    examples = load_dataset()[: args.limit]
    by_question = {e.question: e.response for e in examples}

    def echo(prompt: str, system: str | None) -> str:
        q = prompt.split("<question>\n", 1)[1].split("\n</question>", 1)[0]
        return by_question[q]

    generator = get_llm(args.provider, responder=echo if args.provider == "mock" else None, model=args.model)
    detectors = build_detectors(args.provider, "hashing" if args.provider == "mock" else "openai", args.model)

    rows = []
    for e in examples:
        answer = generator.complete(ANSWER_TEMPLATE.format(context=e.context, question=e.question), system=ANSWER_SYSTEM).text
        ex = Example(id=e.id, question=e.question, context=e.context, response=answer)
        row = {"id": e.id, "question": e.question, "answer": answer}
        for name in METHODS:
            d = detectors[name].detect(ex)
            row[name] = {"score": d.score, "flagged": d.flagged_claims(1.0) if d.claims is not None else None}
        rows.append(row)

    tag = "mock" if args.provider == "mock" else f"live-{args.provider}"
    path = OUT / f"generated-{tag}.jsonl"
    OUT.mkdir(exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"Wrote {len(rows)} generated answers with detector scores to {path}")


if __name__ == "__main__":
    main()
