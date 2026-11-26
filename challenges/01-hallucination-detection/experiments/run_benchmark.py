"""Run all four detectors on the benchmark and write results.

Protocol
1. Score every example with every detector.
2. Pick each detector's decision threshold on the dev split (max F1).
3. Report metrics on the held-out test split with that threshold.

Usage
    python challenges/01-hallucination-detection/experiments/run_benchmark.py            # offline mock run
    python .../run_benchmark.py --provider anthropic --embedder openai --tag live-claude  # live run (needs keys)

Results are written to challenges/01-hallucination-detection/results/<tag>.{json,md}.
The default tag for the offline run is "mock-run"; never relabel mock output as live.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from halludetect.dataset import DEFAULT_DATASET, load_dataset
from halludetect.factory import METHODS, build_detectors
from halludetect.prompts import PROMPT_VERSION
from llm_lab.metrics import best_threshold, confusion, roc_auc
from llm_lab.text import content_tokens

CHALLENGE = Path(__file__).resolve().parents[1]
RESULTS = CHALLENGE / "results"


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(content_tokens(a)), set(content_tokens(b))
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def claim_predictions(example, detection, threshold) -> list[bool]:
    """Map flagged claims back to gold claims; a gold claim is predicted unsupported if any flagged claim maps to it."""
    pred = [False] * len(example.claims)
    if not example.claims:
        return pred
    for c in detection.flagged_claims(threshold):
        j = max(range(len(example.claims)), key=lambda i: _jaccard(c, example.claims[i].text))
        pred[j] = True
    return pred


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=CHALLENGE, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="mock", choices=["mock", "openai", "anthropic"])
    ap.add_argument("--embedder", default="hashing", choices=["hashing", "openai"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args(argv)
    tag = args.tag or ("mock-run" if args.provider == "mock" else f"live-{args.provider}")
    if args.provider == "mock" and not tag.startswith("mock"):
        raise SystemExit("Mock results must use a tag starting with 'mock'.")

    examples = load_dataset(args.dataset)
    detectors = build_detectors(args.provider, args.embedder, args.model)

    report: dict = {
        "run": {
            "tag": tag,
            "mode": "mock" if args.provider == "mock" else "live",
            "llm_provider": args.provider,
            "embedder": args.embedder,
            "model": args.model,
            "prompt_version": PROMPT_VERSION,
            "dataset": str(Path(args.dataset).relative_to(CHALLENGE.parents[1])),
            "n_examples": len(examples),
            "git_commit": git_commit(),
            "python": platform.python_version(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "methods": {},
    }
    per_example: list[dict] = []

    for name in METHODS:
        det = detectors[name]
        dets, latencies = {}, []
        for ex in examples:
            t0 = time.perf_counter()
            dets[ex.id] = det.detect(ex)
            latencies.append((time.perf_counter() - t0) * 1000)

        dev = [e for e in examples if e.split == "dev"]
        test = [e for e in examples if e.split == "test"]
        thr = best_threshold([e.hallucinated for e in dev], [dets[e.id].score for e in dev])

        def metrics(split, thr=thr, dets=dets):
            y = [e.hallucinated for e in split]
            s = [dets[e.id].score for e in split]
            m = confusion(y, [v >= thr for v in s]).to_dict()
            m["roc_auc"] = roc_auc(y, s)
            return m

        by_type = defaultdict(lambda: [0, 0])
        for e in test:
            if e.hallucinated:
                by_type[e.hallucination_type][1] += 1
                by_type[e.hallucination_type][0] += dets[e.id].score >= thr
        faithful_test = [e for e in test if not e.hallucinated]

        entry = {
            "threshold_from_dev": thr,
            "dev": metrics(dev),
            "test": metrics(test),
            "test_recall_by_type": {k: {"caught": v[0], "total": v[1]} for k, v in sorted(by_type.items())},
            "test_false_positive_rate": sum(dets[e.id].score >= thr for e in faithful_test) / len(faithful_test),
            "llm_calls_total": sum(d.llm_calls for d in dets.values()),
            "llm_calls_per_example": sum(d.llm_calls for d in dets.values()) / len(examples),
            "latency_ms_mean": sum(latencies) / len(latencies),
        }
        llm = getattr(det, "llm", None)
        if llm is not None:
            entry["token_usage"] = llm.usage()
            entry["token_usage_note"] = "Estimated as len/4 in mock mode." if args.provider == "mock" else "Reported by API."
        if det.claim_level:
            yt, yp = [], []
            for e in test:
                yt += [not c.supported for c in e.claims]
                yp += claim_predictions(e, dets[e.id], thr)
            entry["test_claim_level"] = confusion(yt, yp).to_dict()
        report["methods"][name] = entry

        for e in examples:
            d = dets[e.id]
            per_example.append({
                "id": e.id, "method": name, "score": round(d.score, 4), "predicted": d.score >= thr,
                "gold": e.hallucinated, "type": e.hallucination_type,
                "flagged_claims": d.flagged_claims(thr) if d.claims is not None else None,
            })

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}.json").write_text(json.dumps(report, indent=2) + "\n")
    with (out / f"{tag}.predictions.jsonl").open("w") as f:
        for row in per_example:
            f.write(json.dumps(row) + "\n")
    (out / f"{tag}.md").write_text(render_markdown(report))
    print(render_markdown(report))
    return report


def render_markdown(r: dict) -> str:
    run = r["run"]
    banner = (
        "> **Mock run.** LLM calls were answered by deterministic heuristics in "
        "`halludetect/mock_responders.py` and embeddings came from a hashing embedder. These numbers "
        "check the pipeline end to end; they are not measurements of any real model.\n"
        if run["mode"] == "mock"
        else f"> **Live run** with provider `{run['llm_provider']}` model `{run['model']}`.\n"
    )
    lines = [
        f"# Challenge 01 results: `{run['tag']}`",
        "",
        "_Generated by `experiments/run_benchmark.py`. Do not edit by hand._",
        "",
        banner,
        f"Dataset `{run['dataset']}` ({run['n_examples']} examples, 50 dev / 50 test), prompt version "
        f"`{run['prompt_version']}`, commit `{run['git_commit']}`, Python {run['python']}, {run['timestamp_utc']}.",
        "",
        "Thresholds are chosen on the dev split and applied to the test split.",
        "",
        "## Answer-level detection (test split)",
        "",
        "| Method | Threshold | Precision | Recall | F1 | Accuracy | ROC-AUC | FPR on faithful | LLM calls / example |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in r["methods"].items():
        t = m["test"]
        lines.append(
            f"| {name} | {m['threshold_from_dev']:.3f} | {t['precision']:.2f} | {t['recall']:.2f} | {t['f1']:.2f} | "
            f"{t['accuracy']:.2f} | {t['roc_auc']:.2f} | {m['test_false_positive_rate']:.2f} | {m['llm_calls_per_example']:.1f} |"
        )
    lines += ["", "## Claim-level detection (test split)", "",
              "Can the method point at the unsupported sentence, not just the answer?", "",
              "| Method | Claim precision | Claim recall | Claim F1 |", "|---|---|---|---|"]
    for name, m in r["methods"].items():
        if "test_claim_level" in m:
            c = m["test_claim_level"]
            lines.append(f"| {name} | {c['precision']:.2f} | {c['recall']:.2f} | {c['f1']:.2f} |")
        else:
            lines.append(f"| {name} | n/a | n/a | n/a |")
    types = sorted({k for m in r["methods"].values() for k in m["test_recall_by_type"]})
    lines += ["", "## Recall by hallucination type (test split, caught / total)", "",
              "| Method | " + " | ".join(types) + " |", "|---|" + "---|" * len(types)]
    for name, m in r["methods"].items():
        cells = [f"{m['test_recall_by_type'].get(t, {}).get('caught', 0)}/{m['test_recall_by_type'].get(t, {}).get('total', 0)}"
                 for t in types]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "## Dev split (for reference)", "", "| Method | F1 | ROC-AUC |", "|---|---|---|"]
    for name, m in r["methods"].items():
        lines.append(f"| {name} | {m['dev']['f1']:.2f} | {m['dev']['roc_auc']:.2f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
