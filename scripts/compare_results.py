"""Fail if two benchmark result files disagree on any metric.

CI uses this to prove the committed mock-run numbers are exactly what the code
produces today. If you change a detector, re-run the benchmark and commit the
new results in the same PR.
"""

import json
import sys

IGNORED = {"run", "latency_ms_mean"}


def strip(obj):
    if isinstance(obj, dict):
        return {k: strip(v) for k, v in obj.items() if k not in IGNORED}
    return obj


def main(a: str, b: str) -> int:
    ra, rb = strip(json.load(open(a))), strip(json.load(open(b)))
    if ra != rb:
        print("Benchmark results differ from the committed file. Re-run the benchmark and commit results/.")
        return 1
    print("Benchmark results match the committed file.")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
