"""Fail if benchmarks/results.json (from run_redaction_benchmark.py) drops
below the committed floor in benchmarks/thresholds.json. Run after the
benchmark in CI so a detector regression fails the build instead of
quietly shipping.

Usage:
    python benchmarks/run_redaction_benchmark.py
    python benchmarks/check_thresholds.py
"""

import json
import os
import sys

HERE = os.path.dirname(__file__)
RESULTS_PATH = os.path.join(HERE, "results.json")
THRESHOLDS_PATH = os.path.join(HERE, "thresholds.json")


def main():
    if not os.path.exists(RESULTS_PATH):
        print(f"{RESULTS_PATH} not found -- run run_redaction_benchmark.py first", file=sys.stderr)
        sys.exit(2)

    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)
    with open(THRESHOLDS_PATH, "r", encoding="utf-8") as f:
        thresholds = json.load(f)

    failures = []
    for tier, metrics in thresholds.items():
        if tier.startswith("_"):
            continue
        if tier not in results:
            failures.append(f"tier {tier!r} is in thresholds.json but missing from results.json")
            continue
        actual = results[tier]["overall"]
        for metric, floor in metrics.items():
            value = actual.get(metric)
            if value is None or value < floor:
                failures.append(f"{tier}/{metric}: {value} is below the committed floor {floor}")

    if failures:
        print("Benchmark regression detected:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        sys.exit(1)

    print("All tiers meet their committed thresholds.")


if __name__ == "__main__":
    main()
