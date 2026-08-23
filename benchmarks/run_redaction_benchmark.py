"""Redaction benchmark: precision/recall/F1 per entity type per detector
tier, latency percentiles, and a rough infra cost estimate.

Runs entirely against benchmarks/corpus/*.json -- synthetic, labeled
incidents in the real processed_incidents/{id}.json schema plus a parallel
"spans" ground-truth array. Zero Azure/network access required, which
matters because this clone can't reach Kusto at all.

Usage:
    python benchmarks/run_redaction_benchmark.py [--tier regex|regex+presidio|all]
"""

import argparse
import glob
import json
import os
import statistics
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from guard.detectors import RegexDetector, Span, create_presidio_detector, merge_spans  # noqa: E402

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")

# Rough infra-cost placeholder for the detector pass itself (CPU time to
# run regex/NER over the text) -- NOT the LLM token cost, which is a
# separate, already-modeled figure in config.py (AI_SERVICE_INPUT_COST /
# AI_SERVICE_OUTPUT_COST). Tune ASSUMED_COMPUTE_COST_PER_HOUR to match
# whatever compute this actually runs on before trusting the $ column.
ASSUMED_COMPUTE_COST_PER_HOUR = 0.10


def load_corpus() -> List[dict]:
    docs = []
    for path in sorted(glob.glob(os.path.join(CORPUS_DIR, "*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            docs.append(json.load(f))
    return docs


def texts_with_ground_truth(doc: dict) -> List[Tuple[str, List[dict]]]:
    """Yield (text, ground_truth_spans) for every labeled field in a doc."""
    by_field: Dict[Tuple[str, object], List[dict]] = {}
    for span in doc.get("spans", []):
        key = (span["field"], span.get("entry_index"))
        by_field.setdefault(key, []).append(span)

    results = []
    for idx, entry in enumerate(doc["conversation"]):
        gt = by_field.get(("content", idx), [])
        results.append((entry["content"], gt))
    if "summary" in doc:
        gt = by_field.get(("summary", None), [])
        results.append((doc["summary"], gt))
    return results


def overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def score_text(predicted: List[Span], ground_truth: List[dict], counters: Dict[str, Dict[str, int]]) -> None:
    """Greedy overlap matching per entity type: a ground-truth span counts
    as found if some predicted span of the same type overlaps it at all.
    Exact-boundary matching is too brittle for a regex/NER benchmark --
    what matters is whether the sensitive substring left the process."""
    gt_by_type: Dict[str, List[dict]] = {}
    for g in ground_truth:
        gt_by_type.setdefault(g["entity_type"], []).append(dict(g, matched=False))

    pred_by_type: Dict[str, List[Span]] = {}
    for p in predicted:
        pred_by_type.setdefault(p.entity_type, []).append(p)

    all_types = set(gt_by_type) | set(pred_by_type)
    for entity_type in all_types:
        counters.setdefault(entity_type, {"tp": 0, "fp": 0, "fn": 0})
        gts = gt_by_type.get(entity_type, [])
        preds = list(pred_by_type.get(entity_type, []))
        used_pred = [False] * len(preds)

        for g in gts:
            found = False
            for i, p in enumerate(preds):
                if used_pred[i]:
                    continue
                if overlaps(g["start"], g["end"], p.start, p.end):
                    used_pred[i] = True
                    found = True
                    break
            if found:
                counters[entity_type]["tp"] += 1
            else:
                counters[entity_type]["fn"] += 1

        counters[entity_type]["fp"] += used_pred.count(False)


def build_detectors(tier: str):
    detectors = [RegexDetector()]
    if tier in ("regex+presidio", "all"):
        presidio = create_presidio_detector()
        if presidio is None:
            print("WARNING: presidio unavailable, skipping presidio tier "
                  "(install requirements-redaction.txt)", file=sys.stderr)
        else:
            detectors.append(presidio)
    return detectors


def run_tier(tier: str, docs: List[dict]) -> Tuple[Dict[str, Dict[str, int]], List[float]]:
    detectors = build_detectors(tier)
    counters: Dict[str, Dict[str, int]] = {}
    latencies_ms: List[float] = []

    for doc in docs:
        for text, ground_truth in texts_with_ground_truth(doc):
            start = time.perf_counter()
            spans: List[Span] = []
            for detector in detectors:
                spans.extend(detector.detect(text))
            spans = merge_spans(spans)
            latencies_ms.append((time.perf_counter() - start) * 1000)
            score_text(spans, ground_truth, counters)

    return counters, latencies_ms


def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(pct / 100 * (len(ordered) - 1))))
    return ordered[idx]


def render_markdown(tier_results: Dict[str, Tuple[Dict[str, Dict[str, int]], List[float]]]) -> str:
    lines = ["# Redaction benchmark results", ""]
    lines.append(f"Corpus: `benchmarks/corpus/*.json` ({len(load_corpus())} synthetic incidents)")
    lines.append("")

    for tier, (counters, latencies) in tier_results.items():
        lines.append(f"## Tier: `{tier}`")
        lines.append("")
        lines.append("| Entity type | Precision | Recall | F1 | TP | FP | FN |")
        lines.append("|---|---|---|---|---|---|---|")
        total_tp = total_fp = total_fn = 0
        for entity_type in sorted(counters):
            c = counters[entity_type]
            p, r, f1 = precision_recall_f1(c["tp"], c["fp"], c["fn"])
            lines.append(f"| {entity_type} | {p:.2f} | {r:.2f} | {f1:.2f} | {c['tp']} | {c['fp']} | {c['fn']} |")
            total_tp += c["tp"]
            total_fp += c["fp"]
            total_fn += c["fn"]
        p, r, f1 = precision_recall_f1(total_tp, total_fp, total_fn)
        lines.append(f"| **overall** | **{p:.2f}** | **{r:.2f}** | **{f1:.2f}** | {total_tp} | {total_fp} | {total_fn} |")
        lines.append("")

        p50 = percentile(latencies, 50)
        p95 = percentile(latencies, 95)
        mean_s = statistics.mean(latencies) / 1000 if latencies else 0.0
        cost_per_1k = mean_s * (ASSUMED_COMPUTE_COST_PER_HOUR / 3600) * 1000
        lines.append(f"- p50 latency: {p50:.2f} ms/text, p95 latency: {p95:.2f} ms/text ({len(latencies)} texts)")
        lines.append(f"- estimated cost: ${cost_per_1k:.4f} per 1,000 incidents "
                      f"(assumes ${ASSUMED_COMPUTE_COST_PER_HOUR:.2f}/compute-hour; "
                      "detector CPU time only, not LLM token cost)")
        lines.append("")

    return "\n".join(lines)


def overall_metrics(counters: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    tp = sum(c["tp"] for c in counters.values())
    fp = sum(c["fp"] for c in counters.values())
    fn = sum(c["fn"] for c in counters.values())
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    return {"precision": precision, "recall": recall, "f1": f1}


def render_json(tier_results: Dict[str, Tuple[Dict[str, Dict[str, int]], List[float]]]) -> dict:
    out = {}
    for tier, (counters, latencies) in tier_results.items():
        per_entity = {}
        for entity_type, c in counters.items():
            precision, recall, f1 = precision_recall_f1(c["tp"], c["fp"], c["fn"])
            per_entity[entity_type] = {"precision": precision, "recall": recall, "f1": f1, **c}
        out[tier] = {
            "overall": overall_metrics(counters),
            "per_entity": per_entity,
            "latency_ms_p50": percentile(latencies, 50),
            "latency_ms_p95": percentile(latencies, 95),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=["regex", "regex+presidio", "all"], default="all")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "RESULTS.md"))
    parser.add_argument("--json-out", default=os.path.join(os.path.dirname(__file__), "results.json"))
    args = parser.parse_args()

    docs = load_corpus()
    if not docs:
        print(f"No corpus files found in {CORPUS_DIR}", file=sys.stderr)
        sys.exit(1)

    tiers = ["regex", "regex+presidio"] if args.tier == "all" else [args.tier]
    tier_results = {}
    for tier in tiers:
        tier_results[tier] = run_tier(tier, docs)

    report = render_markdown(tier_results)
    print(report)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nWrote {args.out}", file=sys.stderr)

    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(render_json(tier_results), f, indent=2)
    print(f"Wrote {args.json_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
