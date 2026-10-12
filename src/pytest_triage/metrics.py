"""Scoring. Every predictor is scored by the same code, so numbers compare."""

import math
from collections import Counter
from collections.abc import Callable, Iterable

from pytest_triage.config import LABELS
from pytest_triage.dataset import targets

Predictor = Callable[[dict], set[str]]


def confusion(predictions: Iterable[tuple[set[str], set[str]]]) -> dict[str, Counter]:
    """Per label: true positives, false positives, false negatives.

    `predictions` yields (truth, guess) pairs. Labels outside LABELS are ignored.
    """
    counts = {label: Counter() for label in LABELS}
    for truth, guess in predictions:
        for label in LABELS:
            if label in guess and label in truth:
                counts[label]["tp"] += 1
            elif label in guess:
                counts[label]["fp"] += 1
            elif label in truth:
                counts[label]["fn"] += 1
    return counts


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, recall, F1 - each 0.0 when its denominator is 0."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def label_report(counts: dict[str, Counter]) -> dict:
    """Per-label P/R/F1/support, plus macro-F1 (labels equal) and micro-F1 (issues equal)."""
    per_label = {}
    for label in LABELS:
        c = counts[label]
        p, r, f1 = prf(c["tp"], c["fp"], c["fn"])
        per_label[label] = {"precision": p, "recall": r, "f1": f1, "support": c["tp"] + c["fn"]}
    total = Counter()
    for c in counts.values():
        total.update(c)
    return {
        "labels": per_label,
        "macro_f1": sum(v["f1"] for v in per_label.values()) / len(LABELS),
        "micro_f1": prf(total["tp"], total["fp"], total["fn"])[2],
    }


def score_predictor(predict: Predictor, issues: list[dict]) -> dict:
    return label_report(confusion((targets(i), predict(i) & set(LABELS)) for i in issues))


def hit_at_k(ranked: list[int], truth: Iterable[int], k: int) -> bool:
    return any(t in ranked[:k] for t in truth)


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a rate. Honest error bars for small n."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)
