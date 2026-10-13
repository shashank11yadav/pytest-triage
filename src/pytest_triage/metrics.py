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


def duplicate_clusters(truth: dict[int, list[int]]) -> dict[int, set[int]]:
    """Group known duplicate pairs into clusters (union-find).

    If A duplicates C and B duplicates C, then flagging A as a duplicate of B is
    correct too - but a per-pair check would call it wrong.
    """
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for dup, targets_ in truth.items():
        for t in targets_:
            parent[find(dup)] = find(t)
    clusters: dict[int, set[int]] = {}
    for x in list(parent):
        clusters.setdefault(find(x), set()).add(x)
    return {x: clusters[find(x)] for x in parent}


def bootstrap_diff(
    truth: list[set[str]],
    a: list[set[str]],
    b: list[set[str]],
    rounds: int = 2000,
    seed: int = 0,
) -> dict:
    """Paired bootstrap of macro-F1(a) - macro-F1(b): resample issues with
    replacement, rescore both predictors on the SAME resample each round.
    Says whether a gap is bigger than what reshuffling the test set produces."""
    import random

    def macro(pred: list[set[str]], rows: list[int]) -> float:
        return label_report(confusion((truth[r], pred[r]) for r in rows))["macro_f1"]

    n = len(truth)
    rng = random.Random(seed)
    diffs = sorted(
        macro(a, rows) - macro(b, rows)
        for rows in ([rng.randrange(n) for _ in range(n)] for _ in range(rounds))
    )
    everything = list(range(n))
    return {
        "diff": macro(a, everything) - macro(b, everything),
        "ci95": [diffs[int(0.025 * rounds)], diffs[int(0.975 * rounds) - 1]],
        "p_not_better": sum(d <= 0 for d in diffs) / rounds,
        "rounds": rounds,
    }


def selective_curve(
    truth: list[set[str]],
    suggested: list[set[str]],
    confidence: list[dict[str, float]],
    thresholds: Iterable[float],
) -> list[dict]:
    """Precision vs. coverage when a suggestion is shown only if its confidence
    is >= t. At each t: label precision and recall (micro, vs. maintainer labels),
    the share of issues still getting at least one suggestion, and the share of
    suggested labels kept."""
    total = sum(len(s) for s in suggested)
    curve = []
    for t in thresholds:
        kept = [
            {lbl for lbl in s if c[lbl] >= t} for s, c in zip(suggested, confidence, strict=True)
        ]
        tp = sum(len(k & tr) for k, tr in zip(kept, truth, strict=True))
        fp = sum(len(k - tr) for k, tr in zip(kept, truth, strict=True))
        fn = sum(len(tr - k) for k, tr in zip(kept, truth, strict=True))
        p, r, _ = prf(tp, fp, fn)
        curve.append(
            {
                "threshold": round(t, 2),
                "precision": round(p, 4),
                "recall": round(r, 4),
                "issue_coverage": round(sum(bool(k) for k in kept) / len(kept), 4),
                "labels_kept": round((tp + fp) / total, 4) if total else 0.0,
                "precision_ci95": [round(x, 4) for x in wilson(tp, tp + fp)],
            }
        )
    return curve


def operating_point(curve: list[dict], min_precision: float) -> dict | None:
    """The lowest threshold whose precision reaches `min_precision` (most coverage)."""
    return next((c for c in curve if c["precision"] >= min_precision), None)
