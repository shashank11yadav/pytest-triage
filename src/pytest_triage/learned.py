"""A supervised label classifier trained on maintainers' own past labels.

Why: issue-classification competitions (NLBSE 2022-24) were won by small models
trained on labelled issues, not by prompting, and VS Code's triage bot uses a
trained classifier with per-category confidence thresholds. We already have an
embedding for every issue, so one logistic regression per label costs nothing
to train and - unlike the LLM, whose Interactions API exposes no logprobs -
gives a probability per label that can be thresholded.

Protocol (no decision ever looks at the test split):
    choose C and thresholds  fit < 2023, tune on 2023, score 2024 (dev)
    final test run            fit < 2024, tune on 2024, score 2025+ (test)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from pytest_triage.config import LABELS
from pytest_triage.dataset import targets
from pytest_triage.metrics import Predictor, prf

THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


def label_matrix(issues: list[dict]) -> np.ndarray:
    """(n issues, n labels) 0/1 matrix of maintainer labels."""
    return np.array([[label in targets(i) for label in LABELS] for i in issues], dtype=int)


class LabelModel:
    """One balanced logistic regression per label, plus a tuned threshold each."""

    def __init__(self, C: float = 1.0):
        self.C = C
        self.models: list[LogisticRegression] = []
        self.thresholds = np.full(len(LABELS), 0.5)

    def fit(self, vectors: np.ndarray, issues: list[dict]) -> "LabelModel":
        y = label_matrix(issues)
        self.models = [
            LogisticRegression(C=self.C, class_weight="balanced", max_iter=2000).fit(
                vectors, y[:, j]
            )
            if len(set(y[:, j])) == 2
            else _Constant(float(y[0, j]))  # a label never (or always) seen: no model to fit
            for j in range(len(LABELS))
        ]
        return self

    def proba(self, vectors: np.ndarray) -> np.ndarray:
        return np.column_stack([m.predict_proba(vectors)[:, 1] for m in self.models])

    def tune_thresholds(self, vectors: np.ndarray, issues: list[dict]) -> "LabelModel":
        """Per label, the threshold that maximises F1 on a HELD-OUT set."""
        p, y = self.proba(vectors), label_matrix(issues)
        for j in range(len(LABELS)):
            best = max(THRESHOLD_GRID, key=lambda t: (_f1(y[:, j], p[:, j] >= t), -t))
            self.thresholds[j] = best
        return self

    def predict(self, vectors: np.ndarray) -> list[set[str]]:
        hits = self.proba(vectors) >= self.thresholds
        return [{LABELS[j] for j in np.flatnonzero(row)} for row in hits]


class _Constant:
    """Stand-in for a label with only one class in the training data."""

    def __init__(self, p: float):
        self.p = p

    def predict_proba(self, vectors: np.ndarray) -> np.ndarray:
        return np.tile([1 - self.p, self.p], (len(vectors), 1))


def _f1(y: np.ndarray, guess: np.ndarray) -> float:
    tp = int((guess & (y == 1)).sum())
    return prf(tp, int((guess & (y == 0)).sum()), int((~guess & (y == 1)).sum()))[2]


def train(
    fit_issues: list[dict],
    fit_vectors: np.ndarray,
    tune_issues: list[dict],
    tune_vectors: np.ndarray,
    Cs: tuple[float, ...] = (0.3, 1.0, 3.0),
) -> LabelModel:
    """Pick C by macro-F1 on the tuning set (after tuning thresholds there)."""
    from pytest_triage.metrics import confusion, label_report

    def score(model: LabelModel) -> float:
        preds = model.predict(tune_vectors)
        return label_report(confusion(zip((targets(i) for i in tune_issues), preds, strict=True)))[
            "macro_f1"
        ]

    candidates = [
        LabelModel(C).fit(fit_vectors, fit_issues).tune_thresholds(tune_vectors, tune_issues)
        for C in Cs
    ]
    return max(candidates, key=score)


def as_predictor(model: LabelModel, vectors_by_number: dict[int, np.ndarray]) -> Predictor:
    def predict(issue: dict) -> set[str]:
        return model.predict(vectors_by_number[issue["number"]][None, :])[0]

    return predict


def proba_lookup(model: LabelModel, vectors_by_number: dict[int, np.ndarray]):
    """issue -> {label: probability}, for ensembles."""

    def lookup(issue: dict) -> dict[str, float]:
        p = model.proba(vectors_by_number[issue["number"]][None, :])[0]
        return dict(zip(LABELS, p.tolist(), strict=True))

    return lookup


def combine(
    llm: set[str],
    probs: dict[str, float],
    variant: str,
    route: dict[str, str] | None = None,
    learned: set[str] | None = None,
) -> set[str]:
    """Ways to combine LLM labels with the classifier's probabilities.

    gate:G    keep an LLM label only if the classifier gives it >= G (cuts extras)
    boost:B   add any label the classifier is >= B sure of (adds misses)
    gate:G+boost:B  both
    route     per label, trust whichever source scored higher on dev
    """
    if variant == "route":
        assert route is not None and learned is not None
        return {lbl for lbl in LABELS if lbl in (llm if route[lbl] == "llm" else learned)}
    out = set(llm)
    for part in variant.split("+"):
        kind, value = part.split(":")
        if kind == "gate":
            out = {lbl for lbl in out if probs[lbl] >= float(value)}
        elif kind == "boost":
            out |= {lbl for lbl, p in probs.items() if p >= float(value)}
        else:
            raise ValueError(f"unknown variant part {part!r}")
    return out
