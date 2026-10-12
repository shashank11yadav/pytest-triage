"""Cheap label predictors. Anything that costs money has to beat these.

Every predictor is a function: issue -> set of labels. Anything learned (the
majority label, keywords, neighbours) comes from MEMORY issues only.
"""

import re
from collections import Counter

from pytest_triage.config import LABELS
from pytest_triage.dataset import targets
from pytest_triage.metrics import Predictor
from pytest_triage.search import IssueIndex


def predict_everything(issue: dict) -> set[str]:
    """Perfect recall, useless precision: shows why recall alone means nothing."""
    return set(LABELS)


def make_majority(memory: list[dict]) -> Predictor:
    label, _ = Counter(lbl for i in memory for lbl in targets(i)).most_common(1)[0]
    return lambda issue: {label}


def _title_words(issue: dict) -> set[str]:
    return set(re.findall(r"[a-z_]{3,}", (issue["title"] or "").lower()))


def mine_keywords(
    memory: list[dict],
    min_issues: int = 15,
    min_hits: int = 8,
    min_share: float = 0.4,
    top: int = 8,
) -> dict[str, list[str]]:
    """For each label: title words that mostly appear on that label's issues.

    share = (memory issues with the word AND the label) / (memory issues with the word).
    Titles only: on memory, title-only matching beat title+body (macro-F1 0.36 vs
    0.30) - bodies are full of code where "fixture", "config", "collect" are everywhere.
    """
    docs = [(_title_words(i), targets(i)) for i in memory]
    frequency = Counter(w for words, _ in docs for w in words)
    keywords = {}
    for label in LABELS:
        hits = Counter(w for words, labels in docs if label in labels for w in words)
        ranked = sorted(
            (
                (hits[w] / frequency[w], w)
                for w in hits
                if frequency[w] >= min_issues and hits[w] >= min_hits
            ),
            reverse=True,
        )
        keywords[label] = [w for share, w in ranked if share >= min_share][:top]
    return keywords


def make_keyword_predictor(keywords: dict[str, list[str]]) -> Predictor:
    def predict(issue: dict) -> set[str]:
        words = _title_words(issue)
        return {label for label, kws in keywords.items() if words & set(kws)}

    return predict


def make_knn(index: IssueIndex, k: int = 10, min_share: float = 0.3) -> Predictor:
    """Label vote of the k most similar labelled memory issues. Free, no LLM.

    `index` must contain only labelled MEMORY issues, so no test label leaks in.
    """

    def predict(issue: dict) -> set[str]:
        neighbours = [index.get(n) for n in index.similar(issue, k=k, before="9999")]
        votes = Counter(lbl for n in neighbours if n for lbl in targets(n))
        return {label for label, count in votes.items() if count / k >= min_share}

    return predict
