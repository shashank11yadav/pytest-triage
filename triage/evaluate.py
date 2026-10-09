"""Score label predictors on the test set.

A predictor is any function: issue dict -> set of labels it thinks apply.
Every approach we build (keywords, k-NN, Gemini) gets scored by the same
code, so the numbers are comparable.

Run it:
    uv run python -m triage.evaluate
"""

from collections import Counter
from collections.abc import Callable

from triage.dataset import LABELS, is_evaluable, load_issues, split, targets

Predictor = Callable[[dict], set[str]]


# --- metrics ----------------------------------------------------------------

def count(predict: Predictor, issues: list[dict]) -> dict[str, Counter]:
    """For each label, count true positives, false positives, false negatives."""
    counts = {label: Counter() for label in LABELS}
    for issue in issues:
        truth = targets(issue)
        guess = predict(issue) & set(LABELS)  # ignore anything we don't score
        for label in LABELS:
            if label in guess and label in truth:
                counts[label]["tp"] += 1
            elif label in guess:
                counts[label]["fp"] += 1
            elif label in truth:
                counts[label]["fn"] += 1
    return counts


def scores(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    # TODO(you): return (precision, recall, f1) as floats.
    #   precision = tp / (tp + fp)
    #   recall    = tp / (tp + fn)
    #   f1        = 2 * precision * recall / (precision + recall)
    #
    #   The trap: any of those denominators can be 0. A predictor that never
    #   says "bug" has tp + fp == 0. Python raises ZeroDivisionError.
    #   Rule: if a denominator is 0, that score is 0.0.
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / \
        (precision + recall) if precision + recall > 0 else 0.0
    return (precision, recall, f1)


def report(name: str, counts: dict[str, Counter]) -> float:
    """Print a per-label table; return macro-F1 (plain average of label F1s)."""
    print(f"\n== {name}")
    print(f"{'label':<22}{'P':>6}{'R':>6}{'F1':>6}{'support':>9}")
    f1s = []
    for label in LABELS:
        c = counts[label]
        p, r, f1 = scores(c["tp"], c["fp"], c["fn"])
        # how many test issues really have this label
        support = c["tp"] + c["fn"]
        print(f"{label:<22}{p:>6.2f}{r:>6.2f}{f1:>6.2f}{support:>9}")
        f1s.append(f1)
    macro = sum(f1s) / len(f1s)
    print(f"{'macro-F1':<22}{'':>12}{macro:>6.2f}")
    return macro


# --- baselines --------------------------------------------------------------

def predict_everything(issue: dict) -> set[str]:
    """Say every label applies. Perfect recall, useless precision."""
    return set(LABELS)


def make_majority(memory: list[dict]) -> Predictor:
    """Always predict the single most common label - counted on MEMORY only."""
    label, _ = Counter(l for i in memory for l in targets(i)).most_common(1)[0]
    return lambda issue: {label}


# Words that suggest each label, matched as case-insensitive substrings of the
# TITLE only. Chosen from MEMORY issues (never test - that would be leakage):
# for each title word, the share of memory issues containing it that carry the
# label, keeping words with high share and 8+ occurrences.
# Title-only beat title+body on memory (macro-F1 0.36 vs 0.30): bodies are full
# of code and tracebacks where "fixture", "config", "collect" appear everywhere.
# Known substring traps we accepted (measured, barely matters): "doc" matches
# "docker", "ini" matches "init", "how" matches "show".
KEYWORDS: dict[str, list[str]] = {
    "type: bug": ["crash", "internal", "breaks", "broken", "valueerror", "unicode", "ascii"],
    "type: question": ["question", "how"],
    "type: enhancement": ["better", "warn", "ability"],
    "type: proposal": ["allow", "provide", "feature", "disable"],
    # "doc" covers docs, document(ation)
    "type: docs": ["doc", "rst", "changelog"],
    "topic: fixtures": ["fixture", "scope", "autouse"],
    "topic: reporting": ["summary", "terminal", "verbose", "color", "short"],
    "topic: collection": ["collect", "importlib", "rootdir", "discovery", "pyargs"],
    "topic: config": ["ini", "options", "config"],
    # parametrize/-d/-ation
    "topic: parametrize": ["parametriz", "ids", "indirect"],
}


def keyword_predict(issue: dict) -> set[str]:
    # TODO(you): return every label whose keywords appear in the issue's
    #   TITLE (not the body - see the measurement above KEYWORDS).
    #   - case-insensitive ("Typo" must match "typo")
    #   - a label is predicted if ANY of its keywords appears
    match = set()
    title = issue["title"].lower() if issue.get("title") else ""
    for label in KEYWORDS:
        for keyword in KEYWORDS[label]:
            if keyword in title:
                match.add(label)
    return match


# --- run --------------------------------------------------------------------

def main() -> None:
    issues = [i for i in load_issues() if is_evaluable(i)]
    memory, test = split(issues)
    print(f"scoring on {len(test)} test issues")

    report("everything", count(predict_everything, test))
    report("majority", count(make_majority(memory), test))
    report("keywords", count(keyword_predict, test))


if __name__ == "__main__":
    main()
