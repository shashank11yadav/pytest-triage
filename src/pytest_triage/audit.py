"""Label audit: are the model's "false positive" labels really wrong?

Maintainers label minimally: on the test set they apply 1.01 of our 10 labels per
issue, while the model predicts 1.66. A label the model adds that maintainers
didn't is therefore not necessarily wrong - it may just be one nobody applied.

Two raters judge the same random sample of such labels as correct / wrong /
unsure, blind to each other. The share judged correct adjusts the measured
precision, and Cohen's kappa says how far the raters agree (if they don't, the
adjustment means little).

    eval/audit/sample.jsonl            sampled (issue number, label) pairs
    eval/audit/sample_meta.json        predictor, seed, and its TP/FP label counts
    eval/audit/ratings_<rater>.jsonl   one verdict per pair
Only numbers, labels and verdicts are stored - never issue text.
"""

import json
import random
from collections.abc import Callable
from pathlib import Path

from pytest_triage.config import EVAL
from pytest_triage.dataset import targets
from pytest_triage.metrics import Predictor, wilson

AUDIT = EVAL / "audit"
SAMPLE = AUDIT / "sample.jsonl"
META = AUDIT / "sample_meta.json"
VERDICTS = {"c": "correct", "w": "wrong", "u": "unsure"}
ISSUE_URL = "https://github.com/pytest-dev/pytest/issues/"


def false_positives(issues: list[dict], predict: Predictor) -> tuple[list[dict], int]:
    """(every predicted label maintainers didn't apply, number of true positives)."""
    fps, tp = [], 0
    for issue in issues:
        guess, truth = predict(issue), targets(issue)
        tp += len(guess & truth)
        fps += [{"number": issue["number"], "label": label} for label in sorted(guess - truth)]
    return fps, tp


def draw_sample(fps: list[dict], n: int, seed: int = 11) -> list[dict]:
    pool = sorted(fps, key=lambda f: (f["number"], f["label"]))
    return random.Random(seed).sample(pool, min(n, len(pool)))


def write_sample(sample: list[dict], meta: dict, directory: Path = AUDIT) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / SAMPLE.name).write_text("".join(json.dumps(s) + "\n" for s in sample))
    (directory / META.name).write_text(json.dumps(meta, indent=2) + "\n")


def load_sample(directory: Path = AUDIT) -> tuple[list[dict], dict]:
    sample = [json.loads(line) for line in (directory / SAMPLE.name).read_text().splitlines()]
    return sample, json.loads((directory / META.name).read_text())


def ratings_path(rater: str, directory: Path = AUDIT) -> Path:
    return directory / f"ratings_{rater}.jsonl"


def load_ratings(rater: str, directory: Path = AUDIT) -> dict[tuple[int, str], str]:
    path = ratings_path(rater, directory)
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return {(r["number"], r["label"]): r["verdict"] for r in rows}


def add_rating(rater: str, item: dict, verdict: str, directory: Path = AUDIT) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with ratings_path(rater, directory).open("a") as f:
        f.write(json.dumps({"number": item["number"], "label": item["label"], "verdict": verdict}))
        f.write("\n")


def render_item(issue: dict, label: str, definition: str, position: str) -> str:
    body = " ".join((issue["body"] or "").split())[:900]
    return (
        f"\n[{position}]  {ISSUE_URL}{issue['number']}\n"
        f"Title: {issue['title']}\n"
        f"Maintainers applied: {', '.join(issue['labels']) or '(none)'}\n"
        f"Model ALSO suggested: >>> {label} <<<  ({definition})\n\n{body}\n"
    )


def rate_interactively(
    rater: str,
    sample: list[dict],
    by_number: dict[int, dict],
    definitions: dict[str, str],
    ask: Callable[[str], str] = input,
    show: Callable[[str], None] = print,
    directory: Path = AUDIT,
) -> int:
    """Rate unrated items; resumable. Returns how many were rated this session."""
    done = load_ratings(rater, directory)
    todo = [s for s in sample if (s["number"], s["label"]) not in done]
    show(
        f"{len(done)} already rated, {len(todo)} to go. "
        "Judge only whether the shown LABEL fits the issue."
    )
    rated = 0
    for k, item in enumerate(todo, start=len(done) + 1):
        show(
            render_item(
                by_number[item["number"]],
                item["label"],
                definitions[item["label"]],
                f"{k}/{len(sample)}",
            )
        )
        while (
            answer := ask("Is this label correct? [c]orrect / [w]rong / [u]nsure / [q]uit: ")
            .strip()
            .lower()[:1]
        ) not in (*VERDICTS, "q"):
            show("Please type c, w, u or q.")
        if answer == "q":
            break
        add_rating(rater, item, VERDICTS[answer], directory)
        rated += 1
    return rated


def adjusted_precision(tp: int, fp: int, verdicts: list[str]) -> dict:
    """Precision if the audited share of 'false positives' were really correct.
    `unsure` verdicts are excluded. The interval comes from the share's Wilson CI."""
    correct, wrong = verdicts.count("correct"), verdicts.count("wrong")
    judged = correct + wrong
    share = correct / judged if judged else 0.0
    lo, hi = wilson(correct, judged)
    total = tp + fp
    return {
        "measured_precision": tp / total if total else 0.0,
        "share_of_fps_correct": share,
        "share_ci95": [lo, hi],
        "adjusted_precision": (tp + fp * share) / total if total else 0.0,
        "adjusted_ci95": [(tp + fp * lo) / total, (tp + fp * hi) / total] if total else [0, 0],
        "rated": len(verdicts),
        "unsure": verdicts.count("unsure"),
    }


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Agreement between two raters beyond chance: 1 perfect, 0 chance, <0 worse."""
    n = len(a)
    if n == 0:
        return 0.0
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    categories = set(a) | set(b)
    expected = sum((a.count(c) / n) * (b.count(c) / n) for c in categories)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)
