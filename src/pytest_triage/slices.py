"""Where does the classifier fail? Scores broken down by what the issue looks like.

A single macro-F1 hides that a tool can be good on tidy bug reports and useless
on one-line complaints. Each slice is a property of the issue text a maintainer
would recognise; slices overlap. Micro-F1 is used because per-label macro-F1 is
noise on slices of 30 issues, and every slice gets a bootstrap interval.
"""

import random
import re
from collections.abc import Callable

from pytest_triage.dataset import targets
from pytest_triage.metrics import confusion, label_report


def _body(issue: dict) -> str:
    return issue["body"] or ""


TRACEBACK = re.compile(r"Traceback \(most recent call last\)|^E {2,}\S", re.MULTILINE)
QUESTION = re.compile(r"^(how|why|is|can|what|does|should|where)\b", re.IGNORECASE)

SLICES: dict[str, Callable[[dict], bool]] = {
    "all": lambda i: True,
    "body < 400 chars": lambda i: len(_body(i)) < 400,
    "body 400-2000 chars": lambda i: 400 <= len(_body(i)) <= 2000,
    "body > 2000 chars": lambda i: len(_body(i)) > 2000,
    "has a traceback": lambda i: bool(TRACEBACK.search(_body(i))),
    "no traceback": lambda i: not TRACEBACK.search(_body(i)),
    "used the bug template": lambda i: "detailed description of the bug" in _body(i),
    "has a code block": lambda i: "```" in _body(i),
    "no code block": lambda i: "```" not in _body(i),
    "title is a question": lambda i: (
        i["title"].rstrip().endswith("?") or bool(QUESTION.match(i["title"]))
    ),
    "maintainers gave no type: label": lambda i: not any(t.startswith("type:") for t in targets(i)),
}


def micro_f1(truth: list[set[str]], pred: list[set[str]]) -> float:
    return label_report(confusion(zip(truth, pred, strict=True)))["micro_f1"]


def bootstrap_ci(
    truth: list[set[str]], pred: list[set[str]], rounds: int = 1000, seed: int = 0
) -> list[float]:
    rng, n = random.Random(seed), len(truth)
    scores = sorted(
        micro_f1([truth[r] for r in rows], [pred[r] for r in rows])
        for rows in ([rng.randrange(n) for _ in range(n)] for _ in range(rounds))
    )
    return [scores[int(0.025 * rounds)], scores[int(0.975 * rounds) - 1]]


def slice_report(issues: list[dict], predictions: dict[str, list[set[str]]]) -> dict:
    """{slice: {"n", predictor: {"micro_f1", "ci95"}}}; predictions are aligned with issues."""
    out = {}
    for name, member in SLICES.items():
        rows = [k for k, issue in enumerate(issues) if member(issue)]
        if not rows:
            continue
        truth = [targets(issues[k]) for k in rows]
        out[name] = {"n": len(rows)}
        for predictor, preds in predictions.items():
            pred = [preds[k] for k in rows]
            out[name][predictor] = {
                "micro_f1": round(micro_f1(truth, pred), 4),
                "ci95": [round(x, 4) for x in bootstrap_ci(truth, pred)],
            }
    return out
