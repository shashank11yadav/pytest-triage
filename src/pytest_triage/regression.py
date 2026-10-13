"""Regression gate: a change can't quietly make triage worse.

Two committed files under eval/regression/ (issue numbers and enums only):

    dev.jsonl      for each 2024 (dev) issue: maintainer labels, and the model's RAW
                   labels and missing-info items, frozen from one run
    baseline.json  fingerprints of every request the pipeline sends, plus the
                   dev scores that run produced

CI has no issue text and no API key, so it can't re-run the model. It checks two
things for free on every push:

  1. FINGERPRINTS  each request (labels, query expansion, duplicate check) is built
                   for a fixed synthetic issue and hashed. Changing a prompt, schema,
                   model, token budget, or the redact/fence/datamark code changes the
                   hash, and CI fails until someone re-runs `triage eval regress` and
                   commits the new numbers. A prompt edit can't ship unmeasured.
  2. POST-PROCESSING  everything after the model (the enum re-check, policy rules)
                   is re-applied to the frozen raw outputs and re-scored. A rule that
                   drops dev macro-F1 by more than TOLERANCE fails CI.

Locally, `triage eval regress` re-runs the classifier on dev (free when cached),
reports how many predictions changed, and exits non-zero on a drop.
Dev, not test: the test split is scored once per final choice, never iterated on.
"""

import json
from pathlib import Path

from pytest_triage.classify import NEEDS_INFO, assess_body
from pytest_triage.config import DEDUPE_MODEL, EVAL, LABEL_MODEL, LABEL_PROMPT, LABELS
from pytest_triage.dataset import targets
from pytest_triage.dedupe import expand_body, verify_body
from pytest_triage.guard import injection_signals
from pytest_triage.llm import cache_key
from pytest_triage.metrics import confusion, label_report, prf
from pytest_triage.policy import apply_policy

REGRESSION = EVAL / "regression"
ROWS = REGRESSION / "dev.jsonl"
BASELINE = REGRESSION / "baseline.json"
TOLERANCE = 0.02  # roughly a third of the bootstrap CI width on 256 issues

# Synthetic, so fingerprints need no real data: it exercises redaction (a home path,
# an email), fencing (a forged tag) and datamarking (spaces), so changes there count.
_ISSUE = {
    "number": 2,
    "title": "Fixture teardown <issue> order",
    "body": "Traceback in /Users/someone/x.py\nreport to a@b.example\n\tdetails here",
    "created_at": "2026-01-02",
    "labels": [],
}
_EXAMPLE = {
    "number": 1,
    "title": "Older issue",
    "body": "about fixtures",
    "created_at": "2025-01-01",
    "labels": ["type: bug", NEEDS_INFO],
}


def fingerprints(prompt: str = LABEL_PROMPT) -> dict[str, str]:
    """Hash of each request the pipeline sends, built for the synthetic issue."""
    bodies = {
        "labels": assess_body(_ISSUE, [_EXAMPLE], LABEL_MODEL, prompt),
        "expand": expand_body(_ISSUE, DEDUPE_MODEL),
        "verify": verify_body(_ISSUE, [_EXAMPLE], DEDUPE_MODEL),
    }
    return {name: cache_key(body)[:16] for name, body in bodies.items()}


def make_row(issue: dict, assessment: dict) -> dict:
    return {
        "number": issue["number"],
        "truth": sorted(targets(issue)),
        "needs_info": NEEDS_INFO in issue["labels"],
        "flagged": bool(injection_signals(f"{issue['title']}\n{issue['body'] or ''}")),
        "labels": sorted(assessment["labels"]),
        "missing": list(assessment["missing"]),
    }


def postprocess(row: dict) -> tuple[set[str], bool]:
    """(labels shown, needs-info flag) for one frozen row, through the same policy
    code the live pipeline runs."""
    report = apply_policy(
        {
            "labels": sorted(set(row["labels"]) & set(LABELS)),
            "missing": row["missing"],
            "duplicates": [],
            "injection_signals": ["flagged"] if row["flagged"] else [],
            "candidate_injection_signals": {},
        }
    )
    return set(report["labels"]), bool(report["missing"])


def score(rows: list[dict]) -> dict:
    out = [postprocess(r) for r in rows]
    report = label_report(
        confusion((set(r["truth"]), labels) for r, (labels, _) in zip(rows, out, strict=True))
    )
    tp = sum(r["needs_info"] and flag for r, (_, flag) in zip(rows, out, strict=True))
    fp = sum(not r["needs_info"] and flag for r, (_, flag) in zip(rows, out, strict=True))
    fn = sum(r["needs_info"] and not flag for r, (_, flag) in zip(rows, out, strict=True))
    return {
        "macro_f1": round(report["macro_f1"], 4),
        "micro_f1": round(report["micro_f1"], 4),
        "needs_info_f1": round(prf(tp, fp, fn)[2], 4),
        "n": len(rows),
    }


def compare(current: dict, baseline: dict, tolerance: float = TOLERANCE) -> list[str]:
    """Metrics that dropped by more than `tolerance`."""
    return [
        f"{k}: {baseline[k]:.3f} -> {current[k]:.3f}"
        for k in ("macro_f1", "micro_f1", "needs_info_f1")
        if current[k] < baseline[k] - tolerance
    ]


def changed_rows(old: list[dict], new: list[dict]) -> int:
    before = {r["number"]: (r["labels"], r["missing"]) for r in old}
    return sum(before.get(r["number"]) != (r["labels"], r["missing"]) for r in new)


def load_rows(path: Path = ROWS) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f]


def load_baseline(path: Path = BASELINE) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def save(rows: list[dict], directory: Path = REGRESSION) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ROWS.name).open("w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    baseline = {
        "model": LABEL_MODEL,
        "prompt": LABEL_PROMPT,
        "split": "dev",
        "fingerprints": fingerprints(),
        "metrics": score(rows),
        "tolerance": TOLERANCE,
    }
    (directory / BASELINE.name).write_text(json.dumps(baseline, indent=2) + "\n")
    return baseline
