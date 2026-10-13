"""The full pipeline for one issue: labels, missing info, duplicates, a reply draft.

A fixed pipeline, not a free-roaming agent: every issue gets the same 3 LLM
calls (assess, expand, verify), so cost and latency are predictable and each
step is evaluated on its own. The reply is a TEMPLATE filled from structured
outputs - the model never writes prose that reaches a maintainer.
"""

import time

from pytest_triage import config, guard, llm, policy
from pytest_triage.classify import assess
from pytest_triage.config import DEDUPE_MODEL, LABEL_MODEL, LABEL_PROMPT
from pytest_triage.dedupe import find_duplicates
from pytest_triage.search import IssueIndex


def _reply_worthy(output: str) -> bool:
    """Only outputs precise enough to say to a reporter go in the reply draft."""
    return config.MEASURED_PRECISION[output] >= config.REPLY_MIN_PRECISION


def draft_reply(report: dict) -> str:
    """Suggested comment to the reporter, built only from enum values and gated
    by measured precision (config.REPLY_MIN_PRECISION). Outputs below the bar -
    today both missing-info and duplicates - appear only in the maintainer view."""
    lines = ["Thanks for the report!"]
    if report["missing"] and _reply_worthy("missing"):
        lines.append("\nTo help us investigate, could you add:")
        lines += [f"- {item}" for item in report["missing"]]
    if len(lines) == 1:
        return ""  # nothing useful to say: don't draft a comment at all
    return "\n".join(lines)


def triage_issue(
    issue: dict,
    index: IssueIndex,
    labelled: IssueIndex,
    model: str | None = None,
    prompt: str = LABEL_PROMPT,
) -> dict:
    """`index`: all issues (duplicate search). `labelled`: triaged issues (few-shot examples).
    Both are searched only for issues created before this one."""
    spent_before = llm.spent()
    started = time.perf_counter()
    timings = {}

    t = time.perf_counter()
    label_model, dedupe_model = model or LABEL_MODEL, model or DEDUPE_MODEL
    assessment = assess(issue, labelled, label_model, before=issue["created_at"], prompt=prompt)
    timings["assess_s"] = time.perf_counter() - t

    t = time.perf_counter()
    dupes = find_duplicates(issue, index, expand=True, model=dedupe_model)
    timings["duplicates_s"] = time.perf_counter() - t

    report = {
        "number": issue["number"],
        "title": issue["title"],
        "labels": sorted(assessment["labels"]),
        "missing": assessment["missing"],
        "duplicates": dupes["duplicates"],
        "duplicate_verdicts": dupes["verdicts"],
        "candidates": dupes["candidates"],
        "injection_signals": guard.injection_signals(f"{issue['title']}\n{issue['body'] or ''}"),
        # Candidates are stranger-written text too, and they reach the verifier's prompt.
        "candidate_injection_signals": {
            n: signals
            for n in dupes["candidates"]
            if (c := index.get(n))
            and (signals := guard.injection_signals(f"{c['title']}\n{c['body'] or ''}"))
        },
        "models": {"labels": label_model, "duplicates": dedupe_model, "prompt": prompt},
    }
    report = policy.apply_policy(report)
    report["reply_draft"] = draft_reply(report)
    created_at_of = {n: c["created_at"] for n in dupes["candidates"] if (c := index.get(n))}
    report["warnings"] = policy.check_report(report, issue, created_at_of)
    report["trace"] = {
        "latency_s": round(time.perf_counter() - started, 2),
        **{k: round(v, 2) for k, v in timings.items()},
        "cost_usd": round(llm.spent() - spent_before, 6),  # 0 when served from cache
    }
    return report
