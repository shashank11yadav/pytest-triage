"""The full pipeline for one issue: labels, missing info, duplicates, a reply draft.

A fixed pipeline, not a free-roaming agent: every issue gets the same 3 LLM
calls (assess, expand, verify), so cost and latency are predictable and each
step is evaluated on its own. The reply is a TEMPLATE filled from structured
outputs - the model never writes prose that reaches a maintainer.
"""

import time

from pytest_triage import guard, llm
from pytest_triage.classify import assess
from pytest_triage.config import WORKHORSE
from pytest_triage.dedupe import find_duplicates
from pytest_triage.search import IssueIndex


def draft_reply(report: dict) -> str:
    """Suggested maintainer comment. Built only from enums and issue numbers."""
    lines = ["Thanks for the report!"]
    dupes = report["duplicates"]
    if dupes:
        refs = ", ".join(f"#{n}" for n in dupes)
        lines.append(f"\nThis looks like it may be the same problem as {refs}. Could you check?")
    if report["missing"]:
        lines.append("\nTo help us investigate, could you add:")
        lines += [f"- {item}" for item in report["missing"]]
    if len(lines) == 1:
        return ""  # nothing useful to say: don't draft a comment at all
    return "\n".join(lines)


def triage_issue(
    issue: dict, index: IssueIndex, labelled: IssueIndex, model: str = WORKHORSE
) -> dict:
    """`index`: all issues (duplicate search). `labelled`: triaged issues (few-shot examples).
    Both are searched only for issues created before this one."""
    spent_before = llm.spent()
    started = time.perf_counter()
    timings = {}

    t = time.perf_counter()
    assessment = assess(issue, labelled, model, before=issue["created_at"])
    timings["assess_s"] = time.perf_counter() - t

    t = time.perf_counter()
    dupes = find_duplicates(issue, index, expand=True, model=model)
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
        "model": model,
    }
    report["reply_draft"] = draft_reply(report)
    report["trace"] = {
        "latency_s": round(time.perf_counter() - started, 2),
        **{k: round(v, 2) for k, v in timings.items()},
        "cost_usd": round(llm.spent() - spent_before, 6),  # 0 when served from cache
    }
    return report
