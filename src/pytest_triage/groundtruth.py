"""Duplicate ground truth: for each issue closed as a duplicate, which OLDER
issue(s) does it duplicate?

GitHub records only state_reason == "duplicate"; the target lives in a
maintainer's comment as "#1234". That makes this SILVER truth (machine-
extracted). 15 of 43 duplicates yield a usable target; the rest were closed
with no link, duplicate a Discussion, were "already fixed", or point forward
in time. The 3 pairs no retriever finds were checked by hand (2026-10-12):
all correct - they're root-cause duplicates with little word overlap.
"""

import re

import httpx

from pytest_triage.github import fetch_comments

MAINTAINERS = {"OWNER", "MEMBER", "COLLABORATOR"}


def is_duplicate(issue: dict) -> bool:
    return issue["state_reason"] == "duplicate" or "closed as duplicate" in issue["labels"]


def maintainer_text(comments: list[dict]) -> str:
    """Only maintainers decide duplicates; a reporter's "maybe #12?" doesn't count."""
    return "\n".join(c["body"] or "" for c in comments if c["author_association"] in MAINTAINERS)


def mentioned_numbers(text: str) -> list[int]:
    """Every #1234, as ints, deduplicated in first-seen order.
    `#issuecomment-99` isn't matched: \\d+ needs a digit right after the #."""
    return list(dict.fromkeys(int(n) for n in re.findall(r"#(\d+)", text)))


def targets_for(dup: dict, mentioned: list[int], by_number: dict[int, dict]) -> list[int]:
    """Mentioned numbers that are older issues in the snapshot (PRs aren't in it)."""
    return [
        n
        for n in mentioned
        if n in by_number and n != dup["number"] and by_number[n]["created_at"] < dup["created_at"]
    ]


def build_truth(client: httpx.Client, repo: str, issues: list[dict]) -> dict[int, list[int]]:
    by_number = {i["number"]: i for i in issues}
    truth = {}
    for dup in (i for i in issues if is_duplicate(i)):
        comments = fetch_comments(client, repo, dup["number"])
        found = targets_for(dup, mentioned_numbers(maintainer_text(comments)), by_number)
        if found:
            truth[dup["number"]] = found
    return truth
