"""What we predict, and what we grade on.

- `targets`: an issue's true labels, restricted to LABELS.
- `is_evaluable`: only triaged issues (any label at all). Unlabelled means
  "nobody looked", not "no labels apply" - scoring those would punish
  correct predictions.
- `split`: by creation date, so every test issue is newer than everything the
  system may learn from. A random split would leak the future into memory.
"""

from pytest_triage.config import LABELS, SPLIT_DATE


def targets(issue: dict) -> set[str]:
    return set(issue["labels"]) & set(LABELS)


def is_evaluable(issue: dict) -> bool:
    return bool(issue["labels"])


def split(issues: list[dict], date: str = SPLIT_DATE) -> tuple[list[dict], list[dict]]:
    """(memory, test). ISO-8601 UTC strings sort correctly as plain text."""
    memory = [i for i in issues if i["created_at"] < date]
    test = [i for i in issues if i["created_at"] >= date]
    return memory, test


def issue_text(issue: dict, max_body: int) -> str:
    """Title + the start of the body: what retrieval indexes and searches with."""
    return f"{issue['title']}\n{(issue['body'] or '')[:max_body]}"
