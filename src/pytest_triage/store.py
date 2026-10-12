"""The local issue snapshot: one JSON object per line, sorted by creation time."""

import json
import os
from collections.abc import Iterable
from pathlib import Path

from pytest_triage.config import ISSUES


def load_issues(path: Path = ISSUES) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `triage fetch` first.")
    with path.open() as f:
        return [json.loads(line) for line in f]


def save_issues(issues: Iterable[dict], path: Path = ISSUES) -> int:
    """Write atomically: a crash mid-write never leaves a half-written snapshot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    count = 0
    with tmp.open("w") as f:
        for issue in issues:
            f.write(json.dumps(issue) + "\n")
            count += 1
    os.replace(tmp, path)
    return count


def merge(old: Iterable[dict], new: Iterable[dict]) -> list[dict]:
    """Newer copies replace older ones by issue number; result sorted by created_at."""
    by_number = {i["number"]: i for i in old}
    by_number.update({i["number"]: i for i in new})
    return sorted(by_number.values(), key=lambda i: (i["created_at"], i["number"]))


def last_updated(issues: Iterable[dict]) -> str | None:
    """The newest updated_at in the snapshot: where an incremental fetch resumes."""
    stamps = [i["updated_at"] for i in issues if i.get("updated_at")]
    return max(stamps) if stamps else None
