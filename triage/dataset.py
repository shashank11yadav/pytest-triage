"""Turn the raw snapshot into something we can evaluate on.

Two decisions live here:
  1. LABELS - which labels we try to predict.
  2. split() - which issues the system may learn from ("memory") and which
     we grade it on ("test"). The split is by date, so the test set is always
     newer than anything in memory, just like real life.

Run it:
    uv run python -m triage.dataset
"""

import json
from collections import Counter
from pathlib import Path

SNAPSHOT = Path("data/issues.jsonl")
SPLIT_DATE = "2025-01-01"  # created before -> memory, on/after -> test

# The 5 most common `type:` and 5 most common `topic:` labels in the snapshot
# (counted 2026-10-08). Each has 184+ examples; rarer labels are too thin to
# learn from or score fairly.
LABELS = (
    "type: bug", "type: question", "type: enhancement", "type: proposal", "type: docs",
    "topic: fixtures", "topic: reporting", "topic: collection", "topic: config",
    "topic: parametrize",
)


def load_issues(path: Path = SNAPSHOT) -> list[dict]:
    """Read the JSONL snapshot written by triage.fetch."""
    with path.open() as f:
        return [json.loads(line) for line in f]


def is_evaluable(issue: dict) -> bool:
    """True if a maintainer has triaged this issue (it has ANY label at all).

    An unlabelled issue means "nobody looked", not "no labels apply".
    Scoring it as "correct answer: nothing" would punish right predictions.
    """
    return bool(issue["labels"])


def targets(issue: dict) -> set[str]:
    # TODO(you): the issue's "correct answers": the labels it has that are
    #   also in LABELS. Ignore every other label ("good first issue",
    #   "plugin: xdist", ...). Return a set.
    #   Expected: labels ["type: bug", "good first issue", "topic: fixtures"]
    #             -> {"type: bug", "topic: fixtures"}
    #   Hint: Python sets have an operator for "items in both".
    return set(issue.get("labels")) & set(LABELS)


def split(issues: list[dict], date: str = SPLIT_DATE) -> tuple[list[dict], list[dict]]:
    # TODO(you): return (memory, test).
    #   memory = issues created BEFORE `date`, test = created ON or AFTER it.
    #   Keep the original order inside each list.
    #
    #   `created_at` looks like "2024-12-31T23:59:59Z". You can compare these
    #   strings directly with < and >= - no date parsing needed. Before you
    #   write it, work out WHY plain text comparison gives the right answer
    #   here. (When would it NOT work?)
    memory = [issue for issue in issues if issue.get("created_at") < date]
    test = [issue for issue in issues if issue.get("created_at") >= date]
    return (memory, test)


def main() -> None:
    issues = [i for i in load_issues() if is_evaluable(i)]
    memory, test = split(issues)
    print(
        f"evaluable issues: {len(issues)}  (memory {len(memory)}, test {len(test)})\n")

    memory_counts = Counter(label for i in memory for label in targets(i))
    test_counts = Counter(label for i in test for label in targets(i))
    print(f"{'label':<22}{'memory':>8}{'test':>8}")
    for label in LABELS:
        print(f"{label:<22}{memory_counts[label]:>8}{test_counts[label]:>8}")


if __name__ == "__main__":
    main()
