"""Build duplicate ground truth: for each issue closed as a duplicate, which
OLDER issue(s) does it duplicate?

GitHub only records state_reason == "duplicate". The target lives in a
maintainer's comment as "#1234", in many phrasings ("duplicate of #6492",
"This is #8741 plus #8079", ...). We extract those numbers automatically, so
this is SILVER ground truth (machine-extracted), not gold (human-checked).

Output: eval/duplicates.json - issue numbers only, safe to commit.

Run it:
    GITHUB_TOKEN=$(gh auth token) uv run python -m triage.dupes

Docs used (checked 2026-10-09):
    issue comments  https://docs.github.com/en/rest/issues/comments#list-issue-comments
"""

import json
import os
import re
from pathlib import Path

from triage.dataset import load_issues
from triage.fetch import REPO, get_page, make_client

OUT = Path("eval/duplicates.json")

# `author_association` on each comment says who wrote it. Only maintainers
# decide what's a duplicate - a reporter writing "maybe related to #12" is not.
MAINTAINERS = {"OWNER", "MEMBER", "COLLABORATOR"}


def is_duplicate(issue: dict) -> bool:
    """Closed with GitHub's "duplicate" reason, or carrying pytest's old label."""
    return issue["state_reason"] == "duplicate" or "closed as duplicate" in issue["labels"]


def maintainer_text(comments: list[dict]) -> str:
    """All maintainer comment bodies joined into one string."""
    return "\n".join(c["body"] or "" for c in comments
                     if c["author_association"] in MAINTAINERS)


def mentioned_numbers(text: str) -> list[int]:
    # TODO(you): every issue number written as "#1234" in `text`, as ints,
    #   each number once, in order of FIRST appearance.
    #   "dup of #8741 plus #8079, see #8741 again" -> [8741, 8079]
    #
    #   Hint: re.findall(pattern, text) returns every match. In a regex,
    #   \d+ means "one or more digits", and ( ) marks the part you want back.
    #   Careful: GitHub links contain "#issuecomment-99" - that is not an
    #   issue number, and your pattern must not return it.
    match = re.findall(r"#(\d+)", text)
    order = []
    for num in match:
        n = int(num)
        if n in order:
            continue
        order.append(n)
    return order


def targets_for(dup: dict, mentioned: list[int], by_number: dict[int, dict]) -> list[int]:
    # TODO(you): keep only the mentioned numbers that could really be the
    #   issue this one duplicates. Keep order. A number survives if:
    #   (a) it's an issue in our snapshot (by_number) - PRs aren't in there
    #   (b) it isn't the duplicate issue itself
    #   (c) it was created BEFORE `dup` - a finder can only find what already
    #       existed when the issue arrived (same idea as the time split)
    #
    #   by_number maps issue number -> issue dict, e.g. by_number[6492]
    dups = []
    for num in mentioned:
        if num in by_number and num != dup["number"] and by_number[num]["created_at"] < dup["created_at"]:
            dups.append(num)
    return dups


def main() -> None:
    issues = load_issues()
    by_number = {i["number"]: i for i in issues}
    dups = [i for i in issues if is_duplicate(i)]
    print(f"{len(dups)} duplicate issues; fetching their comments...")

    truth = {}
    with make_client(os.environ["GITHUB_TOKEN"]) as client:
        for dup in dups:
            # One page of 100 is plenty: duplicates get closed fast, with few comments.
            url = f"/repos/{REPO}/issues/{dup['number']}/comments"
            comments = get_page(client, url, {"per_page": 100}).json()
            found = targets_for(dup, mentioned_numbers(
                maintainer_text(comments)), by_number)
            truth[dup["number"]] = found
            print(f"  #{dup['number']:<6} -> {found or 'no usable target'}")

    usable = {n: t for n, t in truth.items() if t}
    print(f"\n{len(usable)} of {len(dups)} duplicates have a usable target")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(usable, indent=2) + "\n")
    print(f"saved to {OUT}")


if __name__ == "__main__":
    main()
