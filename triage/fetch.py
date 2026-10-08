"""Download every issue from a GitHub repo into a local JSONL snapshot.

Run it:
    GITHUB_TOKEN=$(gh auth token) uv run python -m triage.fetch

Docs used (checked 2026-10-08):
    list issues  https://docs.github.com/en/rest/issues/issues#list-repository-issues
    pagination   https://docs.github.com/en/rest/using-the-rest-api/using-pagination-in-the-rest-api
    rate limits  https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api
"""

import json
import os
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx

REPO = "pytest-dev/pytest"
OUT = Path("data/issues.jsonl")
MAX_ATTEMPTS = 5  # give up on one page after this many rate-limited tries

# Only the fields we need later. Dropping `user` keeps usernames out of our data.
KEEP = ("number", "title", "body", "state", "state_reason",
        "created_at", "closed_at", "comments")


def make_client(token: str) -> httpx.Client:
    """An HTTP client that sends GitHub's recommended headers on every request."""
    return httpx.Client(
        base_url="https://api.github.com",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2026-03-10",
        },
        timeout=30.0,
    )


def is_pull_request(item: dict) -> bool:
    # TODO(you): GitHub's issues endpoint also returns pull requests.
    #   The docs say PRs can be identified by one specific key in the item.
    #   Return True for a PR, False for a real issue. One line.
    if item.get("pull_request", ""):
        return True
    return False


def slim(item: dict) -> dict:
    """Keep only the fields in KEEP, and flatten labels to a list of names."""
    issue = {key: item.get(key) for key in KEEP}
    issue["labels"] = [label["name"] for label in item.get("labels", [])]
    return issue


def wait_seconds(response: httpx.Response, now: float) -> float | None:
    # TODO(you): decide how long to wait before retrying, following the
    #   rate-limit docs. `now` is the current time in epoch seconds
    #   (passed in, not read from the clock, so tests can control it).
    #
    #   - status is not 403 or 429           -> None (not rate-limited)
    #   - has a `retry-after` header         -> that many seconds, as a float
    #   - `x-ratelimit-remaining` is "0"     -> seconds until `x-ratelimit-reset`,
    #                                           never negative
    #   - otherwise (403/429, no hints)      -> 60.0
    #
    #   Hint: response.status_code, response.headers.get("retry-after").
    #   Header values are always strings — convert before doing maths.
    if response.status_code not in [403, 429]:
        return None
    if response.headers.get("retry-after"):
        return float(response.headers.get("retry-after"))
    if response.headers.get("x-ratelimit-remaining") == "0":
        return max(float(response.headers.get("x-ratelimit-reset")) - now, 0.0)
    return 60.0


def get_page(client: httpx.Client, url: str, params: dict | None,
             sleep: Callable[[float], None] = time.sleep) -> httpx.Response:
    """GET one page, waiting and retrying while GitHub says we're rate-limited."""
    for _ in range(MAX_ATTEMPTS):
        response = client.get(url, params=params)
        wait = wait_seconds(response, time.time())
        if wait is None:
            response.raise_for_status()  # any other error (404, 500...) stops us loudly
            return response
        print(f"rate-limited, sleeping {wait:.0f}s")
        sleep(wait)
    raise RuntimeError(
        f"still rate-limited after {MAX_ATTEMPTS} attempts: {url}")


def fetch_issues(client: httpx.Client, repo: str,
                 sleep: Callable[[float], None] = time.sleep) -> Iterator[dict]:
    """Yield every issue (not PR) in `repo`, oldest first, already slimmed."""
    url = f"/repos/{repo}/issues"
    params = {"state": "all", "per_page": 100, "direction": "asc"}
    # TODO(you): the pagination loop.
    #   1. Fetch a page with get_page(client, url, params, sleep).
    #   2. For each item in response.json(): skip pull requests, yield slim(item).
    #   3. Find the next page's URL in the Link header. httpx parses it for you:
    #      response.links is a dict like {"next": {"url": "..."}}.
    #   4. No "next" link -> you're done. Otherwise loop with that URL.
    #
    #   Watch out: the "next" URL already carries its own query string,
    #   including the `after=` cursor. If you also pass `params`, httpx
    #   REPLACES that whole query string with yours (checked on httpx 0.28.1),
    #   so the cursor vanishes and you fetch page 1 forever. Use params=None
    #   after the first page.
    while True:
        response = get_page(client, url, params, sleep)
        for item in response.json():
            if is_pull_request(item):
                continue
            yield slim(item)
        if response.links.get("next"):
            url = response.links.get("next").get("url")
            params = None
        else:
            break


def save_jsonl(issues: Iterator[dict], path: Path) -> int:
    """Write one JSON object per line. Returns how many were written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as f:
        for issue in issues:
            f.write(json.dumps(issue) + "\n")
            count += 1
            if count % 500 == 0:
                print(f"{count} issues...")
    return count


def main() -> None:
    token = os.environ["GITHUB_TOKEN"]  # KeyError here = you forgot to set it
    with make_client(token) as client:
        count = save_jsonl(fetch_issues(client, REPO), OUT)
    print(f"saved {count} issues to {OUT}")


if __name__ == "__main__":
    main()
