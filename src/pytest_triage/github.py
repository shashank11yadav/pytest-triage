"""Read-only GitHub client: issues and comments, with pagination and rate limits.

This project never writes to GitHub. There are no POST/PATCH calls here, on purpose.

Docs (checked 2026-10-08):
    list issues   https://docs.github.com/en/rest/issues/issues#list-repository-issues
    comments      https://docs.github.com/en/rest/issues/comments#list-issue-comments
    pagination    https://docs.github.com/en/rest/using-the-rest-api/using-pagination-in-the-rest-api
    rate limits   https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api
"""

import time
from collections.abc import Callable, Iterator

import httpx

API_VERSION = "2026-03-10"
MAX_ATTEMPTS = 5

# Fields kept per issue. `user` is dropped: usernames aren't needed to triage.
KEEP = (
    "number",
    "title",
    "body",
    "state",
    "state_reason",
    "created_at",
    "updated_at",
    "closed_at",
    "comments",
)

Sleep = Callable[[float], None]


def make_client(token: str) -> httpx.Client:
    return httpx.Client(
        base_url="https://api.github.com",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": API_VERSION,
        },
        timeout=30.0,
    )


def is_pull_request(item: dict) -> bool:
    """The issues endpoint also returns PRs; they carry a `pull_request` key."""
    return "pull_request" in item


def slim(item: dict) -> dict:
    issue = {key: item.get(key) for key in KEEP}
    issue["labels"] = [label["name"] for label in item.get("labels", [])]
    return issue


def wait_seconds(response: httpx.Response, now: float) -> float | None:
    """How long to back off, or None if this response isn't a rate limit.

    `retry-after` is a duration; `x-ratelimit-reset` is an epoch timestamp.
    """
    if response.status_code not in (403, 429):
        return None
    if retry_after := response.headers.get("retry-after"):
        return float(retry_after)
    if response.headers.get("x-ratelimit-remaining") == "0":
        return max(float(response.headers["x-ratelimit-reset"]) - now, 0.0)
    return 60.0


def get_page(
    client: httpx.Client, url: str, params: dict | None, sleep: Sleep = time.sleep
) -> httpx.Response:
    for _ in range(MAX_ATTEMPTS):
        response = client.get(url, params=params)
        wait = wait_seconds(response, time.time())
        if wait is None:
            response.raise_for_status()
            return response
        print(f"rate-limited, sleeping {wait:.0f}s", flush=True)
        sleep(wait)
    raise RuntimeError(f"still rate-limited after {MAX_ATTEMPTS} attempts: {url}")


def iter_items(
    client: httpx.Client, url: str, params: dict, sleep: Sleep = time.sleep
) -> Iterator[dict]:
    """Every item across all pages, following the Link header's cursor.

    After page 1, params must be None: httpx would REPLACE the next URL's query
    string (including its `after=` cursor) with ours and loop on page 1 forever.
    """
    next_url: str | None = url
    next_params: dict | None = params
    while next_url:
        response = get_page(client, next_url, next_params, sleep)
        yield from response.json()
        next_url = response.links.get("next", {}).get("url")
        next_params = None


def fetch_issues(
    client: httpx.Client, repo: str, since: str | None = None, sleep: Sleep = time.sleep
) -> Iterator[dict]:
    """Issues (not PRs), slimmed. With `since`, only those updated after it."""
    params = {"state": "all", "per_page": 100, "direction": "asc"}
    if since:
        params |= {"since": since, "sort": "updated"}
    for item in iter_items(client, f"/repos/{repo}/issues", params, sleep):
        if not is_pull_request(item):
            yield slim(item)


def fetch_comments(
    client: httpx.Client, repo: str, number: int, sleep: Sleep = time.sleep
) -> list[dict]:
    url = f"/repos/{repo}/issues/{number}/comments"
    return list(iter_items(client, url, {"per_page": 100}, sleep))
