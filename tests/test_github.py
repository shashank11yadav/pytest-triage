"""GitHub client. No network: httpx.MockTransport fakes the API."""

import httpx

from pytest_triage.github import (
    fetch_comments,
    fetch_issues,
    is_pull_request,
    slim,
    wait_seconds,
)

BASE = "https://api.github.com"


def item(number, labels=(), pr=False):
    raw = {
        "number": number,
        "title": f"issue {number}",
        "body": "text",
        "state": "open",
        "state_reason": None,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-02T00:00:00Z",
        "closed_at": None,
        "comments": 0,
        "user": {"login": "someone"},
        "labels": [{"name": n} for n in labels],
    }
    if pr:
        raw["pull_request"] = {"url": "..."}
    return raw


def fake_github(responses):
    seen = []

    def handler(request):
        seen.append(request.url)
        return responses.pop(0)

    return httpx.Client(base_url=BASE, transport=httpx.MockTransport(handler)), seen


def no_sleep(_):
    pass


def test_pull_requests_detected():
    assert is_pull_request(item(1, pr=True))
    assert not is_pull_request(item(1))


def test_slim_flattens_labels_and_drops_username():
    out = slim(item(7, labels=["type: bug", "topic: fixtures"]))
    assert out["labels"] == ["type: bug", "topic: fixtures"]
    assert "user" not in out
    assert out["updated_at"] == "2026-01-02T00:00:00Z"


def test_wait_seconds_rules():
    assert wait_seconds(httpx.Response(200), now=0) is None
    assert wait_seconds(httpx.Response(404), now=0) is None
    both = {"retry-after": "30", "x-ratelimit-remaining": "0", "x-ratelimit-reset": "1000"}
    assert wait_seconds(httpx.Response(429, headers=both), now=900) == 30.0  # duration wins
    reset = {"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1000"}
    assert wait_seconds(httpx.Response(403, headers=reset), now=900) == 100.0
    assert wait_seconds(httpx.Response(403, headers=reset), now=1100) == 0.0
    assert wait_seconds(httpx.Response(429), now=0) == 60.0


def test_follows_next_link_and_skips_pull_requests():
    next_url = f"{BASE}/repositories/1/issues?state=all&per_page=100&after=abc"
    client, seen = fake_github(
        [
            httpx.Response(
                200,
                json=[item(1), item(2, pr=True), item(3)],
                headers={"link": f'<{next_url}>; rel="next"'},
            ),
            httpx.Response(200, json=[item(4)]),
        ]
    )
    assert [i["number"] for i in fetch_issues(client, "o/r", sleep=no_sleep)] == [1, 3, 4]
    assert str(seen[1]) == next_url  # cursor kept: params not re-sent


def test_since_switches_to_sort_by_updated():
    client, seen = fake_github([httpx.Response(200, json=[])])
    list(fetch_issues(client, "o/r", since="2026-10-01T00:00:00Z", sleep=no_sleep))
    assert seen[0].params["since"] == "2026-10-01T00:00:00Z"
    assert seen[0].params["sort"] == "updated"


def test_retries_the_same_page_after_rate_limit():
    slept = []
    client, seen = fake_github(
        [httpx.Response(429, headers={"retry-after": "5"}), httpx.Response(200, json=[item(1)])]
    )
    assert [i["number"] for i in fetch_issues(client, "o/r", sleep=slept.append)] == [1]
    assert slept == [5.0]
    assert seen[0] == seen[1]


def test_fetch_comments_returns_raw_comments():
    client, _ = fake_github([httpx.Response(200, json=[{"body": "hi"}])])
    assert fetch_comments(client, "o/r", 5, sleep=no_sleep) == [{"body": "hi"}]
