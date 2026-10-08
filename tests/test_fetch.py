"""Tests for triage/fetch.py. No network: httpx.MockTransport fakes GitHub."""

import httpx

from triage.fetch import fetch_issues, is_pull_request, slim, wait_seconds

BASE = "https://api.github.com"


def issue(number, labels=(), pr=False):
    """A minimal item shaped like GitHub's real response."""
    item = {"number": number, "title": f"issue {number}", "body": "text",
            "state": "open", "state_reason": None, "created_at": "2026-01-01T00:00:00Z",
            "closed_at": None, "comments": 0, "user": {"login": "someone"},
            "labels": [{"name": name} for name in labels]}
    if pr:
        item["pull_request"] = {"url": "..."}
    return item


def fake_github(responses):
    """A client whose requests are answered, in order, from `responses`."""
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return responses.pop(0)

    client = httpx.Client(base_url=BASE, transport=httpx.MockTransport(handler))
    return client, seen


# --- is_pull_request -------------------------------------------------------

def test_pull_request_detected():
    assert is_pull_request(issue(1, pr=True)) is True


def test_real_issue_is_not_a_pull_request():
    assert is_pull_request(issue(1)) is False


# --- slim ------------------------------------------------------------------

def test_slim_flattens_labels_and_drops_username():
    out = slim(issue(7, labels=["type: bug", "topic: fixtures"]))
    assert out["labels"] == ["type: bug", "topic: fixtures"]
    assert "user" not in out


# --- wait_seconds ----------------------------------------------------------

def test_success_needs_no_wait():
    assert wait_seconds(httpx.Response(200), now=0) is None


def test_not_found_is_not_a_rate_limit():
    assert wait_seconds(httpx.Response(404), now=0) is None


def test_retry_after_wins():
    response = httpx.Response(429, headers={"retry-after": "30",
                                            "x-ratelimit-remaining": "0",
                                            "x-ratelimit-reset": "1000"})
    # now=900 so a duration is never confused with a timestamp, and both
    # hints are present so the test also pins down which one wins.
    assert wait_seconds(response, now=900) == 30.0


def test_waits_until_reset_when_quota_is_used_up():
    response = httpx.Response(403, headers={"x-ratelimit-remaining": "0",
                                            "x-ratelimit-reset": "1000"})
    assert wait_seconds(response, now=900) == 100.0


def test_reset_in_the_past_never_waits_negative():
    response = httpx.Response(403, headers={"x-ratelimit-remaining": "0",
                                            "x-ratelimit-reset": "1000"})
    assert wait_seconds(response, now=1100) == 0.0


def test_rate_limited_without_hints_waits_a_minute():
    assert wait_seconds(httpx.Response(429), now=0) == 60.0


# --- fetch_issues ----------------------------------------------------------

def test_follows_next_link_and_skips_pull_requests():
    next_url = f"{BASE}/repositories/1/issues?state=all&per_page=100&after=abc"
    client, seen = fake_github([
        httpx.Response(200, json=[issue(1), issue(2, pr=True), issue(3)],
                       headers={"link": f'<{next_url}>; rel="next"'}),
        httpx.Response(200, json=[issue(4)]),  # no link header = last page
    ])
    numbers = [i["number"] for i in fetch_issues(client, "o/r", sleep=lambda s: None)]
    assert numbers == [1, 3, 4]
    assert seen[1] == next_url  # second request used the Link URL exactly


def test_retries_the_same_page_after_rate_limit():
    slept = []
    client, seen = fake_github([
        httpx.Response(429, headers={"retry-after": "5"}),
        httpx.Response(200, json=[issue(1)]),
    ])
    numbers = [i["number"] for i in fetch_issues(client, "o/r", sleep=slept.append)]
    assert numbers == [1]
    assert slept == [5.0]
    assert seen[0] == seen[1]  # same page asked twice, not skipped
