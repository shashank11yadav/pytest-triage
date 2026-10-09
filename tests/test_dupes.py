"""Tests for triage/dupes.py. Pure functions, no network."""

from triage.dupes import is_duplicate, maintainer_text, mentioned_numbers, targets_for


def issue(number, created_at="2024-01-01T00:00:00Z", state_reason=None, labels=()):
    return {"number": number, "created_at": created_at,
            "state_reason": state_reason, "labels": list(labels)}


# --- is_duplicate / maintainer_text -------------------------

def test_duplicate_by_reason_or_old_label():
    assert is_duplicate(issue(1, state_reason="duplicate"))
    assert is_duplicate(issue(2, state_reason="completed", labels=["closed as duplicate"]))
    assert not is_duplicate(issue(3, state_reason="completed"))


def test_only_maintainer_comments_count():
    comments = [{"body": "maybe #12?", "author_association": "NONE"},
                {"body": "dup of #34", "author_association": "MEMBER"},
                {"body": None, "author_association": "OWNER"}]
    assert maintainer_text(comments) == "dup of #34\n"


# --- mentioned_numbers -----------------------------------------------------

def test_finds_numbers_in_order_without_repeats():
    text = "This is #8741 plus #8079. Related: #5591, and again #8741"
    assert mentioned_numbers(text) == [8741, 8079, 5591]


def test_returns_ints_not_strings():
    assert mentioned_numbers("dup of #6492") == [6492]


def test_no_numbers():
    assert mentioned_numbers("This is a duplicate") == []


def test_ignores_comment_anchors_in_links():
    text = "see https://github.com/pytest-dev/pytest/issues/12#issuecomment-99"
    assert mentioned_numbers(text) == []


# --- targets_for -----------------------------------------------------------

def test_keeps_only_older_issues_in_snapshot_and_not_itself():
    by_number = {
        10: issue(10, "2020-01-01T00:00:00Z"),
        20: issue(20, "2021-01-01T00:00:00Z"),
        30: issue(30, "2023-01-01T00:00:00Z"),   # newer than the duplicate
    }
    dup = issue(25, "2022-01-01T00:00:00Z")
    by_number[25] = dup
    # 99 isn't in the snapshot (a PR, or another repo); 25 is itself; 30 is newer
    assert targets_for(dup, [20, 99, 25, 30, 10], by_number) == [20, 10]
