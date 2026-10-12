from pytest_triage.dataset import is_evaluable, issue_text, split, targets
from pytest_triage.store import last_updated, merge


def issue(number, created_at="2024-06-01T00:00:00Z", labels=(), **extra):
    return {"number": number, "created_at": created_at, "labels": list(labels), **extra}


def test_unlabelled_issue_is_not_evaluable_but_any_label_is():
    assert not is_evaluable(issue(1))
    assert is_evaluable(issue(1, labels=["plugin: xdist"]))


def test_targets_keeps_only_predicted_labels():
    i = issue(1, labels=["type: bug", "good first issue", "topic: fixtures"])
    assert targets(i) == {"type: bug", "topic: fixtures"}


def test_split_boundary_goes_to_test_and_order_is_kept():
    issues = [issue(n, f"20{20 + n}-03-01T00:00:00Z") for n in range(1, 7)]
    issues.append(issue(9, "2024-12-31T23:59:59Z"))
    issues.append(issue(10, "2025-01-01T00:00:00Z"))
    memory, test = split(issues, date="2025-01-01")
    assert [i["number"] for i in memory] == [1, 2, 3, 4, 9]
    assert [i["number"] for i in test] == [5, 6, 10]


def test_issue_text_truncates_body_and_handles_none():
    assert issue_text({"title": "T", "body": None}, 5) == "T\n"
    assert issue_text({"title": "T", "body": "abcdefgh"}, 5) == "T\nabcde"


# --- store ---------------------------------------------------------------------


def test_merge_replaces_by_number_and_sorts_by_creation():
    old = [issue(1, "2020-01-01", title="old"), issue(2, "2021-01-01")]
    new = [issue(1, "2020-01-01", title="edited"), issue(3, "2019-01-01")]
    merged = merge(old, new)
    assert [i["number"] for i in merged] == [3, 1, 2]
    assert merged[1]["title"] == "edited"


def test_last_updated_is_the_newest_stamp():
    issues = [issue(1, updated_at="2026-01-01"), issue(2, updated_at="2026-03-01"), issue(3)]
    assert last_updated(issues) == "2026-03-01"
    assert last_updated([]) is None
