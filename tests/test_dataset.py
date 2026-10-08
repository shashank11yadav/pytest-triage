"""Tests for triage/dataset.py. Pure functions, no files, no network."""

from triage.dataset import is_evaluable, split, targets


def issue(number, created_at="2024-06-01T00:00:00Z", labels=()):
    return {"number": number, "created_at": created_at, "labels": list(labels)}


# --- is_evaluable ----------------------------------------------------------

def test_unlabelled_issue_is_not_evaluable():
    assert is_evaluable(issue(1)) is False


def test_any_label_makes_it_evaluable_even_one_we_dont_predict():
    assert is_evaluable(issue(1, labels=["plugin: xdist"])) is True


# --- targets ---------------------------------------------------------------

def test_targets_keeps_only_labels_we_predict():
    i = issue(1, labels=["type: bug", "good first issue", "topic: fixtures"])
    assert targets(i) == {"type: bug", "topic: fixtures"}


def test_targets_is_empty_when_no_predicted_labels():
    assert targets(issue(1, labels=["plugin: xdist"])) == set()


# --- split -----------------------------------------------------------------

def test_split_boundary_goes_to_test():
    last_old = issue(1, "2024-12-31T23:59:59Z")
    first_new = issue(2, "2025-01-01T00:00:00Z")
    memory, test = split([last_old, first_new], date="2025-01-01")
    assert [i["number"] for i in memory] == [1]
    assert [i["number"] for i in test] == [2]


def test_split_keeps_order_and_loses_nothing():
    issues = [issue(n, f"20{20 + n}-03-01T00:00:00Z") for n in range(1, 7)]  # 2021..2026
    memory, test = split(issues, date="2024-01-01")
    assert [i["number"] for i in memory] == [1, 2, 3]
    assert [i["number"] for i in test] == [4, 5, 6]
