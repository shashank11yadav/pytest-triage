import pytest

from pytest_triage.policy import ContractViolation, apply_policy, check_report

ISSUE = {"number": 100, "created_at": "2026-05-01"}
OLDER = {7: "2025-01-01", 8: "2025-02-01"}


def report(**overrides):
    base = {
        "labels": ["type: bug"],
        "missing": [],
        "duplicates": [],
        "candidates": [7, 8],
        "injection_signals": [],
        "candidate_injection_signals": {},
        "reply_draft": "",
    }
    return base | overrides


def test_clean_report_passes_through():
    out = apply_policy(report(duplicates=[7]))
    assert out["route"] == "suggest"
    assert out["labels"] == ["type: bug"] and out["duplicates"] == [7]
    assert check_report(out, ISSUE, OLDER) == []


def test_flagged_issue_is_routed_with_nothing_suggested():
    out = apply_policy(
        report(injection_signals=["triage bot"], missing=["pytest and Python versions"])
    )
    assert out["route"] == "human"
    assert out["labels"] == out["missing"] == out["duplicates"] == []
    assert out["raw"]["labels"] == ["type: bug"]
    check_report(out, ISSUE, OLDER)  # routed reports satisfy the contract


def test_flagged_candidate_cannot_be_reported_as_duplicate():
    out = apply_policy(report(duplicates=[7, 8], candidate_injection_signals={8: ["triage bot"]}))
    assert out["duplicates"] == [7]
    assert out["quarantined_candidates"] == [8]
    assert out["route"] == "suggest"  # the issue itself is clean


@pytest.mark.parametrize(
    "bad",
    [
        {"labels": ["type: bogus"]},
        {"missing": ["your bank details"]},
        {"duplicates": [100]},
        {"duplicates": [9]},  # never a candidate
        {"duplicates": [7], "candidates": [7]},  # with a newer created_at below
    ],
)
def test_contract_violations_raise(bad):
    created = OLDER | {7: "2026-06-01"} if bad.get("candidates") == [7] else OLDER
    with pytest.raises(ContractViolation):
        check_report(report(**bad) | {"route": "suggest"}, ISSUE, created)


def test_routed_report_that_still_suggests_is_a_violation():
    with pytest.raises(ContractViolation):
        check_report(report(route="human"), ISSUE, OLDER)


def test_soft_oddities_are_warnings_not_rewrites():
    out = report(labels=["type: bug", "type: docs"], route="suggest")
    assert check_report(out, ISSUE, OLDER) == ["more than one type: label"]
    assert out["labels"] == ["type: bug", "type: docs"]
