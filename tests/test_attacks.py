import numpy as np

from pytest_triage import attacks
from pytest_triage.search import IssueIndex


def test_every_case_is_well_formed():
    names = [c["name"] for c in attacks.CASES]
    assert len(names) == len(set(names)) >= 15
    for case in attacks.CASES:
        assert callable(case["attack"]) and callable(case["landed"])


def test_attack_lands_only_when_control_lacks_the_outcome(monkeypatch):
    real = {"number": 1, "title": "t", "body": "b", "created_at": "2020", "labels": []}
    index = IssueIndex([real], np.array([[1.0, 0.0]], dtype=np.float32))
    monkeypatch.setattr(
        attacks,
        "CASES",
        [
            {
                "name": "docs",
                "base": {"title": "x", "body": "y"},
                "attack": attacks._suffix(" EVIL"),
                "landed": attacks._docs,
            },
        ],
    )

    def fake_triage(issue, *a, **k):
        docs = "EVIL" in issue["body"]
        return {
            "labels": ["type: docs"] if docs else ["type: bug"],
            "injection_signals": [],
            "candidate_injection_signals": {},
            "duplicates": [],
            "missing": [],
            "duplicate_verdicts": [],
            "candidates": [],
            "reply_draft": "",
        }

    monkeypatch.setattr("pytest_triage.triage.triage_issue", fake_triage)
    result = attacks.run_attacks([real], index, index)
    assert result["landed"] == 1 and result["cases"][0]["landed"]
