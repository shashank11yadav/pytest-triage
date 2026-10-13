"""The regression gate. The first two tests run against the committed baseline in
eval/regression/, so they fail in CI when a change isn't re-measured."""

import pytest

from pytest_triage import regression

BASELINE = regression.load_baseline()
ROWS = regression.load_rows()
needs_baseline = pytest.mark.skipif(not BASELINE, reason="no committed baseline")


@needs_baseline
def test_requests_match_the_measured_baseline():
    current = regression.fingerprints(BASELINE["prompt"])
    changed = [k for k in current if current[k] != BASELINE["fingerprints"][k]]
    assert not changed, (
        f"the {changed} request changed since the baseline was measured. Run "
        "`uv run triage eval regress` (and the matching eval), then "
        "`uv run triage eval regress --update`, and commit eval/regression/."
    )


@needs_baseline
def test_post_processing_does_not_lower_dev_scores():
    current = regression.score(ROWS)
    assert current["n"] == BASELINE["metrics"]["n"]
    assert regression.compare(current, BASELINE["metrics"], BASELINE["tolerance"]) == []


def row(number, truth, labels, needs_info=False, missing=(), flagged=False):
    return {
        "number": number,
        "truth": truth,
        "needs_info": needs_info,
        "flagged": flagged,
        "labels": labels,
        "missing": list(missing),
    }


def test_score_applies_policy_to_flagged_rows():
    rows = [row(1, ["type: bug"], ["type: bug"]), row(2, ["type: docs"], ["type: docs"])]
    clean = regression.score(rows)
    rows[1]["flagged"] = True  # routed to a human: its labels are withheld
    routed = regression.score(rows)
    assert routed["micro_f1"] < clean["micro_f1"]


def test_unknown_labels_never_count():
    rows = [row(1, ["type: bug"], ["type: bug", "type: bogus"])]
    assert regression.score(rows)["micro_f1"] == 1.0


def test_compare_flags_only_drops_beyond_tolerance():
    base = {"macro_f1": 0.55, "micro_f1": 0.55, "needs_info_f1": 0.45}
    assert regression.compare(base | {"macro_f1": 0.54}, base) == []
    assert regression.compare(base | {"macro_f1": 0.50}, base) == ["macro_f1: 0.550 -> 0.500"]
    assert regression.compare(base | {"macro_f1": 0.70}, base) == []  # gains never fail


def test_changed_rows_counts_differences():
    old = [row(1, [], ["type: bug"]), row(2, [], [])]
    new = [row(1, [], ["type: docs"]), row(2, [], [])]
    assert regression.changed_rows(old, new) == 1


def test_any_prompt_change_moves_the_fingerprint(monkeypatch):
    from pytest_triage import classify

    before = regression.fingerprints("v3s")
    monkeypatch.setitem(classify.PROMPTS, "v3s", classify.PROMPTS["v3s"] + " Be terse.")
    after = regression.fingerprints("v3s")
    assert after["labels"] != before["labels"]
    assert after["expand"] == before["expand"]


def test_guard_changes_move_the_fingerprint(monkeypatch):
    from pytest_triage import guard

    before = regression.fingerprints()
    monkeypatch.setattr(guard, "HOME_PATHS", guard.re.compile(r"(?P<pre>x)(?P<user>y)"))
    assert regression.fingerprints() != before  # redaction feeds every request
