from pytest_triage.slices import SLICES, slice_report


def issue(body, labels=("type: bug",), title="crash"):
    return {"title": title, "body": body, "labels": list(labels)}


def test_slices_split_by_issue_shape():
    tb = issue("x" * 500 + "\nTraceback (most recent call last):\n  ...")
    assert SLICES["has a traceback"](tb) and not SLICES["no traceback"](tb)
    assert SLICES["body 400-2000 chars"](tb)
    assert SLICES["title is a question"](issue("", title="How do I skip a test?"))
    assert SLICES["maintainers gave no type: label"](issue("", labels=["topic: fixtures"]))


def test_report_has_n_and_interval_per_predictor():
    issues = [issue("short"), issue("short", labels=["type: docs"])]
    preds = {"p": [{"type: bug"}, {"type: bug"}]}
    out = slice_report(issues, preds)
    assert out["all"]["n"] == 2
    assert out["all"]["p"]["micro_f1"] == 0.5
    lo, hi = out["all"]["p"]["ci95"]
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0
    assert "body > 2000 chars" not in out  # empty slices are left out
