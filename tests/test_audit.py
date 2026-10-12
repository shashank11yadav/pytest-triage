import pytest

from pytest_triage import audit


def issue(number, labels=(), title="t", body="b"):
    return {"number": number, "labels": list(labels), "title": title, "body": body}


def test_false_positives_and_true_positives():
    issues = [issue(1, ["type: bug"]), issue(2, ["type: docs"])]
    predict = lambda i: {"type: bug", "topic: fixtures"}  # noqa: E731
    fps, tp = audit.false_positives(issues, predict)
    assert tp == 1
    assert fps == [
        {"number": 1, "label": "topic: fixtures"},
        {"number": 2, "label": "topic: fixtures"},
        {"number": 2, "label": "type: bug"},
    ]


def test_sample_is_reproducible_and_independent_of_input_order():
    fps = [{"number": n, "label": "type: bug"} for n in range(100)]
    assert audit.draw_sample(fps, 10) == audit.draw_sample(list(reversed(fps)), 10)
    assert len(audit.draw_sample(fps, 500)) == 100


def test_rating_session_is_resumable_and_validates_input(tmp_path):
    sample = [{"number": 1, "label": "type: bug"}, {"number": 2, "label": "type: docs"}]
    by_number = {1: issue(1), 2: issue(2)}
    defs = {"type: bug": "d", "type: docs": "d"}
    answers = iter(["x", "c", "q"])  # invalid, then correct, then quit
    shown = []
    rated = audit.rate_interactively(
        "me",
        sample,
        by_number,
        defs,
        ask=lambda _: next(answers),
        show=shown.append,
        directory=tmp_path,
    )
    assert rated == 1
    assert any("Please type" in s for s in shown)
    assert audit.load_ratings("me", tmp_path) == {(1, "type: bug"): "correct"}

    answers = iter(["w"])
    audit.rate_interactively(
        "me",
        sample,
        by_number,
        defs,
        ask=lambda _: next(answers),
        show=lambda s: None,
        directory=tmp_path,
    )
    assert audit.load_ratings("me", tmp_path)[(2, "type: docs")] == "wrong"


def test_adjusted_precision_ignores_unsure():
    result = audit.adjusted_precision(tp=40, fp=60, verdicts=["correct", "wrong", "unsure"])
    assert result["measured_precision"] == pytest.approx(0.4)
    assert result["share_of_fps_correct"] == pytest.approx(0.5)
    assert result["adjusted_precision"] == pytest.approx(0.7)  # (40 + 60*0.5) / 100
    assert result["unsure"] == 1


def test_cohen_kappa():
    assert audit.cohen_kappa(["correct", "wrong"] * 5, ["correct", "wrong"] * 5) == 1.0
    a = ["correct"] * 5 + ["wrong"] * 5
    b = ["correct", "wrong"] * 5
    assert audit.cohen_kappa(a, b) == pytest.approx(0.2)
