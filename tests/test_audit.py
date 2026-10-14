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


def test_ai_rater_is_blind_and_falls_back_to_unsure(monkeypatch):
    monkeypatch.setattr(audit, "load_ratings", lambda *a, **k: {})
    seen = []

    def fake_generate(body):
        seen.append(body["input"])
        return ({"verdict": "correct"} if len(seen) == 1 else {"verdict": "maybe"}), {}

    monkeypatch.setattr("pytest_triage.llm.generate_json", fake_generate)
    written = []
    monkeypatch.setattr(audit, "add_rating", lambda rater, item, v, d=None: written.append(v))
    sample = [{"number": 1, "label": "type: bug"}, {"number": 2, "label": "type: docs"}]
    by_number = {1: issue(1, ["type: question"]), 2: issue(2)}
    audit.rate_with_model(sample, by_number, {"type: bug": "d", "type: docs": "d"}, "m")
    assert written == ["correct", "unsure"]  # invalid verdict -> unsure
    assert "Maintainers applied: type: question" in seen[0]
    assert all("llm-rater-a" not in s for s in seen)  # never shown other raters' verdicts
