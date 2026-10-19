from pytest_triage.shadow import score_shadow


def prediction(number, labels=(), needs_info=False, duplicates=()):
    return {
        "number": number,
        "labels": list(labels),
        "needs_info": needs_info,
        "duplicates": list(duplicates),
        "latency_s": 1.0 + number / 100,
        "cost_usd": 0.001,
    }


def issue(number, labels=(), state_reason=None):
    return {"number": number, "labels": list(labels), "state_reason": state_reason}


def test_score_counts_untriaged_and_scores_only_triaged():
    preds = [
        prediction(1, ["type: bug"]),
        prediction(2, ["type: docs"], needs_info=True),
        prediction(3, ["type: bug"], duplicates=[1]),
    ]
    issues = [
        issue(1, ["type: bug"]),
        issue(2, ["status: needs information"]),
        issue(3, [], state_reason="duplicate"),  # closed as dup but never labelled
    ]
    result = score_shadow(preds, issues)
    assert result["predicted"] == 3
    assert result["maintainer_triaged"] == 2
    assert result["still_untriaged"] == 1
    assert result["labels"]["labels"]["type: bug"]["f1"] == 1.0
    assert result["needs_info"]["precision"] == 1.0
    assert result["flagged_and_closed_duplicate"] == [3]
    assert result["cost_usd_total"] == 0.003
    assert result["predicted_before_any_maintainer_label"] == 3  # no labels_at_prediction


def test_predictions_for_unknown_issues_are_ignored():
    assert score_shadow([prediction(99)], [issue(1)])["predicted"] == 0


def test_shadow_scores_shown_and_all_labels_separately():
    preds = [prediction(1, ["type: bug"]) | {"labels_all": ["type: bug", "topic: config"]}]
    preds.append(prediction(2, []) | {"labels_all": ["type: docs"]})
    issues = [issue(1, ["type: bug"]), issue(2, ["type: docs"])]
    result = score_shadow(preds, issues)
    assert result["issues_with_a_shown_label"] == 1
    assert result["triaged_with_a_shown_label"] == 1
    assert result["labels_all"]["labels"]["type: docs"]["f1"] == 1.0
    assert result["labels"]["labels"]["type: docs"]["f1"] == 0.0


def test_shown_labels_count_untriaged_issues_too():
    # A label shown on an issue maintainers haven't labelled yet still counts as shown,
    # but only triaged issues count towards the scored subset.
    preds = [prediction(1, ["type: bug"]), prediction(2, ["topic: collection"]), prediction(3, [])]
    issues = [issue(1, ["type: bug"]), issue(2, []), issue(3, [])]
    result = score_shadow(preds, issues)
    assert result["issues_with_a_shown_label"] == 2
    assert result["triaged_with_a_shown_label"] == 1
    assert result["maintainer_triaged"] == 1


def test_one_bad_answer_skips_that_issue_only(monkeypatch, tmp_path):
    from pytest_triage import shadow
    from pytest_triage.llm import BadOutput

    monkeypatch.setattr(shadow, "PREDICTIONS", tmp_path / "p.jsonl")
    monkeypatch.setattr(shadow, "TRACES", tmp_path / "t.jsonl")

    def fake_triage(issue, *a, **k):
        if issue["number"] == 1:
            raise BadOutput("cut off")
        return {
            "route": "suggest",
            "labels": ["type: bug"],
            "labels_all": ["type: bug"],
            "missing": [],
            "duplicates": [],
            "models": {},
            "trace": {"latency_s": 1.0, "cost_usd": 0.001},
        }

    monkeypatch.setattr("pytest_triage.triage.triage_issue", fake_triage)
    new = [{"number": n, "created_at": "2026-10-15", "labels": [], "title": "t"} for n in (1, 2)]
    assert shadow.run_shadow(new, None, None, since="2026-10-14", limit=10) == 1
    assert [p["number"] for p in shadow.load_predictions(tmp_path / "p.jsonl")] == [2]
