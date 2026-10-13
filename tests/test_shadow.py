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
