import pytest

from pytest_triage import baselines
from pytest_triage.metrics import confusion, hit_at_k, label_report, prf, wilson


def issue(labels=(), title=""):
    return {"labels": list(labels), "title": title, "body": ""}


def test_confusion_sorts_into_tp_fp_fn():
    c = confusion([({"type: bug"}, {"type: bug"}), ({"type: docs"}, {"type: bug"})])
    assert c["type: bug"] == {"tp": 1, "fp": 1}
    assert c["type: docs"] == {"fn": 1}


def test_prf_and_zero_division():
    assert prf(8, 2, 2) == pytest.approx((0.8, 0.8, 0.8))
    assert prf(0, 0, 5) == (0.0, 0.0, 0.0)
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)


def test_label_report_macro_and_micro():
    c = confusion([({"type: bug"}, {"type: bug"}), ({"type: docs"}, set())])
    report = label_report(c)
    assert report["labels"]["type: bug"]["f1"] == 1.0
    assert report["labels"]["type: docs"]["support"] == 1
    assert report["macro_f1"] == pytest.approx(0.1)  # 1 of 10 labels perfect
    assert report["micro_f1"] == pytest.approx(prf(1, 0, 1)[2])


def test_hit_at_k():
    assert hit_at_k([5, 6, 7], [7], 3)
    assert not hit_at_k([5, 6, 7], [7], 2)
    assert hit_at_k([5, 6, 7], [99, 6], 2)


def test_wilson_is_wide_for_small_n():
    lo, hi = wilson(10, 15)
    assert 0.40 < lo < 0.45 and 0.84 < hi < 0.88
    assert wilson(0, 0) == (0.0, 0.0)


# --- baselines ------------------------------------------------------------------


def test_majority_is_counted_on_memory():
    memory = [issue(["type: docs"]), issue(["type: docs"]), issue(["type: bug"])]
    assert baselines.make_majority(memory)(issue()) == {"type: docs"}


def test_mined_keywords_are_frequent_and_specific():
    memory = [issue(["type: docs"], "typo in docs page") for _ in range(20)]
    memory += [issue(["type: bug"], "crash on page load") for _ in range(20)]
    keywords = baselines.mine_keywords(memory, min_issues=10, min_hits=5, min_share=0.6)
    assert "typo" in keywords["type: docs"]
    assert "page" not in keywords["type: docs"]  # on bugs too: share 0.5 < 0.6
    assert "crash" in keywords["type: bug"]


def test_keyword_predictor_matches_title_words_case_insensitively():
    predict = baselines.make_keyword_predictor({"type: docs": ["typo"], "type: bug": ["crash"]})
    assert predict(issue(title="Typo in README")) == {"type: docs"}
    assert predict(issue(labels=["type: bug"], title="segfault")) == set()  # never reads labels
