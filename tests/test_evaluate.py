"""Tests for triage/evaluate.py. Pure functions, no files, no network."""

import pytest

import triage.evaluate as ev


def issue(labels=(), title="", body=""):
    return {"labels": list(labels), "title": title, "body": body}


# --- count (pins down what tp/fp/fn mean) --------------------

def test_count_sorts_each_label_into_tp_fp_fn():
    issues = [issue(["type: bug"]), issue(["type: docs"])]
    always_bug = lambda i: {"type: bug"}
    c = ev.count(always_bug, issues)
    assert c["type: bug"] == {"tp": 1, "fp": 1}   # right once, false alarm once
    assert c["type: docs"] == {"fn": 1}            # missed the docs issue


# --- scores ----------------------------------------------------------------

def test_scores_basic():
    p, r, f1 = ev.scores(tp=8, fp=2, fn=2)
    assert (p, r, f1) == pytest.approx((0.8, 0.8, 0.8))


def test_scores_uneven():
    p, r, f1 = ev.scores(tp=3, fp=1, fn=0)
    assert (p, r) == pytest.approx((0.75, 1.0))
    assert f1 == pytest.approx(2 * 0.75 * 1.0 / 1.75)


def test_never_predicting_scores_zero_not_crash():
    assert ev.scores(tp=0, fp=0, fn=5) == (0.0, 0.0, 0.0)


def test_nothing_to_find_and_nothing_guessed_is_zero():
    assert ev.scores(tp=0, fp=0, fn=0) == (0.0, 0.0, 0.0)


# --- keyword_predict -------------------------------------------------------
# These swap in a fixed KEYWORDS dict, so they test your matching logic,
# not your choice of words.

@pytest.fixture
def rules(monkeypatch):
    monkeypatch.setattr(ev, "KEYWORDS", {"type: docs": ["typo", "documentation"],
                                         "topic: fixtures": ["fixture"]})


def test_keywords_match_case_insensitively(rules):
    assert ev.keyword_predict(issue(title="Typo in README")) == {"type: docs"}


def test_keywords_can_predict_several_labels(rules):
    got = ev.keyword_predict(issue(title="Typo in the Fixtures page"))
    assert got == {"type: docs", "topic: fixtures"}


def test_keywords_ignore_the_body(rules):
    # Measured on memory: body text (code, tracebacks) drowns the signal.
    got = ev.keyword_predict(issue(title="segfault", body="my fixture has a typo"))
    assert got == set()


def test_keywords_never_read_the_answer(rules):
    # The issue's true labels must not influence the prediction.
    # Its true labels are docs + a label we don't predict; its title says fixture.
    got = ev.keyword_predict(issue(labels=["type: docs", "good first issue"],
                                   title="fixture teardown order"))
    assert got == {"topic: fixtures"}


def test_no_keywords_no_labels(rules):
    assert ev.keyword_predict(issue(title="segfault on import")) == set()
