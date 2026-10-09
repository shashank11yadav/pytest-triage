"""Tests for triage/index.py. Pure functions only - no model, no files."""

import numpy as np

from triage.index import hit, issue_text, rank, rrf


# --- issue_text ---------------------------------------------

def test_issue_text_handles_missing_body():
    assert issue_text({"title": "Crash", "body": None}) == "Crash\n"


# --- rank ------------------------------------------------------------------

def test_rank_best_first_skipping_disallowed():
    scores = np.array([0.2, 0.9, 0.5, 0.7])
    allowed = np.array([True, False, True, True])
    assert rank(scores, [10, 20, 30, 40], allowed, top=2) == [40, 30]


def test_rank_returns_fewer_when_few_are_allowed():
    scores = np.array([0.2, 0.9, 0.5])
    allowed = np.array([False, False, True])
    assert rank(scores, [10, 20, 30], allowed, top=5) == [30]


def test_rank_returns_plain_ints():
    got = rank(np.array([1.0]), [7], np.array([True]), top=1)
    assert got == [7] and type(got[0]) is int


# --- rrf -------------------------------------------------------------------

def test_rrf_item_both_lists_like_beats_item_one_list_loves():
    # 2 is 2nd in both lists; 1 is 1st in one list and absent from the other.
    fused = rrf([[1, 2, 3], [4, 2, 5]])
    assert fused[0] == 2


def test_rrf_keeps_every_item():
    assert set(rrf([[1, 2], [3]])) == {1, 2, 3}


def test_rrf_k_decides_one_strong_vote_vs_two_weak_ones():
    # Item 1: 1st in one list only.   Item 2: 3rd in both lists.
    rankings = [[1, 9, 2], [8, 7, 2]]
    # k=0:  1 gets 1/1 = 1.00      2 gets 1/3 + 1/3 = 0.67   -> 1 wins
    assert rrf(rankings, k=0)[0] == 1
    # k=60: 1 gets 1/61 = 0.016    2 gets 2/63 = 0.032       -> 2 wins
    assert rrf(rankings)[0] == 2


# --- hit -------------------------------------------------------------------

def test_hit_inside_k():
    assert hit([5, 6, 7], targets=[7], k=3) is True


def test_hit_outside_k():
    assert hit([5, 6, 7], targets=[7], k=2) is False


def test_hit_any_of_several_targets():
    assert hit([5, 6, 7], targets=[99, 6], k=2) is True
