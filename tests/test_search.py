"""Search logic with tiny hand-made vectors: no model download, no files."""

import numpy as np

from pytest_triage.search import IssueIndex, rank, rrf, tokenize


def issue(number, created_at, title):
    return {"number": number, "created_at": created_at, "title": title, "body": None, "labels": []}


def test_tokenize_lowercases_and_keeps_identifiers():
    assert tokenize("Fixture `request_param` Broke!") == ["fixture", "request_param", "broke"]


def test_rank_best_first_skipping_disallowed():
    scores = np.array([0.2, 0.9, 0.5, 0.7])
    allowed = np.array([True, False, True, True])
    assert rank(scores, [10, 20, 30, 40], allowed, top=2) == [40, 30]
    assert rank(scores, [10, 20, 30, 40], np.array([False] * 3 + [True]), top=5) == [40]


def test_rrf_agreement_and_k():
    assert rrf([[1, 2, 3], [4, 2, 5]])[0] == 2
    assert set(rrf([[1, 2], [3]])) == {1, 2, 3}
    rankings = [[1, 9, 2], [8, 7, 2]]
    assert rrf(rankings, k=0)[0] == 1  # one strong vote wins when k is small
    assert rrf(rankings)[0] == 2  # agreement wins at k=60


def make_index():
    issues = [
        issue(1, "2020-01-01", "fixture teardown order is wrong"),
        issue(2, "2021-01-01", "colour output broken on windows"),
        issue(3, "2022-01-01", "fixture teardown runs twice"),
        issue(4, "2023-01-01", "fixture teardown order regression"),
    ]
    vectors = np.array([[1, 0], [0, 1], [0.9, 0.1], [1, 0]], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return IssueIndex(issues, vectors)


def test_similar_never_returns_newer_issues_or_itself():
    index = make_index()
    for method in ("bm25", "dense", "hybrid"):
        found = index.similar(index.get(3), method=method)
        assert 3 not in found and 4 not in found
        assert found[0] == 1


def test_similar_respects_explicit_cutoff():
    index = make_index()
    assert set(index.similar(index.get(4), before="2021-06-01", method="dense")) == {1, 2}
