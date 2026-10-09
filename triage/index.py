"""Search past issues for likely duplicates: keyword (BM25), meaning
(embeddings), and both fused (RRF). Scored by hit@5 on eval/duplicates.json.

Run it:
    uv run python -m triage.index
(First run downloads the ~90MB embedding model and embeds all issues, which
takes a few minutes. The vectors are cached in data/, so later runs are fast.)

Docs used (checked 2026-10-09):
    fastembed  https://github.com/qdrant/fastembed (Getting Started notebook)
"""

import json
import re
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi

from triage.dataset import load_issues

DUPES = Path("eval/duplicates.json")
VECTORS = Path("data/vectors.npy")  # gitignored with the rest of data/
MAX_BODY = 500  # characters of body to index; the rest is mostly tracebacks
TOP = 100       # how deep each ranking goes before fusion
K = 5           # "did the original show up in the top K?"


def issue_text(issue: dict) -> str:
    """What we index and search with: title + the start of the body."""
    return f"{issue['title']}\n{(issue['body'] or '')[:MAX_BODY]}"


def tokenize(text: str) -> list[str]:
    """BM25 matches tokens literally, so this IS the keyword-matching rule."""
    return re.findall(r"\w+", text.lower())


def embed_all(issues: list[dict]) -> np.ndarray:
    """One vector per issue, rows in the same order as `issues`. Cached on disk.

    We call embed() for both sides, not query_embed() / passage_embed():
    those are for asymmetric search (short question vs. long document). Here
    both sides are issues, so the search is symmetric.

    Rows are scaled to length 1 ("normalised"), so a plain dot product between
    two rows IS their cosine similarity - the whole matrix scores in one step.
    """
    if VECTORS.exists():
        vectors = np.load(VECTORS)
        if len(vectors) == len(issues):
            return vectors
    model = TextEmbedding()  # BAAI/bge-small-en-v1.5, 384 numbers per text
    # embed() is a generator, so we can report progress as vectors arrive.
    # Without this, ~5 minutes of silence looks exactly like a hang.
    rows = []
    for row in model.embed([issue_text(i) for i in issues]):
        rows.append(row)
        if len(rows) % 500 == 0:
            print(f"  embedded {len(rows)}/{len(issues)}", flush=True)
    vectors = np.array(rows)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    np.save(VECTORS, vectors)
    return vectors


# --- the three pieces you write ---------------------------------------------

def rank(scores: np.ndarray, numbers: list[int], allowed: np.ndarray, top: int) -> list[int]:
    # TODO(you): turn raw scores into a ranked list of issue numbers.
    #   scores[i] is how well issue numbers[i] matches the query.
    #   allowed[i] is True if issue i may be returned (created BEFORE the query).
    #   Return the issue numbers of the best `top` allowed issues, best first.
    #
    #   Expected: scores [0.2, 0.9, 0.5, 0.7], numbers [10, 20, 30, 40],
    #             allowed [True, False, True, True], top 2  ->  [40, 30]
    #             (20 scored highest but isn't allowed)
    #
    #   Plain Python with zip() and sorted() is fine. If you want numpy:
    #   np.argsort gives the positions that would sort an array (ascending).
    sn = []
    for score, number, ok in zip(scores, numbers, allowed):
        if ok:
            sn.append((score, number))

    sn.sort(reverse=True)
    nums = [s[1] for s in sn[:top]]
    return nums


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    # TODO(you): reciprocal rank fusion - you built this in agent-lab Stage 2.
    #   Write it again from memory; it can be much shorter than last time.
    #   Each ranking gives every item it contains 1 / (k + position), with
    #   positions starting at 1. Add up each item's votes across rankings.
    #   Return all items, highest total first.
    #   Hint: a dict of totals, then sorted(..., key=..., reverse=True).
    total = {}
    for ranking in rankings:
        for pos, l in enumerate(ranking, start=1):
            vote = 1 / (k + pos)
            total[l] = total.get(l, 0) + vote
    return sorted(total, key=total.get, reverse=True)


def hit(ranked: list[int], targets: list[int], k: int) -> bool:
    # TODO(you): True if ANY target appears in the first k of `ranked`.
    #   One line.
    for number in ranked[:k]:
        if number in targets:
            return True
    return False


# --- run --------------------------------------------------------------------

def main() -> None:
    issues = load_issues()
    numbers = [i["number"] for i in issues]
    position = {n: p for p, n in enumerate(numbers)}
    created = np.array([i["created_at"] for i in issues])
    truth = {int(n): t for n, t in json.loads(DUPES.read_text()).items()}

    print("building BM25 index...")
    # Simplification: BM25 learns word rarity from ALL issues, including ones
    # newer than a query. The effect on rarity of 15 queries is tiny; `allowed`
    # still stops newer issues from being RETURNED, which is what matters.
    bm25 = BM25Okapi([tokenize(issue_text(i)) for i in issues])
    print("embedding issues (cached after the first run)...")
    vectors = embed_all(issues)

    results = {"bm25": [], "dense": [], "hybrid": []}
    print(f"\n{'dup':>7}  {'targets':<20}{'bm25':>6}{'dense':>7}{'hybrid':>8}   (rank of first target, '-' = not in top {TOP})")
    for dup_number, targets in truth.items():
        p = position[dup_number]
        allowed = created < created[p]  # only issues that existed already
        by_bm25 = rank(bm25.get_scores(
            tokenize(issue_text(issues[p]))), numbers, allowed, TOP)
        by_dense = rank(vectors @ vectors[p], numbers, allowed, TOP)
        by_hybrid = rrf([by_bm25, by_dense])[:TOP]

        cells = []
        for name, ranked in (("bm25", by_bm25), ("dense", by_dense), ("hybrid", by_hybrid)):
            results[name].append(hit(ranked, targets, K))
            found = [ranked.index(t) + 1 for t in targets if t in ranked]
            cells.append(str(min(found)) if found else "-")
        print(
            f"#{dup_number:>6}  {str(targets):<20}{cells[0]:>6}{cells[1]:>7}{cells[2]:>8}")

    n = len(truth)
    print(
        f"\nhit@{K} on {n} duplicate pairs (each pair = {100 / n:.1f} points):")
    for name, hits in results.items():
        print(f"  {name:<7} {sum(hits):>2}/{n}  = {sum(hits) / n:.0%}")


if __name__ == "__main__":
    main()
