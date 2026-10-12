"""Find similar past issues: BM25 (exact words) + embeddings (meaning), fused with RRF.

    BM25        good at exact tokens (error names, option names), blind to paraphrase
    embeddings  good at paraphrase, vague on rare exact tokens
    RRF         each ranking votes 1/(k + position); agreement beats one strong opinion

Embeddings run locally (fastembed) and are cached in data/vectors.npz keyed by a
hash of each issue's text, so incremental fetches only embed new or edited issues.
"""

import hashlib
import re

import numpy as np
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi

from pytest_triage.config import EMBED_MODEL, MAX_BODY_CHARS, RRF_K, SEARCH_DEPTH, VECTORS
from pytest_triage.dataset import issue_text

_model: TextEmbedding | None = None


def tokenize(text: str) -> list[str]:
    """BM25 matches tokens literally, so this IS the keyword-matching rule."""
    return re.findall(r"\w+", text.lower())


def _text_hash(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def embed_texts(texts: list[str], progress: bool = False) -> np.ndarray:
    """Unit-length vectors (so a dot product IS cosine similarity)."""
    global _model
    if _model is None:
        _model = TextEmbedding(EMBED_MODEL)
    rows = []
    # embed() (not query_embed/passage_embed): issue-to-issue search is symmetric.
    for row in _model.embed(texts):
        rows.append(row)
        if progress and len(rows) % 500 == 0:
            print(f"  embedded {len(rows)}/{len(texts)}", flush=True)
    if not rows:
        return np.zeros((0, 384), dtype=np.float32)
    vectors = np.array(rows, dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def load_vectors(issues: list[dict]) -> np.ndarray:
    """One vector per issue, in order. Only new or edited texts get embedded."""
    texts = [issue_text(i, MAX_BODY_CHARS) for i in issues]
    hashes = [_text_hash(t) for t in texts]

    cached: dict[str, np.ndarray] = {}
    if VECTORS.exists():
        saved = np.load(VECTORS)
        if str(saved["model"]) == EMBED_MODEL:
            cached = dict(zip(saved["hashes"].tolist(), saved["vectors"], strict=True))

    missing = sorted({h for h in hashes if h not in cached})
    if missing:
        print(f"embedding {len(missing)} new/changed issues...", flush=True)
        first_text = dict(zip(hashes, texts, strict=True))
        new = embed_texts([first_text[h] for h in missing], progress=True)
        cached.update(zip(missing, new, strict=True))
        VECTORS.parent.mkdir(parents=True, exist_ok=True)
        keys = list(cached)
        np.savez(
            VECTORS,
            hashes=np.array(keys),
            vectors=np.stack([cached[k] for k in keys]),
            model=np.array(EMBED_MODEL),
        )
    return np.stack([cached[h] for h in hashes])


def rank(scores: np.ndarray, numbers: list[int], allowed: np.ndarray, top: int) -> list[int]:
    """Issue numbers of the `top` best-scoring allowed rows, best first."""
    rows = np.flatnonzero(allowed)
    best = rows[np.argsort(-scores[rows], kind="stable")][:top]
    return [numbers[r] for r in best]


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    totals: dict[int, float] = {}
    for ranking in rankings:
        for position, number in enumerate(ranking, start=1):
            totals[number] = totals.get(number, 0.0) + 1 / (k + position)
    return sorted(totals, key=totals.__getitem__, reverse=True)


class IssueIndex:
    """Searchable set of issues. Queries only ever see issues created before them."""

    def __init__(self, issues: list[dict], vectors: np.ndarray):
        self.issues = issues
        self.numbers = [i["number"] for i in issues]
        self.numbers_array = np.array(self.numbers)
        self.position = {n: p for p, n in enumerate(self.numbers)}
        self.created = np.array([i["created_at"] for i in issues])
        # Simplification: BM25's word-rarity stats include issues newer than a query.
        # The `allowed` mask still stops newer issues from being returned.
        self.bm25 = BM25Okapi([tokenize(issue_text(i, MAX_BODY_CHARS)) for i in issues])
        self.vectors = vectors

    @classmethod
    def build(cls, issues: list[dict]) -> "IssueIndex":
        return cls(issues, load_vectors(issues))

    def get(self, number: int) -> dict | None:
        p = self.position.get(number)
        return None if p is None else self.issues[p]

    def similar(
        self, issue: dict, k: int = SEARCH_DEPTH, method: str = "hybrid", before: str | None = None
    ) -> list[int]:
        """Numbers of the k most similar issues created before `before`
        (default: the query's own created_at). The query itself is never returned."""
        text = issue_text(issue, MAX_BODY_CHARS)
        allowed = self.allowed_for(issue, before)

        rankings = []
        if method in ("bm25", "hybrid"):
            scores = self.bm25.get_scores(tokenize(text))
            rankings.append(rank(scores, self.numbers, allowed, SEARCH_DEPTH))
        if method in ("dense", "hybrid"):
            p = self.position.get(issue["number"])
            vec = self.vectors[p] if p is not None else embed_texts([text])[0]
            rankings.append(rank(self.vectors @ vec, self.numbers, allowed, SEARCH_DEPTH))
        if not rankings:
            raise ValueError(f"unknown method {method!r}")
        return (rrf(rankings) if len(rankings) > 1 else rankings[0])[:k]

    def allowed_for(self, issue: dict, before: str | None = None) -> np.ndarray:
        """Mask of issues a query may see: older than the cutoff, and not itself."""
        cutoff = before or issue["created_at"]
        return (self.created < cutoff) & (self.numbers_array != issue["number"])

    def search_texts(self, texts: list[str], allowed: np.ndarray) -> list[list[int]]:
        """A BM25 ranking and a dense ranking for each free-text query."""
        rankings = []
        for text, vec in zip(texts, embed_texts(texts), strict=True):
            rankings.append(
                rank(self.bm25.get_scores(tokenize(text)), self.numbers, allowed, SEARCH_DEPTH)
            )
            rankings.append(rank(self.vectors @ vec, self.numbers, allowed, SEARCH_DEPTH))
        return rankings
