"""Every tunable number, path and model name, in one place.

Paths are relative to the working directory: run `triage ...` from the repo root.
"""

import os
import shutil
import subprocess
from pathlib import Path

from dotenv import load_dotenv

REPO = "pytest-dev/pytest"

# --- paths ------------------------------------------------------------------
DATA = Path("data")  # gitignored: raw issue text, caches, ledger, traces
EVAL = Path("eval")  # committed: issue numbers, labels, metrics only

ISSUES = DATA / "issues.jsonl"
VECTORS = DATA / "vectors.npz"
LLM_CACHE = DATA / "llm_cache"
LEDGER = DATA / "ledger.jsonl"
TRACES = DATA / "traces.jsonl"

DUPLICATES = EVAL / "duplicates.json"
RESULTS = EVAL / "results"
SHADOW = EVAL / "shadow"

# --- dataset ----------------------------------------------------------------
SPLIT_DATE = "2025-01-01"  # created before -> memory, on/after -> test

# Top 5 `type:` + top 5 `topic:` labels by count (snapshot of 2026-10-08).
# Each has 173+ memory examples; rarer labels are too thin to learn or score.
LABELS = (
    "type: bug",
    "type: question",
    "type: enhancement",
    "type: proposal",
    "type: docs",
    "topic: fixtures",
    "topic: reporting",
    "topic: collection",
    "topic: config",
    "topic: parametrize",
)

# --- search -----------------------------------------------------------------
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # fastembed default, 384 dims, runs locally
MAX_BODY_CHARS = 500  # the rest of a body is mostly tracebacks
SEARCH_DEPTH = 100  # candidates per retriever before fusion
RRF_K = 60

# --- LLM --------------------------------------------------------------------
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
WORKHORSE = "gemini-3.5-flash-lite"
STEP_UP = "gemini-3.8-flash"

# USD per 1M tokens, standard paid tier (https://ai.google.dev/gemini-api/docs/pricing,
# checked 2026-10-12). Cached input is billed at "cached"; thinking tokens at "output".
PRICES = {
    WORKHORSE: {"input": 0.30, "cached": 0.03, "output": 2.50},
    # Promo through 2026-12-31; doubles to 1.50 / 0.15 / 7.50 on 2027-01-01.
    STEP_UP: {"input": 0.75, "cached": 0.075, "output": 3.75},
}
SPEND_CAP_USD = 5.00  # whole project; calls past this are refused, not made


# --- secrets ----------------------------------------------------------------
def github_token() -> str:
    """GITHUB_TOKEN from the environment / .env, else ask the `gh` CLI."""
    load_dotenv(Path(".env"))
    if token := os.environ.get("GITHUB_TOKEN"):
        return token
    if shutil.which("gh"):
        result = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    raise RuntimeError("No GitHub token: set GITHUB_TOKEN or run `gh auth login`.")


def gemini_key() -> str:
    """GEMINI_API_KEY from the environment or ./.env."""
    load_dotenv(Path(".env"))
    if key := os.environ.get("GEMINI_API_KEY"):
        return key
    raise RuntimeError("No Gemini key: put GEMINI_API_KEY=... in .env at the repo root.")
