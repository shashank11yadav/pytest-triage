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
DEV_DATE = "2024-01-01"  # memory issues from here to SPLIT_DATE = dev set for choosing variants

# Top 5 `type:` + top 5 `topic:` labels by count (snapshot of 2026-10-08), plus
# `type: regression` (added 2026-10-14: the type-label check found the model calling
# regressions "bug"; 171 memory examples). Prompts v1-v3s predate it and only ever
# predict the first ten (classify.ORIGINAL_LABELS); all scoring uses all eleven.
LABELS = (
    "type: bug",
    "type: question",
    "type: enhancement",
    "type: proposal",
    "type: docs",
    "type: regression",
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
# What the live pipeline (`triage issue`, `triage shadow`) uses. Chosen on the
# 2024 dev split, never on test - see eval/results/labels_dev.json.
# Dev results (2026-10-12): flash-lite v2 0.535 macro-F1 vs. 3.8-flash v2 0.56 at 2.7x
# the cost and worse needs-info; duplicate verifier precision 36% (lite) vs. 42%
# (3.8-flash), a one-false-alarm difference on 45 issues. Cheap model wins both.
# Prompt v3s (2026-10-13): clarified fixtures/reporting definitions + spotlighting.
# Chosen by a rule fixed before seeing dev results (within 0.01 of v2 on dev AND fewer
# attacks landed): dev 0.55 vs 0.535, attacks 1/19 vs 2/19. Test: +0.010 vs v2
# (CI -0.015..+0.035, not significant); needs-info F1 0.42 vs 0.44.
LABEL_MODEL = WORKHORSE
LABEL_PROMPT = "v4"  # adopted 2026-10-14 by the rule in classify.py (dev results there)
DEDUPE_MODEL = WORKHORSE

# Lowest thinking level each model accepts (3.8-flash rejects "minimal" with a 400,
# verified 2026-10-12). Thinking tokens count against max_output_tokens, so thinking
# models get extra output headroom or the answer is cut off (status "incomplete").
THINKING = {WORKHORSE: "minimal", STEP_UP: "low"}
THINKING_HEADROOM = {WORKHORSE: 0, STEP_UP: 1024}

# The reply draft is text a maintainer could send to the REPORTER. An output goes in
# only if its measured precision on the test set clears this bar; below it, the
# output is shown to the maintainer for review instead. Measured 2026-10-12:
# needs-info precision 0.48; duplicate flags 0.50 with the union candidate pool
# (0.36 before) - so today no reply is drafted.
REPLY_MIN_PRECISION = 0.70
# High-precision mode (on by default): a label is shown only if the trained classifier
# gives it at least this probability. Chosen on dev by `triage eval selective` as the
# lowest threshold reaching REPLY_MIN_PRECISION. None turns the gate off.
LABEL_MIN_CONFIDENCE: float | None = 0.90
MEASURED_PRECISION = {"missing": 0.48, "duplicates": 0.50}

# Whole project; calls past this are refused, not made. The scheduled shadow job sets
# a lower cap through the environment (its ledger lives in the Actions cache).
SPEND_CAP_USD = float(os.environ.get("TRIAGE_SPEND_CAP_USD", "5.00"))

# Prospective shadow mode: issues opened on/after this date are triaged by the daily
# job, within ~24h of opening. The earlier backtest (eval/shadow/backtest.jsonl) was
# predicted after the fact with an older pipeline and is kept separate.
SHADOW_START = "2026-10-14"


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
    # Pasted secrets often carry a newline, spaces or quotes; none belong in a key.
    if key := os.environ.get("GEMINI_API_KEY", "").strip().strip("'\""):
        return key
    raise RuntimeError("No Gemini key: put GEMINI_API_KEY=... in .env at the repo root.")
