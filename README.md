# pytest-triage

**A shadow-mode triage assistant for [pytest-dev/pytest](https://github.com/pytest-dev/pytest) issues,
evaluated against what the maintainers actually did.**

For each new issue it suggests labels, flags missing information, finds possible
duplicates for a maintainer to review, and drafts a reply. It reads from GitHub
and **never writes to it**.

[![CI](https://github.com/OWNER/pytest-triage/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)

## Why

pytest has 6,416 issues. Maintainers used to label nearly all of them. Since
2025, **51% of new issues are never labelled** (it was 21% before). In a
backtest on the 37 issues opened between 1 September and 7 October 2026,
**30 were still untriaged** when the data was collected. This tool's job is
to give every one of them a first pass.

## Results

Each number below is produced by `uv run triage report` from `eval/results/`. None is typed by hand.

### Labels: 10 labels, 237 test issues created after 2025-01-01

| Predictor | macro-F1 | micro-F1 |
|---|---|---|
| **Gemini 3.5 Flash-Lite, prompt v2 (chosen on dev)** | **0.55** | **0.53** |
| Gemini 3.5 Flash-Lite, prompt v1 | 0.52 | 0.51 |
| kNN label vote of similar past issues (free) | 0.41 | 0.39 |
| Keywords mined from past titles (free) | 0.36 | 0.39 |
| Predict every label | 0.18 | 0.18 |
| Always predict the most common label | 0.03 | 0.18 |

The LLM's biggest gains over kNN are on labels that depend on **intent**:
`type: docs` 0.80 vs. 0.40, `type: question` 0.50 vs. 0.25, and
`topic: config` 0.69 vs. 0.43. On labels named by **vocabulary**, the free
kNN vote ties it or wins: `topic: fixtures` 0.77 vs. 0.72 and
`topic: parametrize` 0.54 vs. 0.48.

### Missing information: does the issue need `status: needs information`?

| Predictor | Precision | Recall | F1 |
|---|---|---|---|
| **LLM, prompt v2** | 0.48 | 0.41 | **0.44** |
| Always flag | 0.14 | 1.00 | 0.24 |

### Duplicates: 15 known pairs, plus 30 non-duplicates to measure false alarms

| Method | Result | 95% CI |
|---|---|---|
| Hybrid search (BM25 + embeddings, RRF) | original in top 5 for 5/15 | 15–58% |
| + LLM query expansion and verification (Flash-Lite) | 5/15 correctly flagged; **5/30 false alarms** | 15–58% |
| same, with Gemini 3.8 Flash | 5/15 correctly flagged; 4/30 false alarms | 15–58% |

About 40% of duplicate flags are correct. That's **not good enough to tell a
reporter "this is a duplicate"**, so duplicates appear only in the
maintainer's view and never in the reply draft. In the September–October
backtest, the pipeline had flagged 6 of the 7 issues maintainers later closed
as duplicates. It also flagged 6 that were not closed as duplicates. (That
check shows a flag was raised, not that the suggested target was right.)

### Prompt injection: 4 attacks, each paired with a control issue

0 of 4 landed. The attacks tried to:
- relabel an issue by instruction
- forge the `<issue>` fence with a fake system message
- push a fake duplicate claim
- smuggle a link and an @mention into model-written text

### Cost and latency

- **About $0.0015 per issue** (3 Gemini calls), with a p50 latency of 3.3 s.
- The entire development spend, including every evaluation above, was **$1.17**.

## How it works

```
new issue
  │
  ├─ redact secrets and home paths, fence as untrusted  (guard.py)
  │
  ├─ assess: 1 Gemini call                              (classify.py)
  │     few-shot = 5 most similar labelled past issues + their labels
  │     output   = enum-constrained JSON: labels + missing-info items
  │
  ├─ duplicates: 2 Gemini calls                         (dedupe.py)
  │     expand  → 3 maintainer-style search queries (root cause, not symptoms)
  │     search  → BM25 + embeddings, fused with RRF, older issues only
  │     verify  → which of 8 candidates is the same problem? (enum of candidate numbers)
  │
  └─ report: labels, missing info, possible duplicates (maintainer only),
             injection signals, templated reply draft, trace (latency, cost)
```

## Design decisions

- **Split by time, not at random.** Test issues are all newer than everything
  the system learns from, as in real use. Few-shot examples and duplicate
  candidates come only from issues created before the query.
- **Variants are chosen on a dev split, and test is run once.** Issues from
  2024 decided prompt v1 vs. v2 and Flash-Lite vs. 3.8 Flash. The test set saw
  v1 (the first design) and then the dev winner. v2's rules were prompted by
  v1's error pattern, which appeared on both splits.
- **The cheap model won on evidence.** Gemini 3.8 Flash scored 0.56 vs. 0.535
  dev macro-F1, at 2.7× the cost per call and with a worse needs-info score.
- **Free baselines first.** A kNN vote over embeddings costs nothing and
  scores 0.41. An LLM that can't beat it isn't worth paying for.
- **One call does two jobs.** Labels and missing information come from the
  same request.
- **Untrusted input, in layers.**
  - Redaction, then a fence that issue text can't close.
  - Enum-constrained outputs, re-validated in code.
  - Model-written reasons are length-capped and stripped of links and
    @mentions.
  - The reply is a **template**, so model prose never reaches the reporter.
  - Nothing can post to GitHub.
- **Stateless requests (`store: false`).** Each issue is independent, so
  server-side history would buy nothing and would keep strangers' text at
  Google. Explicit caching isn't available on the Interactions API, and
  implicit caching never triggered: the prompts (about 2,000 tokens) are too
  short. That was measured, not assumed.
- **Cost is enforced in code.** Every call goes through one function with a
  disk cache, a cost ledger and a hard spend cap. Responses with status
  `incomplete` are never cached. 3.8 Flash's thinking tokens count against
  `max_output_tokens`, which silently truncated answers until handled.
- **Duplicate ground truth is silver and small.** Targets come from maintainer
  comments ("duplicate of #N"): 15 usable pairs out of 43 closed duplicates.
  Scoring is cluster-aware: A flagged as a duplicate of B counts when both
  duplicate C. Every duplicate result carries its 95% interval. The 3 pairs
  no search method finds were checked by hand. All are correct
  **root-cause** duplicates, where the reporter's symptom shares almost no
  words with the original's description of the mechanism.

## Limitations

- **The test sets are small.** 237 issues for labels (11–44 per label), and
  15 pairs for duplicates. Differences under about 0.03 macro-F1 are noise.
- **Maintainer labels are not perfect ground truth.** Many issues are never
  labelled, and some labels change after the fact.
- **The backtest is not a true prospective run.** The issues are recent, but
  the predictions were made after they were opened. A scheduled run
  (`triage shadow`) is the honest version, and needs weeks of data.

## Run it

```bash
uv sync
uv run triage fetch                 # download / refresh issues (incremental)
uv run triage stats                 # snapshot + split summary
uv run triage eval labels           # free baselines
uv run triage eval labels --llm     # + Gemini (costs about $0.13)
uv run triage eval dupes [--llm]    # duplicate search / verifier
uv run triage eval attacks          # prompt-injection suite
uv run triage issue 14476           # triage one issue
uv run triage shadow --since 2026-10-01 && uv run triage shadow-score
uv run triage report                # docs/index.html + these tables
uv run pytest && uv run ruff check .
```

Requirements:
- **GitHub token:** either `GITHUB_TOKEN` in `.env`, or `gh auth login`.
- **LLM commands:** `GEMINI_API_KEY` in `.env`.

Docker: `docker build -t pytest-triage . && docker run --rm --env-file .env -v "$PWD/data:/app/data" pytest-triage stats`

## Layout

```
src/pytest_triage/
  github.py store.py        read-only GitHub client, JSONL snapshot
  dataset.py metrics.py     splits, targets, P/R/F1, hit@k, Wilson intervals
  search.py baselines.py    BM25 + embeddings + RRF; free predictors
  llm.py                    the only door to Gemini: cache, ledger, spend cap
  classify.py dedupe.py     labels + missing info; duplicate expansion/verification
  guard.py attacks.py       guardrails and the injection suite
  triage.py shadow.py       per-issue pipeline; shadow-mode run and scoring
  evaluate.py report.py     eval runners; this README's tables and docs/index.html
tests/                      offline tests: no network, no cost
eval/                       committed results: issue numbers and metrics only
```

Raw issue text, embeddings, the LLM cache and the cost ledger live in
`data/`, which is gitignored.
