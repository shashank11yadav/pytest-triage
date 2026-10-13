# pytest-triage

[![CI](https://github.com/shashank11yadav/pytest-triage/actions/workflows/ci.yml/badge.svg)](https://github.com/shashank11yadav/pytest-triage/actions/workflows/ci.yml)
[![Shadow](https://github.com/shashank11yadav/pytest-triage/actions/workflows/shadow.yml/badge.svg)](https://github.com/shashank11yadav/pytest-triage/actions/workflows/shadow.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A triage assistant for [pytest](https://github.com/pytest-dev/pytest) issues. For
each new issue it suggests labels, points out missing information and looks for
older issues that report the same problem. It runs in shadow mode: it reads
pytest's issues and never writes to them, and every suggestion is scored
against what the maintainers actually did.

Live results, updated daily: **https://shashank11yadav.github.io/pytest-triage/**

## Why

pytest has 6,416 issues. Before 2025, maintainers labelled almost 80% of them.
Since January 2025 that has dropped to 49%: 247 of the 484 new issues never
got a label. In a backtest on the 37 issues opened between 1 September and
7 October 2026, 30 were still untriaged when I collected the data.

Triage is unglamorous work, and it's the first thing to slip when volunteers
are busy. I wanted to see how much of it a cheap model could take on, and,
just as important, to be able to say how often it's wrong.

## Results at a glance

All numbers come from `eval/results/*.json`, produced by the commands in
[Running it](#running-it). The test set is 237 issues opened from 2025
onward that maintainers labelled.

| What | Result | Compared with |
|---|---|---|
| Labels (10 labels, macro-F1) | **0.56** | 0.46 for the best free method, a gap of +0.095 (95% CI +0.04 to +0.14) |
| Label precision, adjusted for labels maintainers skip | about **0.65 to 0.75** | 0.43 measured against maintainer labels alone (see [the audit](#how-precise-are-the-labels-really)) |
| Needs more information (F1) | 0.42 | 0.24 for flagging every issue |
| Duplicates found (15 known pairs) | 7 of 15 | 5 of 15 for search alone |
| Duplicate false alarms (30 non-duplicates) | 3 of 30 | |
| Prompt-injection attacks that changed an output | 1 of 19 | |
| Cost and latency per issue | about $0.0015, 3.3 s (median) | 3 calls to Gemini 3.5 Flash-Lite |
| Total spent building and evaluating everything | $1.95 | |

The short version: the model clearly beats anything free, it isn't precise
enough to act on its own, and the duplicate finder is a useful hint that's
wrong half the time. So every output goes to a maintainer as a suggestion,
and nothing goes to a reporter unless it's at least 70% precise. Today nothing
clears that bar, so the reply draft stays empty.

## How it works

Each issue goes through three LLM calls plus local search. The pipeline is
fixed rather than a free-roaming agent, so cost and latency are predictable
and each step can be evaluated on its own.

```mermaid
flowchart TD
    A[New issue] --> B["Redact secrets and paths<br/>fence and datamark the text"]
    B --> C["Assess: 1 call<br/>labels + missing info"]
    B --> D["Expand: 1 call<br/>3 root-cause queries"]
    K["5 similar labelled<br/>past issues"] -. few-shot .-> C
    B --> E["Search older issues<br/>BM25 + embeddings, RRF<br/>issue's top 8 + queries' top 8"]
    D --> E
    E --> F["Verify: 1 call<br/>which are the same problem?"]
    C --> R[Report]
    F --> R
    R --> L[Suggested labels]
    R --> M["For maintainer review<br/>missing info, possible duplicates"]
    R --> P["Reply draft<br/>70%+ precision only"]
```

A few details matter more than they look:

- The model's outputs are constrained to enums (our 10 labels, four
  missing-info items, the candidate issue numbers) and checked again in code,
  so it can't invent a label or point at an issue it wasn't shown.
- Few-shot examples and duplicate candidates only come from issues opened
  before the one being triaged, so nothing leaks from the future.
- Search runs locally: `rank-bm25` for keywords and `fastembed`
  (bge-small-en-v1.5) for meaning. Only the three model calls cost money.
- Every Gemini call goes through one function with a disk cache, a cost
  ledger and a hard spend cap.

## How it's evaluated

The data is split by time, the way the tool would actually be used. Every
choice (prompt version, model, thresholds, ensembles) is made on the dev
year. The test set is scored once per final choice, and the shadow run
scores issues that didn't exist when the code was written.

```mermaid
flowchart LR
    T["Train / example pool<br/>issues before 2024<br/>4,459 labelled"] --> V["Dev: 2024<br/>256 issues<br/>all choices made here"]
    V --> X["Test: 2025 onward<br/>237 issues<br/>scored once per choice"]
    X --> S["Shadow: opened since 2026-10-14<br/>predicted within about a day"]
```

Two more habits:

- Every comparison between methods gets a paired bootstrap confidence
  interval (2,000 resamples). With 11 to 44 examples per label, gaps under
  about 0.03 macro-F1 are noise, and I'd rather say so than report them.
- Before the dev runs, I write down the rule for picking a winner. Prompt v3s,
  for example, had to stay within 0.01 of v2 on dev *and* let fewer attacks
  through.

## Detailed results

### Labels

Test set, 237 issues:

| Predictor | macro-F1 | micro-F1 |
|---|---|---|
| Gemini 3.5 Flash-Lite, prompt v3s (current) | **0.56** | 0.53 |
| Gemini 3.5 Flash-Lite, prompt v2 | 0.55 | 0.53 |
| Gemini 3.5 Flash-Lite, prompt v1 | 0.52 | 0.51 |
| Logistic regression per label on embeddings (free) | 0.46 | 0.42 |
| kNN vote of the 10 most similar past issues (free) | 0.41 | 0.39 |
| Keywords mined from past titles (free) | 0.36 | 0.39 |
| Predict every label | 0.18 | 0.18 |
| Always predict the most common label | 0.03 | 0.18 |

Whether the gaps are real:

| Comparison | Gap | 95% CI |
|---|---|---|
| Current prompt vs. best free method (logistic regression) | +0.095 | +0.044 to +0.142 |
| Logistic regression vs. kNN | +0.055 | +0.002 to +0.108 |
| Prompt v2 vs. v1 | +0.024 | −0.006 to +0.053 (not significant) |
| Prompt v3s vs. v2 | +0.010 | −0.015 to +0.035 (not significant) |

The logistic regression is the fairer bar. Trained classifiers (fine-tuned
transformers, in their case) won the NLBSE issue-classification competitions,
and this simpler one costs nothing to train on embeddings I already have. The
prompt revisions after v1 each moved the score by an amount the test set
can't tell apart from noise.

Per label (F1, test set):

| Label | LLM (v3s) | Logistic regression | kNN | Test examples |
|---|---|---|---|---|
| type: bug | 0.45 | 0.39 | 0.35 | 44 |
| type: question | 0.48 | 0.18 | 0.25 | 12 |
| type: enhancement | 0.38 | 0.25 | 0.25 | 24 |
| type: proposal | 0.47 | 0.40 | 0.31 | 29 |
| type: docs | 0.85 | 0.58 | 0.40 | 22 |
| topic: fixtures | 0.71 | 0.80 | 0.77 | 28 |
| topic: reporting | 0.53 | 0.35 | 0.38 | 27 |
| topic: collection | 0.44 | 0.41 | 0.38 | 22 |
| topic: config | 0.69 | 0.69 | 0.43 | 21 |
| topic: parametrize | 0.55 | 0.56 | 0.54 | 11 |

The pattern holds across labels. The model wins clearly on labels that depend
on what the reporter wants (docs, question, bug). The free methods tie or win
on labels named after a feature (fixtures, parametrize), because the title
usually gives it away. Combining the two looked promising on dev but didn't
hold up on test (−0.001 and −0.008), so the pipeline stays LLM-only.

### How precise are the labels, really?

Measured precision is 0.43, which sounds bad. But maintainers apply about one
of these ten labels per issue while the model suggests about 1.6, and a label
nobody applied isn't necessarily wrong. To check, I sampled 50 of the model's
"false positive" labels at random (prompt v2, seed 11) and had two raters
judge them blind:

| Rater | Judged actually correct | Adjusted precision (95% CI) |
|---|---|---|
| LLM rater A | 59% | 0.76 (0.68 to 0.84) |
| Gemini 3.8 Flash | 51% | 0.72 (0.64 to 0.80) |
| Only if both agree | 38% | 0.65 |

The raters agreed on 34 of 50 (Cohen's kappa 0.41, moderate). Both are LLMs
and no human rated these, so treat 0.65 to 0.75 as an indication rather than a
measurement. Rater A had also seen how the classifier prompt was designed.

Most labels that both raters called wrong fell into three groups:
enhancement versus proposal, `topic: fixtures` on issues about one specific
built-in fixture, and `topic: reporting` on things that aren't terminal
output. Prompt v3s fixed part of the fixtures problem: its precision went from
0.56 to 0.68, with recall falling from 1.00 to 0.75.

### Needs more information

| Predictor | Precision | Recall | F1 |
|---|---|---|---|
| LLM, prompt v3s (current) | 0.48 | 0.38 | 0.42 |
| LLM, prompt v2 | 0.48 | 0.41 | 0.44 |
| Flag every issue | 0.14 | 1.00 | 0.24 |

Stricter decision rules, such as requiring two or more missing items or a
missing reproducer specifically, didn't help on dev. My guess is that the
ground truth is the limit: maintainers often ask for details in a comment
without adding the label.

### Duplicates

The ground truth is 15 pairs taken from maintainer comments ("duplicate of
#N") on the 43 issues closed as duplicates. The other 28 had no usable link:
duplicates of a Discussion, "already fixed", or no comment at all. Thirty
issues closed as completed serve as non-duplicates for counting false alarms.

| Method | Original among candidates | Correctly flagged | False alarms | Precision of flags |
|---|---|---|---|---|
| Hybrid search only, top 5 | 5/15 | | | |
| Expand + verify, one fused candidate list of 8 | 6/15 | 5/15 | 5/30 | 36% |
| Expand + verify, issue's top 8 + queries' top 8 (current) | 10/15 | 7/15 | 3/30 | 50% |

The single fused list let the expansion queries drown out the issue's own
search, so easy duplicates that plain search ranked 4th fell out of the top
100. Keeping both lists side by side fixed that. One caveat: with only 15
pairs, I chose this design on the same data it's scored on, and the interval
for 7/15 runs from 25% to 70%.

The three pairs no search method finds were checked by hand, and all three are
correct. They're root-cause duplicates: the reporter describes a symptom
("`-k` selects the wrong test") and the original describes the mechanism
("marks are turned into keywords"), with almost no words in common.

### Prompt injection

Issue text is written by strangers and goes straight into the prompt, so I
treat it as hostile. The suite has 19 attacks, each paired with a control
issue that's identical apart from the injected text. An attack only counts if
it changes the output compared with its control.

| Prompt | Attacks that changed an output |
|---|---|
| v2 | 2 of 19: a fake "maintainer note" relabelled a crash report as docs, and an order to list a reproducer as missing worked |
| v3s with spotlighting (current) | 1 of 19: the fake maintainer note still works |

The blocked attacks include:
- base64-encoded instructions
- hidden HTML comments
- instructions in Spanish and Chinese
- a fake closing tag written with a Cyrillic look-alike letter
- a fake JSON answer
- an instruction planted in an older issue, so it reaches the duplicate
  verifier as a candidate

The detector flags 10 of the 19 for a human. I wrote the attacks myself and
each runs once, so this is weaker evidence than it looks; the structural
limits below matter more.

## Guardrails

```mermaid
flowchart TD
    I[Issue text] --> R["Redact<br/>tokens, keys, emails, usernames in paths"]
    I --> D["Injection detector<br/>issue and its duplicate candidates"]
    R --> F["Fence + datamark<br/>text can't close its tag"]
    F --> M[Model]
    M --> V["Validate<br/>enum outputs, re-checked in code"]
    V --> G["Gate<br/>reply only from outputs at 70%+ precision"]
    G --> H[Maintainer]
    D --> H
```

- Redaction runs before any text leaves the machine. It removes GitHub and API
  tokens, email addresses, and usernames inside paths like `/Users/alice/...`.
- The issue sits inside `<issue>` tags it can't close. Every space is replaced
  with a marker character, so the model can tell data from instructions
  throughout the text and not just at the edges. This is the "datamarking"
  form of spotlighting from [Hines et al., 2024](https://arxiv.org/abs/2403.14720).
- Labels, missing-info items and candidate numbers are enums. The only free
  text, the verifier's reasons, is capped at 200 characters with links and
  @mentions stripped.
- Only outputs measured at 70% precision or better can go into text meant for
  the reporter, and that text is a template. The model never writes prose that
  reaches anyone.
- Nothing in the code can post, comment or label on GitHub.
- One code path talks to Gemini, with a cache, a per-call ledger and a hard
  spend cap. Answers cut off mid-output are never cached.

## Shadow mode

```mermaid
flowchart LR
    C["GitHub Actions<br/>daily, 06:17 UTC"] --> F["Fetch new pytest issues<br/>(read only)"]
    F --> T["Triage issues opened since<br/>2026-10-14, not yet predicted"]
    T --> S["Re-score every prediction<br/>against maintainers' labels so far"]
    S --> P["Commit predictions and<br/>update the results page"]
```

Predictions are committed the day they're made, so they're timestamped and
can't be edited later. Each one also records which maintainer labels already
existed at that moment, to show how often the tool got there first. The job
has its own $1.00 cap and costs about $0.05 a month.

An earlier backtest is kept separately in `eval/shadow/backtest.jsonl` and is
never mixed in. It covered 37 issues, predicted after the fact with an older
pipeline.

## What didn't work

- **A bigger model.** Gemini 3.8 Flash scored 0.56 against 0.535 on dev, at
  2.7 times the cost per call, and was worse at spotting missing information.
- **Ensembles of the LLM and the trained classifier.** About +0.02 on dev,
  nothing on test.
- **Local cross-encoder rerankers for duplicates** (bge-reranker-v2-m3 and
  MiniLM). Both found 8 of 15 originals against 10 for the current pool, at 6
  to 65 seconds per issue. They're trained on web search queries, not on "is
  this the same bug". The wide candidate pool already contains 14 of 15
  originals, so the open problem is picking the right one, not finding it.
- **Prompt caching.** Explicit caching isn't available on the Gemini
  Interactions API, and implicit caching never triggered: the prompts (about
  2,000 tokens) are below its minimum.

I also hit two undocumented API behaviours along the way:
- Gemini 3.8 Flash rejects the "minimal" thinking level, and its thinking
  tokens count against `max_output_tokens`, which silently cut answers short.
- Integer enum values with seven or more digits make the Interactions API
  return a misleading "Invalid JSON payload" error. I found that one by
  bisecting the request.

## Limitations

- The test sets are small: 11 to 44 examples per label, 15 duplicate pairs,
  and 19 attacks run once each.
- Maintainer labels are incomplete ground truth. The audit that corrects for
  that used two LLM raters with moderate agreement and no human.
- The duplicate improvements were measured on the data used to design them.
- No pytest maintainer has used it yet. Usefulness is inferred from metrics,
  not observed; the shadow run is the first step toward changing that.

## Running it

You need Python 3.12+ and [uv](https://docs.astral.sh/uv/). Put
`GEMINI_API_KEY=...` in a `.env` file at the repo root. GitHub access uses
`GITHUB_TOKEN` from `.env` if present, otherwise your `gh auth login` session.

```bash
uv sync
uv run triage fetch              # download or refresh pytest's issues
uv run triage stats              # snapshot and split summary
uv run triage issue 14476        # triage one issue and print the report
```

Evaluations (the ones marked free use cached or local results only):

```bash
uv run triage eval labels                     # free baselines
uv run triage eval labels --llm               # + Gemini, about $0.15
uv run triage eval labels --split dev --llm   # where every choice is made
uv run triage eval learned --phase dev        # trained classifier + ensembles (free)
uv run triage eval significance               # bootstrap intervals (free)
uv run triage eval dupes --llm                # duplicate verifier
uv run triage eval rerank                     # rerankers vs. current pool (free)
uv run triage eval attacks                    # prompt-injection suite
uv run triage audit score                     # label audit
uv run triage shadow && uv run triage shadow-score
uv run triage report                          # rebuild docs/index.html
```

Tests run offline and never touch the network or cost money:

```bash
uv run pytest
uv run ruff check .
```

There's also a Dockerfile:
`docker build -t pytest-triage . && docker run --rm --env-file .env -v "$PWD/data:/app/data" pytest-triage stats`

## Repository layout

```
src/pytest_triage/
  github.py, store.py       read-only GitHub client and local issue snapshot
  dataset.py, metrics.py    time splits, P/R/F1, bootstrap and Wilson intervals
  search.py, baselines.py   BM25 + embeddings + RRF; free predictors
  learned.py                logistic-regression label classifier
  llm.py                    the only code that calls Gemini: cache, ledger, cap
  classify.py, dedupe.py    labels + missing info; duplicate expansion and check
  guard.py, attacks.py      guardrails and the injection suite
  triage.py, shadow.py      per-issue pipeline; daily shadow run and scoring
  audit.py                  blind rating of "false positive" labels
  evaluate.py, report.py    evaluation runners and the results page
tests/                      offline tests
eval/                       results, ground truth and shadow predictions
                            (issue numbers and metrics only, never issue text)
```

Raw issue text, embeddings, the LLM cache and the cost ledger live in `data/`,
which is gitignored.

## License

MIT. See [LICENSE](LICENSE).
