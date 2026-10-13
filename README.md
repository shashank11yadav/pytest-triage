# pytest-triage

**A shadow-mode triage assistant for [pytest-dev/pytest](https://github.com/pytest-dev/pytest) issues,
evaluated against what the maintainers actually did.**

For each new issue it suggests labels, flags missing information, finds possible
duplicates for a maintainer to review, and drafts a reply. It reads from GitHub
and **never writes to it**.

[![CI](https://github.com/shashank11yadav/pytest-triage/actions/workflows/ci.yml/badge.svg)](https://github.com/shashank11yadav/pytest-triage/actions/workflows/ci.yml)
[![Shadow](https://github.com/shashank11yadav/pytest-triage/actions/workflows/shadow.yml/badge.svg)](https://github.com/shashank11yadav/pytest-triage/actions/workflows/shadow.yml)

**Live results:** https://shashank11yadav.github.io/pytest-triage/

## Why

pytest has 6,416 issues. Maintainers used to label nearly all of them. Since
2025, **51% of new issues are never labelled** (it was 21% before). In a
backtest on the 37 issues opened between 1 September and 7 October 2026,
**30 were still untriaged** when the data was collected. This tool's job is
to give every one of them a first pass.

## Results

Each number below is produced from `eval/results/` by the `triage eval …` and
`triage report` commands. None is typed by hand.

### Labels: 10 labels, 237 test issues created after 2025-01-01

| Predictor | macro-F1 | micro-F1 |
|---|---|---|
| **Gemini 3.5 Flash-Lite, prompt v3s (current; chosen on dev)** | **0.56** | **0.53** |
| Gemini 3.5 Flash-Lite, prompt v2 | 0.55 | 0.53 |
| Gemini 3.5 Flash-Lite, prompt v1 | 0.52 | 0.51 |
| **Logistic regression per label on embeddings, trained on past labels (free)** | **0.46** | 0.42 |
| kNN label vote of similar past issues (free) | 0.41 | 0.39 |
| Keywords mined from past titles (free) | 0.36 | 0.39 |
| Predict every label | 0.18 | 0.18 |
| Always predict the most common label | 0.03 | 0.18 |

**Is the gap real?** These are paired bootstrap intervals (2,000 resamples of
the test issues, from `triage eval significance`):

| Comparison | macro-F1 gap | 95% CI |
|---|---|---|
| LLM v1 vs. free kNN | **+0.12** | +0.06 to +0.17 |
| LLM v2 vs. free kNN | +0.14 | +0.08 to +0.20 |
| LLM v2 vs. LLM v1 | +0.02 | −0.01 to +0.05 (**not significant**) |
| LLM v3s vs. LLM v2 | +0.01 | −0.015 to +0.035 (**not significant**) |
| **LLM v2 vs. trained classifier (best free method)** | **+0.085** | +0.03 to +0.14 |
| Trained classifier vs. kNN | +0.055 | +0.00 to +0.11 |

**The honest headline is +0.085 over the best free method.** That method is
a per-label logistic regression trained on the 4,459 labelled issues before
2024, with each label's threshold tuned on 2024. Issue-classification
competitions are usually won by trained models like this one, so it is the
baseline an LLM has to justify its cost against. Against kNN the gap looks
bigger (+0.12 for v1), but kNN is the weaker baseline. Each prompt revision
since v1 (v2, then v3s) is within noise of the previous one.

**What the prompt versions change:**
- **v2** adds maintainers' distinctions: bug vs. user error, proposal vs.
  enhancement, and fewer labels when unsure.
- **v3s** clarifies two definitions the audit flagged:
  - `topic: fixtures` means the fixture *mechanism*, not built-in plugin
    fixtures.
  - `topic: reporting` means terminal output.

  It also adds **spotlighting** (see Prompt injection).

On test, v3s raised `topic: fixtures` precision from 0.56 to 0.68 but lowered
its recall from 1.00 to 0.75, so its F1 didn't move. v3s was chosen by a
rule written down before the dev run: "within 0.01 of v2 on dev, and fewer
attacks land". On dev it scored 0.55 vs. 0.535, with 1/19 attacks landing vs.
2/19.

The LLM's biggest gains are on labels that depend on **intent**:
`type: docs` 0.80 vs. 0.40, `type: question` 0.50 vs. 0.25, and
`topic: config` 0.69 vs. 0.43. On labels named by **vocabulary**, the free
kNN vote ties it or wins: `topic: fixtures` 0.77 vs. 0.72 and
`topic: parametrize` 0.54 vs. 0.48.

**Precision is understated by incomplete maintainer labels.** The model
suggests 1.66 labels per issue, while maintainers apply 1.01 of these 10, so
measured precision is only 0.43. An audit (`triage audit`) took 50 random
"false positive" labels (seed 11, out of 225) and had two raters judge, blind
to each other, whether each label actually fits the issue:

| Rater | "False positives" judged correct | Adjusted precision (95% CI) |
|---|---|---|
| LLM rater A | 59% | 0.76 (0.68–0.84) |
| Gemini 3.8 Flash (AI) | 51% | 0.72 (0.64–0.80) |
| Strictest: correct only if both agree | 38% | 0.65 |

The raters agreed on 34 of 50 items (Cohen's kappa 0.41, moderate).
**Both raters are LLMs, and no human rated these.** Gemini is judging labels
from its own model family. Rater A was not blind to how the classifier
prompt was designed. Read **about 0.65–0.75** as an indication of real
precision, not a measurement.

The labels both raters called wrong fall into three patterns:
- **Type-family conflicts.** Maintainers chose `proposal` and the model said
  `enhancement`, or maintainers chose `enhancement`/`docs` and the model said
  `bug`.
- **`topic: fixtures` on built-in plugin fixtures** (`monkeypatch`, `tmp_path`,
  `capfd`), which maintainers label with `plugin:` labels instead.
- **`topic: reporting` applied to output-adjacent but non-terminal issues.**

### Missing information: does the issue need `status: needs information`?

| Predictor | Precision | Recall | F1 |
|---|---|---|---|
| **LLM, prompt v3s (current)** | 0.48 | 0.38 | **0.42** |
| LLM, prompt v2 | 0.48 | 0.41 | 0.44 |
| Always flag | 0.14 | 1.00 | 0.24 |

Stricter decision rules (for example, at least 2 missing items) were tried on
dev. None beat this, so the limit is the signal and the ground truth, not the
threshold.

### Duplicates: 15 known pairs, plus 30 non-duplicates to measure false alarms

| Method | Original among candidates | Correctly flagged | False alarms | Flag precision |
|---|---|---|---|---|
| Hybrid search only (top 5) | 5/15 | – | – | – |
| LLM expansion + verify, fused pool (8) | 6/15 | 5/15 | 5/30 | 36% |
| **LLM expansion + verify, union pool (8 + 8)** | **10/15** | **7/15** | **3/30** | **50%** |

The fused pool let the expansion queries drown out the issue's own search:
easy duplicates ranked 4th fell out of the top 100. The **union pool** keeps
the issue's own top 8 *and* adds the expansion queries' top 8. That recovers
the easy matches and adds root-cause ones only the queries find. Caveat: the
union design was chosen on these same 15 pairs, because there are too few
duplicates for a separate dev set. The 95% CI for 7/15 is 25–70%.

**Reply rule.** An output goes into the reply draft (text a maintainer could
send to the reporter) only if its measured precision is at least **0.70**.
Needs-info (0.48) and duplicates (0.50) are both below that bar, so today
both appear only in a "for maintainer review" section, and no reply is
drafted. The bar and the measured values are in `config.py`.

### Prompt injection: 19 attacks, each paired with a control issue

**Spotlighting** (Hines et al., 2024) is applied in prompt v3s. The
untrusted issue text and the few-shot examples are *datamarked*: every space
is replaced with `ˆ`. The model is told that marked text is data, and that
claims inside it about labels, triage decisions, maintainer notes or missing
information are never instructions.

| Prompt | Attacks landed | Which |
|---|---|---|
| v2 | 2/19 | fake maintainer note; forced missing-info request |
| v3 (definitions only) | 2/19 | hidden HTML comment; forced missing-info request |
| **v3s (definitions + spotlighting, current)** | **1/19** | fake maintainer note |

Spotlighting blocked the forced missing-info request. However, **the fake
maintainer note still landed: a crash report got `type: docs`**, even though
that is exactly the attack spotlighting was aimed at. Each attack runs once,
so 1 vs. 2 is weak evidence. All three landed attacks are flagged for a
human by the injection detector, and their impact is low: labels are
suggestions, and missing-info is maintainer-only.

Blocked:
- plain instructions
- base64-encoded instructions
- hidden HTML comments
- Spanish and Chinese instructions
- a role-play persona
- an instruction in the title
- a fake JSON answer
- label flooding
- forged fences, including a Cyrillic look-alike
- "nothing is missing" flattery
- a duplicate claim
- an **injection planted in an older issue** that reaches the verifier as a candidate
- suppression of a real duplicate
- link smuggling
- system-prompt exfiltration

The injection detector now scans the issue *and* its duplicate candidates,
and it flags 10 of 19 attacks. The patterns added after this suite fire on at
most 2 of 6,416 real issues; across all patterns, 6 real issues trigger it.

Two bugs in the suite itself were found and fixed:
- The first planted issue was never retrieved, so its "blocked" was
  meaningless. Every case now reports **NOT REACHED** when an attack never
  reaches the model.
- Fake issue numbers of 7+ digits made the API reject the request (see
  below).

Caveat: the attacks were written by the same author as the defences. The
structural guarantees matter more than this score: enum outputs, a
templated and precision-gated reply, and no write access.

### Cost and latency

- **About $0.0015 per issue** (3 Gemini calls), with a p50 latency of 3.3 s.
- The entire development spend, including every evaluation above, was **$1.95**.

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
  │     search  → own top 8 (BM25 + embeddings, RRF) ∪ queries' top 8; older issues only
  │     verify  → which candidates are the same problem? (enum of candidate numbers)
  │
  └─ report: suggested labels; for maintainer review: possible missing info and
             possible duplicates; injection signals (issue and candidates);
             reply draft (only outputs over the 0.70 precision bar); trace
```

## Design decisions

- **Split by time, not at random.** Test issues are all newer than everything
  the system learns from, as in real use. Few-shot examples and duplicate
  candidates come only from issues created before the query.
- **Variants are chosen on a dev split, and test is run once.** Issues from
  2024 decided prompt v1 vs. v2 and Flash-Lite vs. 3.8 Flash.
- **The cheap model won on evidence.** Gemini 3.8 Flash scored 0.56 vs. 0.535
  dev macro-F1, at 2.7× the cost per call and with a worse needs-info score.
- **Free baselines first.** A kNN vote over embeddings costs nothing and
  scores 0.41. An LLM that can't beat it isn't worth paying for.
- **Things that were tried and rejected:**
  - **LLM + trained-classifier ensembles.** Variants were chosen on dev:
    - gating LLM labels by the classifier's probability
    - adding labels the classifier is confident about
    - per-label routing to whichever source did better

    They gained +0.015 to +0.025 on dev but **−0.001 and −0.008 on test**
    (CIs span zero). The dev gains didn't hold up, so the pipeline stays
    LLM-only.
  - **Local cross-encoder rerankers for duplicates.** bge-reranker-v2-m3
    and MiniLM rerank a wide pool of up to 100 candidates down to 16. Both
    found fewer originals than the union pool (8/15 vs. 10/15), and took 6–65 s per issue.
    They're trained on web question-answering, not "same root cause". The
    wide pool *contains* 14/15 originals, so the open problem is choosing
    from it, not retrieving it.
  - Keeping LLM topic labels only when neighbours agree: +0.015 on dev, which
    is noise.
  - kNN for topic labels: −0.03 on dev.
  - Stricter needs-info rules: no gain on dev.
  - Clarified definitions *without* spotlighting (v3): 0.54 on dev, but
    `topic: config` got worse (0.47 → 0.37) and a new attack landed.
- **One call does two jobs.** Labels and missing information come from the
  same request.
- **What reaches a reporter is gated by measured precision.** Only outputs
  measured at 0.70 or above go into the reply draft. Everything else is for
  maintainer review.
- **Untrusted input, in layers.**
  - Redaction, then a fence that issue text can't close.
  - Enum-constrained outputs, re-validated in code.
  - Model-written reasons are length-capped and stripped of links and
    @mentions.
  - The reply is a template, so model prose never reaches anyone.
  - Nothing can post to GitHub.
- **Stateless requests (`store: false`).** Each issue is independent, so
  server-side history would buy nothing and would keep strangers' text at
  Google. Explicit caching isn't available on the Interactions API, and
  implicit caching never triggered: the prompts (about 2,000 tokens) are too
  short. That was measured, not assumed.
- **An undocumented API limit, found the hard way.** Integer `enum` values of
  7+ digits make the Interactions API return a misleading 400 ("Invalid JSON
  payload: syntax error"). It was isolated by bisection: 999,999 works and
  9,999,999 fails. Real issue numbers are far below that.
- **Cost is enforced in code.** Every call goes through one function with a
  disk cache, a cost ledger and a hard spend cap. Responses with status
  `incomplete` are never cached. 3.8 Flash's thinking tokens count against
  `max_output_tokens`, which silently truncated answers until handled.
- **Duplicate ground truth is silver and small.** Targets come from maintainer
  comments ("duplicate of #N"): 15 usable pairs out of 43 closed duplicates.
  Scoring is cluster-aware: A flagged as a duplicate of B counts when both
  duplicate C. The 3 pairs no plain search finds were checked by hand. All are
  correct **root-cause** duplicates, where the reporter's symptom shares
  almost no words with the original's description of the mechanism.

## Limitations

- **The test sets are small.** 237 issues for labels (11–44 per label), and
  15 pairs for duplicates. Label gaps under about 0.03 macro-F1 are noise (see
  the bootstrap table). Duplicate results have intervals 30–45 points wide.
- **Maintainer labels are incomplete ground truth.** Maintainers apply about 1
  label per issue, so measured precision understates real precision. The
  audit's correction rests on two AI raters with moderate agreement
  (kappa 0.41) and **no human rater**.
- **The duplicate improvements were measured on the data used to design them.**
  There's no separate duplicate dev set.
- **The attack suite was written by the defender.** It also found 2 real
  weaknesses. A suite written by someone else would be stronger evidence.
- **The backtest is not a true prospective run.** The predictions were made
  after the issues were opened, with the older v2 prompt and the fused
  duplicate pool. They are kept separately in `eval/shadow/backtest.jsonl`.
  The honest version, the daily prospective run below, needs weeks of data.
- **No maintainer has used it yet.** Usefulness is inferred from metrics, not
  observed.

## Shadow mode (live)

A scheduled GitHub Actions job (`.github/workflows/shadow.yml`) runs once a
day:

1. Fetch new pytest issues.
2. Triage every issue opened since **2026-10-14** that hasn't been predicted
   yet. Each prediction is made within about a day of the issue opening,
   usually before any maintainer label.
3. Re-score all predictions against what maintainers have done since.
4. Commit the results: `eval/shadow/predictions.jsonl` (issue numbers and
   enum outputs only) and the regenerated results page.

Because each prediction is committed when it's made, it's timestamped and
can't be edited later. The job has its own $1.00 spend cap and costs about
$0.05 a month. It reads pytest's public issues and never writes to them.

## Run it

```bash
uv sync
uv run triage fetch                 # download / refresh issues (incremental)
uv run triage stats                 # snapshot + split summary
uv run triage eval labels           # free baselines
uv run triage eval labels --llm     # + Gemini (costs about $0.13)
uv run triage eval dupes [--llm]    # duplicate search / verifier
uv run triage eval attacks          # prompt-injection suite (19 attack/control pairs)
uv run triage eval significance     # paired bootstrap of label gaps (free)
uv run triage eval learned --phase dev   # trained classifier + ensembles, chosen on dev (free)
uv run triage eval learned --phase test --variants gate:0.3 route
uv run triage eval rerank           # local rerankers vs. the union candidate pool (free)
uv run triage issue 14476           # triage one issue
uv run triage shadow --since 2026-10-01 && uv run triage shadow-score
uv run triage audit rate --rater YOU # blind human audit of "false positive" labels
uv run triage audit rate-ai         # blind LLM second rater (gemini-3.8-flash)
uv run triage audit score           # adjusted precision + agreement between raters
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
