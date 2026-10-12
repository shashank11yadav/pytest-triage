# pytest-triage

A shadow-mode triage assistant for [pytest-dev/pytest](https://github.com/pytest-dev/pytest)
issues. For each issue it suggests labels, finds likely duplicates and flags
missing information. Every suggestion is **measured against what the
maintainers actually did**. It reads from GitHub and never writes to it.

> Work in progress. The full write-up, results and design decisions are coming.

## Quick start

```bash
uv sync
uv run triage fetch          # download / refresh the issue snapshot
uv run triage stats          # snapshot + train/test split summary
uv run triage eval labels    # label baselines vs. maintainer labels
uv run triage eval dupes     # duplicate search hit@5
uv run pytest                # offline tests, no network, no cost
```

You need a GitHub token: either `GITHUB_TOKEN` in `.env`, or be logged in
with `gh auth login`. LLM commands also need `GEMINI_API_KEY` in `.env`.
