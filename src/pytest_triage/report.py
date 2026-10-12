"""Generate the results page (docs/index.html) and README tables from eval/results.

Every number shown anywhere comes from these JSON files - nothing is typed by hand.
The page holds issue numbers and metrics only, never issue text.
"""

import html
import json
from pathlib import Path

from pytest_triage.config import LABELS, RESULTS, SHADOW

DOCS = Path("docs")
ISSUE_URL = "https://github.com/pytest-dev/pytest/issues/"


def _load(name: str) -> dict:
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _pct(x: float) -> str:
    return f"{x:.0%}"


def label_rows(labels: dict) -> list[tuple[str, float, float, int]]:
    """(predictor, macro-F1, micro-F1, n), best macro-F1 first."""
    rows = [(k, v["macro_f1"], v["micro_f1"], v["n"]) for k, v in labels.items()]
    return sorted(rows, key=lambda r: r[1], reverse=True)


def markdown_tables() -> str:
    labels, needs, dupes = _load("labels"), _load("needs_info"), _load("duplicates")
    dedupe, attacks = _load("dedupe"), _load("attacks")
    out = ["### Labels (10 labels, test set)", "", "| Predictor | macro-F1 | micro-F1 | n |"]
    out.append("|---|---|---|---|")
    out += [f"| {k} | {m:.2f} | {mi:.2f} | {n} |" for k, m, mi, n in label_rows(labels)]

    best = next((k for k, *_ in label_rows(labels) if k.startswith("llm")), None)
    if best:
        out += ["", f"Per label, `{best}` vs. `knn`:", "", "| Label | LLM F1 | kNN F1 | support |"]
        out.append("|---|---|---|---|")
        for label in LABELS:
            a, b = labels[best]["labels"][label], labels.get("knn", {}).get("labels", {})
            knn = b.get(label, {}).get("f1", 0.0)
            out.append(f"| {label} | {a['f1']:.2f} | {knn:.2f} | {a['support']} |")

    if needs:
        out += ["", "### Needs more information (binary)", "", "| Predictor | P | R | F1 |"]
        out.append("|---|---|---|---|")
        for k, v in needs.items():
            out.append(f"| {k} | {v['precision']:.2f} | {v['recall']:.2f} | {v['f1']:.2f} |")

    if dupes or dedupe:
        out += ["", "### Duplicates", "", "| Method | Result | 95% CI |", "|---|---|---|"]
        for k, v in dupes.items():
            lo, hi = v["ci95"]
            out.append(f"| search: {k} | hit@5 {v['hits']}/{v['n']} | {_pct(lo)}-{_pct(hi)} |")
        for k, v in dedupe.items():
            lo, hi = v["recall_ci95"]
            out.append(
                f"| LLM verifier ({k}) | flagged {v['correct']}/{v['n_duplicates']} correctly, "
                f"{v['false_alarms']}/{v['n_non_duplicates']} false alarms"
                f" | {_pct(lo)}-{_pct(hi)} |"
            )
    if attacks:
        out += ["", "### Prompt injection", ""]
        for k, v in attacks.items():
            out.append(f"- `{k}`: {v['landed']}/{v['n']} attacks landed")
            out += [
                f"  - {'LANDED' if c['landed'] else 'blocked'}: {c['name']}" for c in v["cases"]
            ]
    return "\n".join(out)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def html_page() -> str:
    labels, needs, dupes = _load("labels"), _load("needs_info"), _load("duplicates")
    dedupe, attacks, shadow = _load("dedupe"), _load("attacks"), _load("shadow")
    sections = []

    rows = [[html.escape(k), f"{m:.2f}", f"{mi:.2f}", str(n)] for k, m, mi, n in label_rows(labels)]
    sections.append(
        "<h2>Labels</h2><p>10 labels, scored on issues created after 2025-01-01 that "
        "maintainers triaged. Anything that costs money must beat the free kNN vote.</p>"
        + _table(["Predictor", "macro-F1", "micro-F1", "n"], rows)
    )
    if needs:
        rows = [
            [html.escape(k), f"{v['precision']:.2f}", f"{v['recall']:.2f}", f"{v['f1']:.2f}"]
            for k, v in needs.items()
        ]
        sections.append(
            "<h2>Needs more information</h2>" + _table(["Predictor", "P", "R", "F1"], rows)
        )
    if dupes or dedupe:
        rows = [
            [
                f"search: {k}",
                f"hit@5 {v['hits']}/{v['n']}",
                "-",
                f"{_pct(v['ci95'][0])}-{_pct(v['ci95'][1])}",
            ]
            for k, v in dupes.items()
        ]
        rows += [
            [
                html.escape(f"LLM verifier: {k}"),
                f"{v['correct']}/{v['n_duplicates']}",
                f"{v['false_alarms']}/{v['n_non_duplicates']}",
                f"{_pct(v['recall_ci95'][0])}-{_pct(v['recall_ci95'][1])}",
            ]
            for k, v in dedupe.items()
        ]
        sections.append(
            "<h2>Duplicates</h2><p>15 known duplicate pairs (silver labels from maintainer "
            "comments) plus non-duplicates to measure false alarms.</p>"
            + _table(["Method", "Found", "False alarms", "95% CI"], rows)
        )
    if attacks:
        rows = [
            [
                html.escape(c["name"]),
                "LANDED" if c["landed"] else "blocked",
                "yes" if c["flagged"] else "no",
            ]
            for v in attacks.values()
            for c in v["cases"]
        ]
        sections.append("<h2>Prompt injection</h2>" + _table(["Attack", "Result", "Flagged"], rows))
    if shadow:
        s = shadow["latest"]
        sections.append(
            "<h2>Shadow mode</h2>"
            f"<p>{s['predicted']} new issues triaged as they arrived; maintainers have since "
            f"triaged {s['maintainer_triaged']}, {s['still_untriaged']} remain untriaged. "
            f"Label macro-F1 on those triaged: {s['labels']['macro_f1']:.2f}. "
            f"Median latency {s['latency_p50_s']}s; total cost ${s['cost_usd_total']}.</p>"
        )
        preds = []
        path = SHADOW / "predictions.jsonl"
        if path.exists():
            preds = [json.loads(line) for line in path.read_text().splitlines()][-30:]
        rows = [
            [
                f'<a href="{ISSUE_URL}{p["number"]}">#{p["number"]}</a>',
                html.escape(", ".join(p["labels"]) or "-"),
                "yes" if p["needs_info"] else "",
                ", ".join(f"#{n}" for n in p["duplicates"]),
            ]
            for p in reversed(preds)
        ]
        sections.append(_table(["Issue", "Predicted labels", "Needs info", "Duplicate of"], rows))

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>pytest-triage results</title>
<style>
:root {{ --bg:#fff; --fg:#1d1d1f; --muted:#6e6e73; --line:#e5e5ea; --accent:#0a7d55; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#111214; --fg:#ececf0; --muted:#9a9aa2; --line:#2a2b30; --accent:#4cc79a; }}
}}
body {{ background:var(--bg); color:var(--fg); margin:0;
  font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:860px; margin:0 auto; padding:32px 16px 64px; }}
h1 {{ margin:0 0 4px; font-size:28px; }} h2 {{ margin:36px 0 8px; font-size:19px; }}
p {{ color:var(--muted); }} a {{ color:var(--accent); }}
table {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }}
th,td {{ text-align:left; padding:7px 10px; border-bottom:1px solid var(--line); }}
th {{ font-weight:600; color:var(--muted); font-size:13px; }}
.wrap {{ overflow-x:auto; }}
</style></head><body><main>
<h1>pytest-triage</h1>
<p>A shadow-mode triage assistant for
<a href="https://github.com/pytest-dev/pytest">pytest-dev/pytest</a> issues, measured
against what maintainers actually did. It never posts to GitHub.</p>
{"".join(f'<div class="wrap">{s}</div>' for s in sections)}
</main></body></html>
"""


def write_report() -> Path:
    DOCS.mkdir(exist_ok=True)
    path = DOCS / "index.html"
    path.write_text(html_page())
    return path
