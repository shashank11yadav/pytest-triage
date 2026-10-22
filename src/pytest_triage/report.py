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
PAGE_URL = "https://shashank11yadav.github.io/pytest-triage/"
# Shown by search results and link previews (Slack, iMessage, LinkedIn, X).
PAGE_DESCRIPTION = (
    "A shadow-mode triage assistant for pytest issues: suggests labels, missing information "
    "and likely duplicates, measured against what maintainers actually did."
)


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
    out = ["### Labels (11 labels, test set)", "", "| Predictor | macro-F1 | micro-F1 | n |"]
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
            if "rates" in v:  # mutation runs: too many rows to list, rates only
                continue
            out += [
                f"  - {'LANDED' if c['landed'] else 'blocked'}: {c['name']}" for c in v["cases"]
            ]
    slices = _load("slices").get("test")
    if slices:
        names = [k for k in next(iter(slices.values())) if k != "n"]
        out += ["", "### Micro-F1 by issue shape (test)", ""]
        out.append("| Slice | n | " + " | ".join(names) + " |")
        out.append("|---|---|" + "---|" * len(names))
        for name, row in slices.items():
            if name == "at":
                continue
            cells = " | ".join(
                f"{row[k]['micro_f1']:.2f} ({row[k]['ci95'][0]:.2f}-{row[k]['ci95'][1]:.2f})"
                for k in names
            )
            out.append(f"| {name} | {row['n']} | {cells} |")
    return "\n".join(out)


def selective_rows(selective: dict) -> list[list[str]]:
    """Operating points chosen on dev, as reported on dev and test."""
    rows = []
    for phase in ("dev", "test"):
        for name, p in selective.get(phase, {}).get("chosen_on_dev", {}).items():
            if p:
                rows.append(
                    [
                        phase,
                        name,
                        f"{p['threshold']:.2f}",
                        f"{p['precision']:.2f} ({p['precision_ci95'][0]:.2f}-"
                        f"{p['precision_ci95'][1]:.2f})",
                        f"{p['recall']:.2f}",
                        _pct(p["issue_coverage"]),
                    ]
                )
    return rows


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
        "<h2>Labels</h2><p>11 labels, scored on issues created after 2025-01-01 that "
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
    selective = _load("selective")
    if selective:
        sections.append(
            "<h2>High-precision mode</h2><p>Show an LLM label only when the trained "
            "classifier also gives it high probability. The threshold was chosen on 2024 "
            "issues (lowest one reaching 70% precision) and then applied once to the test "
            "set.</p>"
            + _table(
                ["Split", "Confidence", "Threshold", "Precision (95% CI)", "Recall", "Issues"],
                [[html.escape(c) for c in r] for r in selective_rows(selective)],
            )
        )
    slices = _load("slices").get("test")
    if slices:
        names = [k for k in next(iter(slices.values())) if k != "n"]
        rows = [
            [html.escape(name), str(row["n"])]
            + [
                f"{row[k]['micro_f1']:.2f} ({row[k]['ci95'][0]:.2f}-{row[k]['ci95'][1]:.2f})"
                for k in names
            ]
            for name, row in slices.items()
            if name != "at"
        ]
        sections.append(
            "<h2>Where it fails</h2><p>Micro-F1 on the test set by issue shape, with "
            "bootstrap 95% intervals. Slices overlap.</p>" + _table(["Slice", "n", *names], rows)
        )
    if attacks:
        rows = []
        for key, v in attacks.items():
            if "rates" in v:
                r = v["rates"]
                rows.append(
                    [
                        html.escape(key),
                        f"{r['landed']['count']}/{r['landed']['n']}",
                        f"{r['fooled_model']['count']}/{r['fooled_model']['n']}",
                        f"{r['flagged']['count']}/{r['flagged']['n']}",
                    ]
                )
        if rows:
            sections.append(
                "<h2>Prompt injection</h2><p>Each attack in 6 variants (verbatim, moved, "
                "quoted, in a code block, zero-width spaces, look-alike letters). "
                "<em>Landed</em> means a maintainer would see the attacker's outcome; "
                "<em>fooled the model</em> counts the model's raw output, before the "
                "policy layer routes flagged issues to a human.</p>"
                + _table(["Run", "Landed", "Fooled the model", "Flagged"], rows)
            )
    if shadow:
        sections.append(_shadow_section(shadow))

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>pytest-triage results</title>
<meta name="description" content="{PAGE_DESCRIPTION}">
<!-- Link previews (Slack, iMessage, LinkedIn, X). The image needs an absolute URL. -->
<meta property="og:type" content="website">
<meta property="og:site_name" content="Shashank Yadav">
<meta property="og:title" content="pytest-triage: triage suggestions for pytest issues">
<meta property="og:description" content="{PAGE_DESCRIPTION}">
<meta property="og:url" content="{PAGE_URL}">
<meta property="og:image" content="{PAGE_URL}og.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="pytest-triage: triage suggestions for pytest issues">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:creator" content="@Shashankyadav30">
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
footer {{ margin-top:40px; border-top:1px solid var(--line); font-size:13px; }}
</style></head><body><main>
<h1>pytest-triage</h1>
<p>A shadow-mode triage assistant for
<a href="https://github.com/pytest-dev/pytest">pytest-dev/pytest</a> issues, measured
against what maintainers actually did. It never posts to GitHub.</p>
{"".join(f'<div class="wrap">{s}</div>' for s in sections)}
<footer><p>By Shashank Yadav &middot;
<a href="https://www.shashankyadav.dev/work/pytest-triage">shashankyadav.dev</a></p></footer>
</main></body></html>
"""


def _shadow_section(shadow: dict) -> str:
    from pytest_triage.config import SHADOW_START

    out = ["<h2>Shadow mode (prospective)</h2>"]
    s = shadow.get("prospective")
    if s:
        out.append(
            f"<p>Since {SHADOW_START}, a daily job triages each new issue within about a day "
            f"of it opening. {s['predicted']} predicted so far; maintainers have since triaged "
            f"{s['maintainer_triaged']}, {s['still_untriaged']} remain untriaged. Label "
            f"macro-F1 on those triaged: {s['labels']['macro_f1']:.2f} (small n while the "
            f"run is young). Median latency {s['latency_p50_s']}s; total cost "
            f"${s['cost_usd_total']}.</p>"
        )
    else:
        out.append(f"<p>Running daily from {SHADOW_START}; no predictions scored yet.</p>")
    path = SHADOW / "predictions.jsonl"
    preds = (
        [json.loads(line) for line in path.read_text().splitlines()][-30:] if path.exists() else []
    )
    if preds:
        rows = [
            [
                f'<a href="{ISSUE_URL}{p["number"]}">#{p["number"]}</a>',
                html.escape(", ".join(p["labels"]) or "-"),
                "yes" if p["needs_info"] else "",
                ", ".join(f"#{n}" for n in p["duplicates"]),
            ]
            for p in reversed(preds)
        ]
        out.append(
            _table(["Issue", "Suggested labels", "Needs info?", "Possible duplicate of"], rows)
        )
    b = shadow.get("backtest")
    if b:
        out.append(
            f"<p>Earlier backtest (predicted after the fact, older pipeline): {b['predicted']} "
            f"issues from 2026-09-01 to 2026-10-07, of which {b['still_untriaged']} were still "
            f"untriaged by maintainers when scored.</p>"
        )
    return "".join(out)


def write_report() -> Path:
    DOCS.mkdir(exist_ok=True)
    path = DOCS / "index.html"
    path.write_text(html_page())
    return path
