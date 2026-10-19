"""Shadow mode: triage new issues as they arrive, then score the predictions
later against what maintainers actually did. This is the honest test - the
model cannot have seen these issues, and nobody tuned anything on them.

    eval/shadow/predictions.jsonl  committed: issue numbers + enum outputs only
    data/traces.jsonl              gitignored: full reports incl. model reasons
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from pytest_triage.classify import NEEDS_INFO
from pytest_triage.config import LABELS, SHADOW, TRACES
from pytest_triage.dataset import is_evaluable, targets
from pytest_triage.metrics import confusion, label_report, prf
from pytest_triage.search import IssueIndex

PREDICTIONS = SHADOW / "predictions.jsonl"  # prospective, from SHADOW_START
BACKTEST = SHADOW / "backtest.jsonl"  # after-the-fact, 2026-09-01..10-07, prompt v2


def load_predictions(path: Path = PREDICTIONS) -> list[dict]:
    """Predictions from a JSONL file (prospective by default)."""
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f]


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def run_shadow(
    issues: list[dict],
    index: IssueIndex,
    labelled: IssueIndex,
    since: str,
    limit: int,
    model: str | None = None,
    confidence=None,
) -> int:
    """Triage issues created on/after `since` that haven't been predicted yet."""
    from pytest_triage.triage import triage_issue

    done = {p["number"] for p in load_predictions()}
    todo = [i for i in issues if i["created_at"] >= since and i["number"] not in done][:limit]
    from pytest_triage.llm import BadOutput

    recorded = 0
    for issue in todo:
        try:
            report = triage_issue(issue, index, labelled, model, confidence=confidence)
        except BadOutput as e:
            # One malformed answer shouldn't stop the run. The issue isn't recorded,
            # so the next run tries it again. A bad key or the spend cap still stops it.
            print(f"  #{issue['number']}: skipped, bad model output ({e})", flush=True)
            continue
        _append(TRACES, report | {"at": datetime.now(UTC).isoformat(timespec="seconds")})
        _append(
            PREDICTIONS,
            {
                "number": issue["number"],
                "created_at": issue["created_at"],
                # Maintainer labels already present when we predicted. The model never
                # sees them; recorded to show how often we got there first.
                "labels_at_prediction": issue["labels"],
                "predicted_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "models": report["models"],
                "route": report["route"],
                "labels": report["labels"],  # shown (high-precision mode when on)
                "labels_all": report["labels_all"],  # everything the model suggested
                "needs_info": bool(report["missing"]),
                "duplicates": report["duplicates"],
                "latency_s": report["trace"]["latency_s"],
                "cost_usd": report["trace"]["cost_usd"],
            },
        )
        recorded += 1
        print(
            f"  #{issue['number']:<6} {report['labels']} missing={len(report['missing'])} "
            f"dupes={report['duplicates']} ${report['trace']['cost_usd']:.4f}",
            flush=True,
        )
    return recorded


def score_shadow(predictions: list[dict], issues: list[dict]) -> dict:
    """Compare predictions with the CURRENT maintainer state of each issue."""
    by_number = {i["number"]: i for i in issues}
    rows = [(p, by_number[p["number"]]) for p in predictions if p["number"] in by_number]
    triaged = [(p, i) for p, i in rows if is_evaluable(i)]

    labels = label_report(
        confusion((targets(i), set(p["labels"]) & set(LABELS)) for p, i in triaged)
    )
    labels_all = label_report(
        confusion(
            (targets(i), set(p.get("labels_all", p["labels"])) & set(LABELS)) for p, i in triaged
        )
    )
    shown = sum(bool(p["labels"]) for p, _ in rows)
    shown_triaged = sum(bool(p["labels"]) for p, _ in triaged)
    tp = sum(p["needs_info"] and NEEDS_INFO in i["labels"] for p, i in triaged)
    fp = sum(p["needs_info"] and NEEDS_INFO not in i["labels"] for p, i in triaged)
    fn = sum(not p["needs_info"] and NEEDS_INFO in i["labels"] for p, i in triaged)

    closed_dupe = [i["number"] for _, i in rows if i["state_reason"] == "duplicate"]
    flagged = [p["number"] for p, _ in rows if p["duplicates"]]
    latencies = sorted(p["latency_s"] for p, _ in rows)
    first = sum(1 for p, _ in rows if not p.get("labels_at_prediction", []))
    return {
        "predicted_before_any_maintainer_label": first,
        "predicted": len(rows),
        "routed_to_human": sum(p.get("route") == "human" for p, _ in rows),
        "maintainer_triaged": len(triaged),
        "still_untriaged": len(rows) - len(triaged),
        "labels": labels,
        "labels_all": labels_all,
        "issues_with_a_shown_label": shown,
        "triaged_with_a_shown_label": shown_triaged,
        "needs_info": dict(zip(("precision", "recall", "f1"), prf(tp, fp, fn), strict=True)),
        "closed_as_duplicate": closed_dupe,
        "flagged_duplicate": flagged,
        "flagged_and_closed_duplicate": sorted(set(closed_dupe) & set(flagged)),
        "cost_usd_total": round(sum(p["cost_usd"] for p, _ in rows), 4),
        "latency_p50_s": latencies[len(latencies) // 2] if latencies else None,
        "latency_p95_s": latencies[int(len(latencies) * 0.95)] if latencies else None,
    }
