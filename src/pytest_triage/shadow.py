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
from pytest_triage.config import LABELS, SHADOW, TRACES, WORKHORSE
from pytest_triage.dataset import is_evaluable, targets
from pytest_triage.metrics import confusion, label_report, prf
from pytest_triage.search import IssueIndex

PREDICTIONS = SHADOW / "predictions.jsonl"


def load_predictions(path: Path = PREDICTIONS) -> list[dict]:
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
    model: str = WORKHORSE,
) -> int:
    """Triage issues created on/after `since` that haven't been predicted yet."""
    from pytest_triage.triage import triage_issue

    done = {p["number"] for p in load_predictions()}
    todo = [i for i in issues if i["created_at"] >= since and i["number"] not in done][:limit]
    for issue in todo:
        report = triage_issue(issue, index, labelled, model)
        _append(TRACES, report | {"at": datetime.now(UTC).isoformat(timespec="seconds")})
        _append(
            PREDICTIONS,
            {
                "number": issue["number"],
                "created_at": issue["created_at"],
                "predicted_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "model": model,
                "labels": report["labels"],
                "needs_info": bool(report["missing"]),
                "duplicates": report["duplicates"],
                "latency_s": report["trace"]["latency_s"],
                "cost_usd": report["trace"]["cost_usd"],
            },
        )
        print(
            f"  #{issue['number']:<6} {report['labels']} missing={len(report['missing'])} "
            f"dupes={report['duplicates']} ${report['trace']['cost_usd']:.4f}",
            flush=True,
        )
    return len(todo)


def score_shadow(predictions: list[dict], issues: list[dict]) -> dict:
    """Compare predictions with the CURRENT maintainer state of each issue."""
    by_number = {i["number"]: i for i in issues}
    rows = [(p, by_number[p["number"]]) for p in predictions if p["number"] in by_number]
    triaged = [(p, i) for p, i in rows if is_evaluable(i)]

    labels = label_report(
        confusion((targets(i), set(p["labels"]) & set(LABELS)) for p, i in triaged)
    )
    tp = sum(p["needs_info"] and NEEDS_INFO in i["labels"] for p, i in triaged)
    fp = sum(p["needs_info"] and NEEDS_INFO not in i["labels"] for p, i in triaged)
    fn = sum(not p["needs_info"] and NEEDS_INFO in i["labels"] for p, i in triaged)

    closed_dupe = [i["number"] for _, i in rows if i["state_reason"] == "duplicate"]
    flagged = [p["number"] for p, _ in rows if p["duplicates"]]
    latencies = sorted(p["latency_s"] for p, _ in rows)
    return {
        "predicted": len(rows),
        "maintainer_triaged": len(triaged),
        "still_untriaged": len(rows) - len(triaged),
        "labels": labels,
        "needs_info": dict(zip(("precision", "recall", "f1"), prf(tp, fp, fn), strict=True)),
        "closed_as_duplicate": closed_dupe,
        "flagged_duplicate": flagged,
        "flagged_and_closed_duplicate": sorted(set(closed_dupe) & set(flagged)),
        "cost_usd_total": round(sum(p["cost_usd"] for p, _ in rows), 4),
        "latency_p50_s": latencies[len(latencies) // 2] if latencies else None,
        "latency_p95_s": latencies[int(len(latencies) * 0.95)] if latencies else None,
    }
