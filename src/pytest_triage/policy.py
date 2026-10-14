"""What the pipeline is allowed to hand to a maintainer. Runs after the model.

ROUTE     an issue whose own text trips the injection detector gets no
          suggestions at all, only "needs a human". The detector fires on 6 of
          6,416 real pytest issues, so this costs almost nothing, and a
          suggestion on a possibly-steered issue is worth less than none.
QUARANTINE a candidate whose text trips the detector can't be reported as a
          duplicate: older issues are stranger-written too, and a planted one
          is how an attacker would reach the verifier.
CONTRACT  invariants every report must satisfy (enums, duplicates older than
          the issue, nothing suggested when routed). A broken invariant is a
          bug, so it raises. Soft oddities are recorded as warnings, not fixed:
          e.g. two `type:` labels look wrong, but maintainers apply two on 4%
          of issues and the model does it on 1 of 256 dev issues, so a rewrite
          rule would remove as many right answers as wrong ones.
"""

from pytest_triage.classify import MISSING
from pytest_triage.config import LABELS


class ContractViolation(Exception):
    """A report broke an invariant. Never shown to a maintainer."""


def apply_policy(report: dict) -> dict:
    """Route or quarantine in place of the raw outputs. Returns a new report; the
    model's own outputs stay under `raw` for traces and evals, never for display."""
    out = dict(report)
    out["raw"] = {k: report[k] for k in ("labels", "missing", "duplicates")}
    out["raw"]["labels"] = report.get("labels_all", report["labels"])
    quarantined = sorted(set(report["duplicates"]) & set(report["candidate_injection_signals"]))
    out["duplicates"] = [n for n in report["duplicates"] if n not in quarantined]
    out["quarantined_candidates"] = quarantined
    out["route"] = "human" if report["injection_signals"] else "suggest"
    if out["route"] == "human":
        out |= {"labels": [], "missing": [], "duplicates": []}
        if "labels_all" in out:
            out["labels_all"] = []
    return out


def check_report(report: dict, issue: dict, created_at_of: dict[int, str]) -> list[str]:
    """Raise ContractViolation on a broken invariant; return soft warnings."""
    problems = []
    if not set(report["labels"]) <= set(LABELS):
        problems.append(f"unknown labels {set(report['labels']) - set(LABELS)}")
    if not set(report["missing"]) <= set(MISSING):
        problems.append("unknown missing-info item")
    for n in report["duplicates"]:
        if n == issue["number"]:
            problems.append("issue flagged as its own duplicate")
        elif n not in report["candidates"]:
            problems.append(f"duplicate #{n} was never a candidate")
        elif created_at_of.get(n, "") >= issue["created_at"]:
            problems.append(f"duplicate #{n} is not older than the issue")
    if report.get("route") == "human" and (
        report["labels"] or report["missing"] or report["duplicates"] or report["reply_draft"]
    ):
        problems.append("routed to a human but still suggests something")
    if problems:
        raise ContractViolation("; ".join(problems))

    warnings = []
    if sum(label.startswith("type:") for label in report["labels"]) > 1:
        warnings.append("more than one type: label")
    if report["missing"] and "type: bug" not in report["labels"]:
        warnings.append("missing info requested on a non-bug")
    return warnings
