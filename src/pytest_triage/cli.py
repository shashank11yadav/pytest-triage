"""Command-line entry point: `uv run triage <command>`."""

import argparse
import json
from collections import Counter

from pytest_triage import config, store


def cmd_fetch(args: argparse.Namespace) -> None:
    """Download issues. Incremental (only updated since last run) unless --full."""
    from pytest_triage.github import fetch_issues, make_client

    old = [] if args.full or not config.ISSUES.exists() else store.load_issues()
    since = store.last_updated(old)
    print(f"fetching {'all issues' if not since else f'issues updated since {since}'}...")
    with make_client(config.github_token()) as client:
        new = list(fetch_issues(client, config.REPO, since=since))
    issues = store.merge(old, new)
    store.save_issues(issues)
    print(f"{len(new)} fetched, {len(issues)} in snapshot ({config.ISSUES})")


def cmd_truth(args: argparse.Namespace) -> None:
    """Rebuild eval/duplicates.json from maintainer comments."""
    from pytest_triage.github import make_client
    from pytest_triage.groundtruth import build_truth

    issues = store.load_issues()
    with make_client(config.github_token()) as client:
        truth = build_truth(client, config.REPO, issues)
    config.DUPLICATES.parent.mkdir(parents=True, exist_ok=True)
    config.DUPLICATES.write_text(json.dumps(truth, indent=2) + "\n")
    print(f"{len(truth)} duplicate pairs with a usable target -> {config.DUPLICATES}")


def cmd_stats(args: argparse.Namespace) -> None:
    from pytest_triage.dataset import is_evaluable, split, targets

    issues = store.load_issues()
    memory, test = split([i for i in issues if is_evaluable(i)])
    recent = [i for i in issues if i["created_at"] >= config.SPLIT_DATE]
    unlabelled = sum(not i["labels"] for i in recent)
    print(f"snapshot: {len(issues)} issues, newest {issues[-1]['created_at']}")
    print(f"since {config.SPLIT_DATE}: {unlabelled}/{len(recent)} never labelled")
    print(f"evaluable: memory {len(memory)}, test {len(test)}\n")
    m, t = Counter(), Counter()
    for i in memory:
        m.update(targets(i))
    for i in test:
        t.update(targets(i))
    print(f"{'label':<22}{'memory':>8}{'test':>8}")
    for label in config.LABELS:
        print(f"{label:<22}{m[label]:>8}{t[label]:>8}")


def _label_setup():
    from pytest_triage.dataset import is_evaluable, split
    from pytest_triage.search import IssueIndex

    memory, test = split([i for i in store.load_issues() if is_evaluable(i)])
    return memory, test, IssueIndex.build(memory)


def cmd_eval_labels(args: argparse.Namespace) -> None:
    from pytest_triage import baselines
    from pytest_triage.evaluate import eval_labels

    memory, test, memory_index = _label_setup()
    keywords = baselines.mine_keywords(memory)
    predictors = {
        "everything": baselines.predict_everything,
        "majority": baselines.make_majority(memory),
        "keywords": baselines.make_keyword_predictor(keywords),
        "knn": baselines.make_knn(memory_index),
    }
    if args.llm:
        from pytest_triage.classify import make_llm_classifier

        predictors[f"llm:{args.model}"] = make_llm_classifier(memory_index, model=args.model)
    eval_labels({k: v for k, v in predictors.items() if not args.only or k in args.only}, test)

    if args.llm:
        from pytest_triage.classify import NEEDS_INFO, make_needs_info
        from pytest_triage.evaluate import eval_needs_info

        always = lambda issue: True  # noqa: E731 - baseline: flag everything
        eval_needs_info("always", always, test, NEEDS_INFO)
        eval_needs_info(
            f"llm:{args.model}", make_needs_info(memory_index, args.model), test, NEEDS_INFO
        )


def cmd_eval_dupes(args: argparse.Namespace) -> None:
    from pytest_triage.evaluate import eval_duplicates
    from pytest_triage.search import IssueIndex

    truth = {int(k): v for k, v in json.loads(config.DUPLICATES.read_text()).items()}
    issues = store.load_issues()
    index = IssueIndex.build(issues)
    if not args.llm:
        eval_duplicates(index, truth)
        return
    from pytest_triage.evaluate import eval_dedupe, sample_negatives

    negatives = sample_negatives(issues, truth, args.negatives)
    eval_dedupe(index, truth, negatives, expand=not args.no_expand, model=args.model)


def _live_indexes():
    """(all issues, labelled issues): what live triage searches. Each query still
    only sees issues created before it."""
    from pytest_triage.dataset import is_evaluable
    from pytest_triage.search import IssueIndex

    issues = store.load_issues()
    return (
        issues,
        IssueIndex.build(issues),
        IssueIndex.build([i for i in issues if is_evaluable(i)]),
    )


def cmd_eval_attacks(args: argparse.Namespace) -> None:
    from pytest_triage.attacks import run_attacks
    from pytest_triage.evaluate import _now, save_result

    _, index, labelled = _live_indexes()
    result = run_attacks(index, labelled, args.model)
    save_result("attacks", args.model, result | {"at": _now()})
    print(f"\n{result['landed']}/{result['n']} attacks landed")


def cmd_issue(args: argparse.Namespace) -> None:
    """Triage one issue from the snapshot and print the report."""
    from pytest_triage.triage import triage_issue

    _, index, labelled = _live_indexes()
    issue = index.get(args.number)
    if issue is None:
        raise SystemExit(f"#{args.number} is not in the snapshot - run `triage fetch` first.")
    report = triage_issue(issue, index, labelled, args.model)
    if args.json:
        print(json.dumps(report, indent=2))
        return
    print(f"#{report['number']} {report['title']}")
    print(f"  labels:     {', '.join(report['labels']) or '-'}")
    print(f"  missing:    {', '.join(report['missing']) or '-'}")
    print(f"  duplicates: {', '.join(f'#{n}' for n in report['duplicates']) or '-'}")
    for v in report["duplicate_verdicts"]:
        print(f"     #{v['number']} ({v['confidence']}): {v['reason']}")
    if report["injection_signals"]:
        print(f"  !! injection signals: {report['injection_signals']}")
    print(f"  maintainers actually applied: {', '.join(issue['labels']) or '-'}")
    print(f"  trace: {report['trace']}")
    if report["reply_draft"]:
        print("\n--- draft reply (not posted) ---\n" + report["reply_draft"])


def cmd_shadow(args: argparse.Namespace) -> None:
    from pytest_triage.shadow import run_shadow

    issues, index, labelled = _live_indexes()
    since = args.since or max(i["created_at"] for i in issues)[:10]
    n = run_shadow(issues, index, labelled, since=since, limit=args.limit, model=args.model)
    print(f"{n} new predictions -> {config.SHADOW / 'predictions.jsonl'}")


def cmd_shadow_score(args: argparse.Namespace) -> None:
    from pytest_triage.evaluate import _now, save_result
    from pytest_triage.shadow import load_predictions, score_shadow

    result = score_shadow(load_predictions(), store.load_issues())
    save_result("shadow", "latest", result | {"at": _now()})
    print(json.dumps({k: v for k, v in result.items() if k != "labels"}, indent=2))
    print(f"label macro-F1 on maintainer-triaged: {result['labels']['macro_f1']:.2f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="triage", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download / refresh the issue snapshot")
    p.add_argument("--full", action="store_true", help="re-download everything")
    p.set_defaults(func=cmd_fetch)

    sub.add_parser("truth", help="rebuild duplicate ground truth").set_defaults(func=cmd_truth)
    sub.add_parser("stats", help="snapshot and split summary").set_defaults(func=cmd_stats)

    ev = sub.add_parser("eval", help="run an evaluation").add_subparsers(dest="what", required=True)
    p = ev.add_parser("labels", help="label predictors vs. maintainer labels")
    p.add_argument("--llm", action="store_true", help="include the Gemini classifier (costs money)")
    p.add_argument("--model", default=config.WORKHORSE)
    p.add_argument("--only", nargs="*", help="run only these predictors")
    p.set_defaults(func=cmd_eval_labels)
    p = ev.add_parser("dupes", help="duplicate search hit@5, or the LLM verifier with --llm")
    p.add_argument("--llm", action="store_true", help="evaluate the LLM verifier (costs money)")
    p.add_argument("--no-expand", action="store_true", help="skip LLM query expansion")
    p.add_argument("--negatives", type=int, default=30, help="non-duplicates to test")
    p.add_argument("--model", default=config.WORKHORSE)
    p.set_defaults(func=cmd_eval_dupes)
    p = ev.add_parser("attacks", help="prompt-injection suite against the real model")
    p.add_argument("--model", default=config.WORKHORSE)
    p.set_defaults(func=cmd_eval_attacks)

    p = sub.add_parser("shadow", help="triage new issues and record predictions")
    p.add_argument("--since", help="YYYY-MM-DD (default: newest issue's date)")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--model", default=config.WORKHORSE)
    p.set_defaults(func=cmd_shadow)
    sub.add_parser("shadow-score", help="score predictions vs. maintainers").set_defaults(
        func=cmd_shadow_score
    )

    p = sub.add_parser("issue", help="triage one issue and print the report")
    p.add_argument("number", type=int)
    p.add_argument("--model", default=config.WORKHORSE)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_issue)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
