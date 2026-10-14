"""Command-line entry point: `uv run triage <command>`."""

import argparse
import json
from collections import Counter

import numpy as np

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


def _label_setup(which: str):
    """(memory, scored issues, index of memory) for the test split, or for the dev
    split (2024 issues scored, few-shot memory = issues before 2024)."""
    from pytest_triage.dataset import is_evaluable, split
    from pytest_triage.search import IssueIndex

    memory, test = split([i for i in store.load_issues() if is_evaluable(i)])
    if which == "dev":
        memory, test = split(memory, date=config.DEV_DATE)
    return memory, test, IssueIndex.build(memory)


def cmd_eval_labels(args: argparse.Namespace) -> None:
    from pytest_triage import baselines
    from pytest_triage.evaluate import eval_labels

    memory, scored, memory_index = _label_setup(args.split)
    suffix = "" if args.split == "test" else "_dev"
    predictors = {
        "everything": baselines.predict_everything,
        "majority": baselines.make_majority(memory),
        "keywords": baselines.make_keyword_predictor(baselines.mine_keywords(memory)),
        "knn": baselines.make_knn(memory_index),
    }
    name = f"llm:{args.model}:{args.prompt}"
    if args.llm:
        from pytest_triage.classify import make_llm_classifier

        predictors[name] = make_llm_classifier(memory_index, args.model, args.prompt)
    chosen = {k: v for k, v in predictors.items() if not args.only or k in args.only}
    eval_labels(chosen, scored, file=f"labels{suffix}")

    if args.llm:
        from pytest_triage.classify import NEEDS_INFO, make_needs_info
        from pytest_triage.evaluate import eval_needs_info

        file = f"needs_info{suffix}"
        eval_needs_info("always", lambda issue: True, scored, NEEDS_INFO, file)
        predict = make_needs_info(memory_index, args.model, args.prompt)
        eval_needs_info(name, predict, scored, NEEDS_INFO, file)


def cmd_eval_significance(args: argparse.Namespace) -> None:
    """Paired bootstrap on the test set, from cached predictions (free)."""
    from pytest_triage import baselines, learned
    from pytest_triage.classify import make_llm_classifier
    from pytest_triage.dataset import is_evaluable, split, targets
    from pytest_triage.evaluate import _now, save_result
    from pytest_triage.metrics import bootstrap_diff
    from pytest_triage.search import load_vectors

    _, test, memory_index = _label_setup("test")
    # The trained classifier, exactly as in `eval learned --phase test`.
    evaluable = [i for i in store.load_issues() if is_evaluable(i)]
    vectors = dict(zip((i["number"] for i in evaluable), load_vectors(evaluable), strict=True))
    memory, _ = split(evaluable)
    fit, tune = split(memory, date=config.DEV_DATE)
    vec = lambda issues: np.array([vectors[i["number"]] for i in issues])  # noqa: E731
    model = learned.train(fit, vec(fit), tune, vec(tune))
    predictors = {
        "llm v4": make_llm_classifier(memory_index, prompt="v4"),
        "llm v3s": make_llm_classifier(memory_index, prompt="v3s"),
        "llm v2": make_llm_classifier(memory_index, prompt="v2"),
        "llm v1": make_llm_classifier(memory_index, prompt="v1"),
        "logreg": learned.as_predictor(model, vectors),
        "knn": baselines.make_knn(memory_index),
    }
    preds = {k: [f(i) for i in test] for k, f in predictors.items()}
    truth = [targets(i) for i in test]
    for a, b in (
        ("llm v4", "logreg"),
        ("llm v4", "llm v3s"),
        ("llm v3s", "logreg"),
        ("llm v3s", "llm v2"),
        ("llm v2", "logreg"),
        ("llm v2", "llm v1"),
        ("logreg", "knn"),
        ("llm v1", "knn"),
    ):
        result = bootstrap_diff(truth, preds[a], preds[b])
        save_result("significance", f"{a} vs {b}", result | {"n": len(test), "at": _now()})
        lo, hi = result["ci95"]
        print(f"{a} - {b}: {result['diff']:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")


def cmd_eval_learned(args: argparse.Namespace) -> None:
    """Trained classifier + LLM ensembles. --phase dev chooses; --phase test runs once."""
    from pytest_triage import learned
    from pytest_triage.classify import make_llm_classifier
    from pytest_triage.dataset import is_evaluable, split, targets
    from pytest_triage.evaluate import eval_labels
    from pytest_triage.metrics import confusion, label_report
    from pytest_triage.search import IssueIndex, load_vectors

    evaluable = [i for i in store.load_issues() if is_evaluable(i)]
    vectors = dict(zip((i["number"] for i in evaluable), load_vectors(evaluable), strict=True))
    vec = lambda issues: np.array([vectors[i["number"]] for i in issues])  # noqa: E731
    memory, test = split(evaluable)
    train_, dev = split(memory, date=config.DEV_DATE)  # < 2024, 2024
    if args.phase == "dev":
        fit, tune = split(train_, date="2023-01-01")  # < 2023, 2023
        scored, llm_memory, suffix = dev, train_, "_dev"
    else:
        fit, tune = train_, dev  # < 2024, 2024
        scored, llm_memory, suffix = test, memory, ""

    model = learned.train(fit, vec(fit), tune, vec(tune))
    thresholds = dict(zip(config.LABELS, model.thresholds.tolist(), strict=True))
    print(f"logreg: C={model.C}, thresholds {thresholds}")
    llm = make_llm_classifier(IssueIndex.build(llm_memory), prompt=config.LABEL_PROMPT)
    probs = learned.proba_lookup(model, vectors)
    logreg = learned.as_predictor(model, vectors)

    def ens(variant, route=None):
        return lambda i: learned.combine(llm(i), probs(i), variant, route, logreg(i))

    if args.phase == "dev":
        predictors = {"logreg": logreg}
        for v in (
            "gate:0.1",
            "gate:0.2",
            "gate:0.3",
            "boost:0.8",
            "boost:0.9",
            "gate:0.2+boost:0.9",
        ):
            predictors[f"ensemble {v}"] = ens(v)
        results = eval_labels(predictors, scored, file=f"labels{suffix}")
        # Routing: per label, whichever of llm / logreg had the higher F1 on dev.
        llm_dev = label_report(confusion((targets(i), llm(i)) for i in scored))["labels"]
        route = {
            lbl: "llm" if llm_dev[lbl]["f1"] >= results["logreg"]["labels"][lbl]["f1"] else "logreg"
            for lbl in config.LABELS
        }
        print("route:", route)
        eval_labels({"ensemble route": ens("route", route)}, scored, file=f"labels{suffix}")
        (config.RESULTS / "route.json").write_text(json.dumps(route, indent=2) + "\n")
    else:
        route = json.loads((config.RESULTS / "route.json").read_text())
        predictors = {"logreg": logreg}
        for v in args.variants:
            predictors[f"ensemble {v}"] = ens(v, route if v == "route" else None)
        eval_labels(predictors, scored, file=f"labels{suffix}")

        from pytest_triage import baselines
        from pytest_triage.evaluate import _now, save_result
        from pytest_triage.metrics import bootstrap_diff

        truth = [targets(i) for i in scored]
        llm_p = [llm(i) for i in scored]
        knn = baselines.make_knn(IssueIndex.build(memory))
        current = f"llm {config.LABEL_PROMPT}"
        pairs = {
            "logreg vs knn": ([logreg(i) for i in scored], [knn(i) for i in scored]),
            f"{current} vs logreg": (llm_p, [logreg(i) for i in scored]),
        }
        for name, predict in predictors.items():
            if name.startswith("ensemble"):
                pairs[f"{name} vs {current}"] = ([predict(i) for i in scored], llm_p)
        for name, (a, b) in pairs.items():
            r = bootstrap_diff(truth, a, b)
            save_result("significance", name, r | {"n": len(scored), "at": _now()})
            print(f"{name}: {r['diff']:+.3f}  95% CI [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]")


def cmd_eval_regress(args: argparse.Namespace) -> None:
    """Re-run the classifier on dev (cached: free unless a prompt changed) and
    compare with the committed baseline. Exits 1 on a drop; --update re-baselines."""
    from pytest_triage import regression
    from pytest_triage.classify import assess
    from pytest_triage.llm import BadOutput

    _, dev, memory_index = _label_setup("dev")
    rows = []
    for issue in dev:
        try:
            assessment = assess(issue, memory_index, config.LABEL_MODEL, prompt=config.LABEL_PROMPT)
        except BadOutput:
            assessment = {"labels": set(), "missing": []}
        rows.append(regression.make_row(issue, assessment))

    old_rows, baseline = regression.load_rows(), regression.load_baseline()
    current = regression.score(rows)
    print(f"dev (n={current['n']}): {current}")
    if baseline:
        print(f"baseline:          {baseline['metrics']}")
        print(f"predictions changed on {regression.changed_rows(old_rows, rows)} issues")
    if args.update:
        regression.save(rows)
        print(f"re-baselined -> {regression.BASELINE}")
        return
    drops = regression.compare(current, baseline["metrics"]) if baseline else []
    if drops:
        raise SystemExit("REGRESSION: " + "; ".join(drops))
    print("no regression")


def _trained(phase: str):
    """(logreg, vectors by number, scored issues, LLM memory index) with the same
    protocol as `eval learned`: dev fits < 2023 and tunes on 2023, test fits < 2024
    and tunes on 2024. Nothing here looks at the test split's labels."""
    from pytest_triage import learned
    from pytest_triage.dataset import is_evaluable, split
    from pytest_triage.search import IssueIndex, load_vectors

    evaluable = [i for i in store.load_issues() if is_evaluable(i)]
    vectors = dict(zip((i["number"] for i in evaluable), load_vectors(evaluable), strict=True))
    vec = lambda issues: np.array([vectors[i["number"]] for i in issues])  # noqa: E731
    memory, test = split(evaluable)
    train_, dev = split(memory, date=config.DEV_DATE)
    if phase == "dev":
        fit, tune = split(train_, date="2023-01-01")
        scored, llm_memory = dev, train_
    else:
        fit, tune, scored, llm_memory = train_, dev, test, memory
    model = learned.train(fit, vec(fit), tune, vec(tune))
    return model, vectors, scored, IssueIndex.build(llm_memory)


def cmd_eval_slices(args: argparse.Namespace) -> None:
    """Scores by issue shape on the test split, from cached predictions (free).
    Descriptive only: no choice is made from these numbers."""
    from pytest_triage import baselines, learned
    from pytest_triage.classify import make_llm_classifier
    from pytest_triage.evaluate import _now, save_result
    from pytest_triage.slices import slice_report

    model, vectors, scored, memory_index = _trained("test")
    predictors = {
        f"llm {config.LABEL_PROMPT}": make_llm_classifier(
            memory_index, config.LABEL_MODEL, config.LABEL_PROMPT
        ),
        "logreg": learned.as_predictor(model, vectors),
        "knn": baselines.make_knn(memory_index),
    }
    preds = {k: [f(i) for i in scored] for k, f in predictors.items()}
    report = slice_report(scored, preds)
    save_result("slices", "test", report | {"at": _now()})
    print(f"{'slice':<34}{'n':>5}" + "".join(f"{k:>22}" for k in predictors))
    for name, row in report.items():
        cells = "".join(
            f"{row[k]['micro_f1']:>8.2f} [{row[k]['ci95'][0]:.2f}-{row[k]['ci95'][1]:.2f}]"
            for k in predictors
        )
        print(f"{name:<34}{row['n']:>5}{cells}")


def cmd_eval_selective(args: argparse.Namespace) -> None:
    """Precision vs. coverage, with the trained classifier's probability as the
    confidence for each LLM label. The threshold is chosen on dev (the lowest one
    reaching REPLY_MIN_PRECISION) and then reported once on test."""
    from pytest_triage import learned
    from pytest_triage.classify import make_llm_classifier
    from pytest_triage.dataset import targets
    from pytest_triage.evaluate import _now, save_result
    from pytest_triage.metrics import operating_point, selective_curve

    grid = [round(x, 2) for x in np.arange(0.0, 0.951, 0.05)]
    bar = config.REPLY_MIN_PRECISION
    chosen = {}
    for phase in ("dev", "test"):
        model, vectors, scored, memory_index = _trained(phase)
        llm = make_llm_classifier(memory_index, config.LABEL_MODEL, config.LABEL_PROMPT)
        probs = learned.proba_lookup(model, vectors)
        truth = [targets(i) for i in scored]
        conf = [probs(i) for i in scored]
        llm_labels = [llm(i) for i in scored]
        logreg_labels = [set(config.LABELS)] * len(scored)  # logreg alone: every label, by p
        result = {
            "n": len(scored),
            "llm gated by logreg p": selective_curve(truth, llm_labels, conf, grid),
            "logreg p alone": selective_curve(truth, logreg_labels, conf, grid),
        }
        if phase == "dev":
            for name in ("llm gated by logreg p", "logreg p alone"):
                point = operating_point(result[name], bar)
                chosen[name] = point["threshold"] if point else None
        result["chosen_on_dev"] = {
            name: next((c for c in result[name] if c["threshold"] == t), None)
            for name, t in chosen.items()
        }
        result["min_precision"] = bar
        save_result("selective", phase, result | {"at": _now()})
        print(f"\n== {phase} (n={len(scored)}); thresholds chosen on dev for precision >= {bar}")
        for name, point in result["chosen_on_dev"].items():
            if point is None:
                print(f"  {name:<24} never reaches {bar} on dev")
                continue
            print(
                f"  {name:<24} t={point['threshold']:.2f}  precision {point['precision']:.2f} "
                f"[{point['precision_ci95'][0]:.2f}-{point['precision_ci95'][1]:.2f}]  "
                f"recall {point['recall']:.2f}  issues covered {point['issue_coverage']:.0%}"
            )


def cmd_eval_rerank(args: argparse.Namespace) -> None:
    from pytest_triage.evaluate import eval_rerank
    from pytest_triage.search import IssueIndex

    truth = {int(k): v for k, v in json.loads(config.DUPLICATES.read_text()).items()}
    eval_rerank(IssueIndex.build(store.load_issues()), truth, args.models)


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
    eval_dedupe(
        index, truth, negatives, expand=not args.no_expand, model=args.model, pool=args.pool
    )


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


def _live_confidence(index, labelled):
    """The high-precision gate's classifier, or None when the gate is off."""
    if config.LABEL_MIN_CONFIDENCE is None:
        return None
    from pytest_triage.learned import make_confidence

    return make_confidence(labelled.issues, lambda n: index.vectors[index.position[n]])


def cmd_eval_attacks(args: argparse.Namespace) -> None:
    from pytest_triage.attacks import MUTATIONS, run_attacks
    from pytest_triage.evaluate import _now, save_result

    issues, index, labelled = _live_indexes()
    mutations = tuple(MUTATIONS) if args.mutations else ("verbatim",)
    result = run_attacks(
        issues, index, labelled, args.model, prompt=args.prompt, mutations=mutations
    )
    key = args.model or (
        "pipeline" if args.prompt == config.LABEL_PROMPT else f"prompt {args.prompt}"
    )
    key += " + mutations" if args.mutations else ""
    save_result("attacks", key, result | {"at": _now()})
    print(
        f"\n{result['landed']}/{result['n']} attacks landed; "
        f"{result['not_reached']} never reached the model (invalid test)"
    )
    for name, r in result["rates"].items():
        lo, hi = r["ci95"]
        print(f"  {name:<13} {r['count']}/{r['n']}  (95% CI {lo:.0%}-{hi:.0%})")


def cmd_issue(args: argparse.Namespace) -> None:
    """Triage one issue from the snapshot and print the report."""
    from pytest_triage.triage import triage_issue

    _, index, labelled = _live_indexes()
    issue = index.get(args.number)
    if issue is None:
        raise SystemExit(f"#{args.number} is not in the snapshot - run `triage fetch` first.")
    confidence = None if args.all_labels else _live_confidence(index, labelled)
    report = triage_issue(issue, index, labelled, args.model, confidence=confidence)
    if args.json:
        print(json.dumps(report, indent=2))
        return
    print(f"#{report['number']} {report['title']}")
    print(f"  suggested labels: {', '.join(report['labels']) or '-'}")
    if report["high_precision"] and report["labels_all"] != report["labels"]:
        held = sorted(set(report["labels_all"]) - set(report["labels"]))
        print(f"  held back (low confidence): {', '.join(held)}")
    print("  for maintainer review (below the precision bar for a reply):")
    print(f"    possibly missing:    {', '.join(report['missing']) or '-'}")
    dupes = ", ".join(f"#{n}" for n in report["duplicates"]) or "-"
    print(f"    possible duplicates: {dupes}")
    for v in report["duplicate_verdicts"]:
        print(f"      #{v['number']} ({v['confidence']}): {v['reason']}")
    if report["injection_signals"]:
        print(f"  !! injection signals: {report['injection_signals']}")
    for n, signals in report["candidate_injection_signals"].items():
        print(f"  !! injection signals in candidate #{n}: {signals}")
    print(f"  maintainers actually applied: {', '.join(issue['labels']) or '-'}")
    print(f"  trace: {report['trace']}")
    if report["reply_draft"]:
        print("\n--- draft reply (not posted) ---\n" + report["reply_draft"])


def cmd_shadow(args: argparse.Namespace) -> None:
    from pytest_triage.shadow import run_shadow

    issues, index, labelled = _live_indexes()
    since = args.since or config.SHADOW_START
    n = run_shadow(
        issues,
        index,
        labelled,
        since=since,
        limit=args.limit,
        model=args.model,
        confidence=_live_confidence(index, labelled),
    )
    print(f"{n} new predictions -> {config.SHADOW / 'predictions.jsonl'}")


def cmd_shadow_score(args: argparse.Namespace) -> None:
    from pytest_triage.evaluate import _now, save_result
    from pytest_triage.shadow import BACKTEST, PREDICTIONS, load_predictions, score_shadow

    issues = store.load_issues()
    for key, path in (("prospective", PREDICTIONS), ("backtest", BACKTEST)):
        predictions = load_predictions(path)
        if not predictions:
            print(f"{key}: no predictions yet")
            continue
        result = score_shadow(predictions, issues)
        save_result("shadow", key, result | {"at": _now(), "since": predictions[0]["created_at"]})
        print(f"== {key}")
        print(json.dumps({k: v for k, v in result.items() if k != "labels"}, indent=2))
        print(f"label macro-F1 on maintainer-triaged: {result['labels']['macro_f1']:.2f}")


def cmd_audit(args: argparse.Namespace) -> None:
    from pytest_triage import audit
    from pytest_triage.classify import DEFINITIONS, make_llm_classifier

    _, test, memory_index = _label_setup("test")
    if args.action.startswith("guided"):
        _guided(args, test, memory_index)
        return
    if args.action == "sample":
        name = f"llm:{config.LABEL_MODEL}:{config.LABEL_PROMPT}"
        fps, tp = audit.false_positives(
            test, make_llm_classifier(memory_index, prompt=config.LABEL_PROMPT)
        )
        sample = audit.draw_sample(fps, args.n)
        audit.write_sample(
            sample,
            {
                "predictor": name,
                "seed": 11,
                "tp": tp,
                "fp": len(fps),
                "n_issues": len(test),
                "sample_size": len(sample),
            },
        )
        print(
            f"{len(fps)} false-positive labels (tp={tp}); sampled {len(sample)} -> {audit.SAMPLE}"
        )
    elif args.action == "rate-ai":
        sample, _ = audit.load_sample()
        by_number = {i["number"]: i for i in test}
        rater = audit.rate_with_model(sample, by_number, DEFINITIONS, args.model)
        print(f"model rater done -> {audit.ratings_path(rater)}")
    else:
        from pytest_triage.evaluate import _now, save_result

        sample, meta = audit.load_sample()
        raters = sorted(
            p.stem.removeprefix("ratings_") for p in audit.AUDIT.glob("ratings_*.jsonl")
        )
        result = {"meta": meta, "raters": {}}
        for rater in raters:
            ratings = audit.load_ratings(rater)
            verdicts = [
                ratings[(s["number"], s["label"])]
                for s in sample
                if (s["number"], s["label"]) in ratings
            ]
            result["raters"][rater] = audit.adjusted_precision(meta["tp"], meta["fp"], verdicts)
        if len(raters) >= 2:
            a, b = (audit.load_ratings(r) for r in raters[:2])
            both = [k for k in ((s["number"], s["label"]) for s in sample) if k in a and k in b]
            result["kappa"] = audit.cohen_kappa([a[k] for k in both], [b[k] for k in both])
            result["agreed_items"] = len(both)
        save_result("audit", "labels", result | {"at": _now()})
        print(json.dumps(result, indent=2))


def _guided(args: argparse.Namespace, test: list[dict], memory_index) -> None:
    """The guided type-label audit (see guided.py)."""
    from pytest_triage import audit, guided
    from pytest_triage.classify import make_llm_classifier
    from pytest_triage.evaluate import _now, save_result
    from pytest_triage.github import make_client

    by_number = {i["number"]: i for i in test}
    plan = guided.load_plan()
    if plan is None:  # first run: draw the warm-up and main samples once, seeded
        predict = make_llm_classifier(memory_index, prompt=config.LABEL_PROMPT)
        fps, _ = audit.false_positives(test, predict)
        types = lambda s: {x for x in s if x.startswith("type:")}  # noqa: E731
        tp = sum(len(types(predict(i)) & types(set(i["labels"]))) for i in test)
        type_fps = [f for f in fps if f["label"].startswith("type:")]
        main = guided.draw_main(type_fps, args.n)
        meta = {
            "predictor": f"llm:{config.LABEL_MODEL}:{config.LABEL_PROMPT}",
            "labels": "type: only",
            "tp": tp,
            "fp": len(type_fps),
            "n_issues": len(test),
            "sample_size": len(main),
            "seeds": {"warmup": 5, "main": 11},
        }
        plan = {"warmup": guided.draw_warmup(test), "main": main, "meta": meta}
        guided.write_plan(plan["warmup"], plan["main"], meta)

    if args.action == "guided":
        token = config.github_token()
        get_comments = guided.cached_comments(lambda: make_client(token), config.REPO)
        guided.run_session(args.rater, plan, by_number, get_comments)
    elif args.action == "guided-ai":
        rater = guided.rate_with_model(plan, by_number, args.model)
        print(f"model rater done -> {guided.answers_path(rater, 'main')}")
    else:

        def other_type(item: dict) -> bool:
            given = {x for x in by_number[item["number"]]["labels"] if x.startswith("type:")}
            return bool(given) and item["label"] not in given

        conflicts = {(m["number"], m["label"]) for m in plan["main"] if other_type(m)}
        result = guided.score(args.rater, plan, conflicts=conflicts)
        save_result("audit", f"guided type labels ({args.rater})", result | {"at": _now()})
        m = result["main"]
        print(
            f"warm-up: matched maintainers on {result['warmup_matched']}/"
            f"{result['warmup_judged']}\n"
            f"main: {m['rated']} answered ({m['unsure']} not sure); "
            f"{m['share_of_fps_correct']:.0%} of the disputed type labels judged correct\n"
            f"type-label precision: measured {m['measured_precision']:.2f}, "
            f"adjusted {m['adjusted_precision']:.2f} "
            f"({m['adjusted_ci95'][0]:.2f}-{m['adjusted_ci95'][1]:.2f})\n"
            f"cautious (the {result['conflicts']} items where maintainers chose another "
            f"type counted as wrong): {result['cautious']['adjusted_precision']:.2f} "
            f"({result['cautious']['adjusted_ci95'][0]:.2f}-"
            f"{result['cautious']['adjusted_ci95'][1]:.2f})"
        )
        for other, a in result["agreement"].items():
            print(f"agreement with {other}: {a['same']}/{a['items']} (kappa {a['kappa']:.2f})")


def cmd_report(args: argparse.Namespace) -> None:
    from pytest_triage.report import markdown_tables, write_report

    print(f"wrote {write_report()}\n")
    print(markdown_tables())


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
    p.add_argument("--prompt", default=config.LABEL_PROMPT, choices=["v1", "v2", "v3", "v3s", "v4"])
    p.add_argument(
        "--split",
        default="test",
        choices=["test", "dev"],
        help="dev = 2024 issues, for choosing variants; test = the final score",
    )
    p.add_argument("--only", nargs="*", help="run only these predictors")
    p.set_defaults(func=cmd_eval_labels)
    p = ev.add_parser("dupes", help="duplicate search hit@5, or the LLM verifier with --llm")
    p.add_argument("--llm", action="store_true", help="evaluate the LLM verifier (costs money)")
    p.add_argument("--no-expand", action="store_true", help="skip LLM query expansion")
    p.add_argument("--negatives", type=int, default=30, help="non-duplicates to test")
    p.add_argument(
        "--pool",
        default="union",
        choices=["union", "fused"],
        help="how candidates are gathered (see dedupe.candidates)",
    )
    p.add_argument("--model", default=config.WORKHORSE)
    p.set_defaults(func=cmd_eval_dupes)
    p = ev.add_parser("rerank", help="local cross-encoder rerankers vs. the union pool (free)")
    p.add_argument(
        "--models",
        nargs="*",
        default=["Xenova/ms-marco-MiniLM-L-12-v2", "BAAI/bge-reranker-v2-m3-int8"],
    )
    p.set_defaults(func=cmd_eval_rerank)
    p = ev.add_parser("regress", help="dev re-run vs. the committed baseline (exit 1 on a drop)")
    p.add_argument("--update", action="store_true", help="write a new baseline")
    p.set_defaults(func=cmd_eval_regress)
    ev.add_parser("slices", help="scores by issue shape on test (free, cached)").set_defaults(
        func=cmd_eval_slices
    )
    ev.add_parser(
        "selective", help="precision vs. coverage with a confidence threshold (free, cached)"
    ).set_defaults(func=cmd_eval_selective)
    p = ev.add_parser("learned", help="trained classifier + LLM ensembles (free, cached)")
    p.add_argument("--phase", choices=["dev", "test"], default="dev")
    p.add_argument("--variants", nargs="*", default=[], help="ensembles to run on test")
    p.set_defaults(func=cmd_eval_learned)
    ev.add_parser(
        "significance", help="paired bootstrap of macro-F1 gaps (free, cached)"
    ).set_defaults(func=cmd_eval_significance)
    p = ev.add_parser("attacks", help="prompt-injection suite against the real model")
    p.add_argument("--model", default=None, help="override config's chosen models")
    p.add_argument("--prompt", default=config.LABEL_PROMPT, choices=["v1", "v2", "v3", "v3s", "v4"])
    p.add_argument(
        "--mutations", action="store_true", help="also run 5 automatic variants of each attack"
    )
    p.set_defaults(func=cmd_eval_attacks)

    p = sub.add_parser("shadow", help="triage new issues and record predictions")
    p.add_argument("--since", help=f"YYYY-MM-DD (default: {config.SHADOW_START})")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--model", default=None, help="override config's chosen models")
    p.set_defaults(func=cmd_shadow)
    sub.add_parser("shadow-score", help="score predictions vs. maintainers").set_defaults(
        func=cmd_shadow_score
    )

    p = sub.add_parser("audit", help="audit of the model's 'false positive' labels")
    p.add_argument(
        "action",
        choices=["sample", "rate-ai", "score", "guided", "guided-ai", "guided-score"],
        help="guided = the step-by-step type-label check",
    )
    p.add_argument("--model", default=config.STEP_UP, help="rater model (rate-ai, guided-ai)")
    p.add_argument("--n", type=int, default=50, help="sample size (sample, guided)")
    p.add_argument("--rater", default="human", help="rater name (guided)")
    p.set_defaults(func=cmd_audit)

    sub.add_parser("report", help="write docs/index.html + print README tables").set_defaults(
        func=cmd_report
    )

    p = sub.add_parser("issue", help="triage one issue and print the report")
    p.add_argument("number", type=int)
    p.add_argument("--model", default=None, help="override config's chosen models")
    p.add_argument("--json", action="store_true")
    p.add_argument(
        "--all-labels", action="store_true", help="turn off high-precision mode for this issue"
    )
    p.set_defaults(func=cmd_issue)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
