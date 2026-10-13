"""Evaluation runners. Results go to eval/results/*.json (numbers only, committed)
so the README and dashboard are generated from data, never typed by hand."""

import json
from datetime import UTC, datetime
from pathlib import Path

from pytest_triage.config import LABELS, RESULTS
from pytest_triage.metrics import Predictor, hit_at_k, score_predictor, wilson
from pytest_triage.search import IssueIndex


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def save_result(name: str, key: str, result: dict, directory: Path = RESULTS) -> None:
    """Merge one result into eval/results/<name>.json under `key`."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[key] = result
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def print_label_table(name: str, result: dict) -> None:
    print(f"\n== {name}  (n={result['n']})")
    print(f"{'label':<22}{'P':>6}{'R':>6}{'F1':>6}{'support':>9}")
    for label in LABELS:
        m = result["labels"][label]
        print(
            f"{label:<22}{m['precision']:>6.2f}{m['recall']:>6.2f}{m['f1']:>6.2f}{m['support']:>9}"
        )
    print(f"{'macro-F1':<22}{'':>12}{result['macro_f1']:>6.2f}")
    print(f"{'micro-F1':<22}{'':>12}{result['micro_f1']:>6.2f}")


def eval_labels(
    predictors: dict[str, Predictor], test: list[dict], file: str = "labels"
) -> dict[str, dict]:
    results = {}
    for name, predict in predictors.items():
        result = score_predictor(predict, test) | {"n": len(test), "at": _now()}
        save_result(file, name, result)
        print_label_table(name, result)
        results[name] = result
    return results


def eval_duplicates(index: IssueIndex, truth: dict[int, list[int]], k: int = 5) -> dict[str, dict]:
    """hit@k per retrieval method, with Wilson 95% intervals (n is small)."""
    methods = ("bm25", "dense", "hybrid")
    hits = {m: 0 for m in methods}
    print(f"\n{'dup':>7}  {'targets':<22}" + "".join(f"{m:>8}" for m in methods))
    for number, wanted in truth.items():
        query = index.get(number)
        if query is None:
            continue
        cells = []
        for m in methods:
            ranked = index.similar(query, method=m)
            hits[m] += hit_at_k(ranked, wanted, k)
            found = [ranked.index(t) + 1 for t in wanted if t in ranked]
            cells.append(str(min(found)) if found else "-")
        print(f"#{number:>6}  {wanted!s:<22}" + "".join(f"{c:>8}" for c in cells))

    n = len(truth)
    results = {}
    print(f"\nhit@{k} on {n} duplicate pairs:")
    for m in methods:
        lo, hi = wilson(hits[m], n)
        results[m] = {"hits": hits[m], "n": n, "k": k, "rate": hits[m] / n, "ci95": [lo, hi]}
        save_result("duplicates", m, results[m] | {"at": _now()})
        print(f"  {m:<7} {hits[m]:>2}/{n} = {hits[m] / n:.0%}   (95% CI {lo:.0%}-{hi:.0%})")
    return results


def sample_negatives(
    issues: list[dict], truth: dict[int, list[int]], n: int, seed: int = 7
) -> list[dict]:
    """Recent issues closed as "completed" and never marked duplicate: presumed non-duplicates.
    (Some could be undetected duplicates, which would make false-alarm counts pessimistic.)"""
    import random

    pool = [
        i
        for i in issues
        if i["created_at"] >= "2025-01-01"
        and i["state_reason"] == "completed"
        and i["number"] not in truth
        and "closed as duplicate" not in i["labels"]
    ]
    return random.Random(seed).sample(pool, min(n, len(pool)))


def eval_dedupe(
    index: IssueIndex,
    truth: dict[int, list[int]],
    negatives: list[dict],
    expand: bool,
    model: str,
    pool: str = "union",
) -> dict:
    """Candidate recall, then the verifier's high-confidence flags on
    duplicates AND non-duplicates (false alarms matter most)."""
    from pytest_triage.dedupe import find_duplicates
    from pytest_triage.llm import BadOutput
    from pytest_triage.metrics import duplicate_clusters

    clusters = duplicate_clusters(truth)
    rows = []
    for number in truth:
        # Correct = anything in the same duplicate cluster, other than the issue itself.
        rows.append((index.get(number), sorted(clusters[number] - {number})))
    rows += [(neg, []) for neg in negatives]

    in_candidates = correct = wrong = missed = false_alarms = errors = 0
    for issue, wanted in rows:
        try:
            result = find_duplicates(issue, index, expand=expand, model=model, pool=pool)
        except BadOutput:
            errors += 1
            continue
        flagged = set(result["duplicates"])
        if wanted:
            in_candidates += any(t in result["candidates"] for t in wanted)
            if flagged & set(wanted):
                correct += 1
            elif flagged:
                wrong += 1
            else:
                missed += 1
            mark = "OK " if flagged & set(wanted) else ("BAD" if flagged else "-- ")
        else:
            false_alarms += bool(flagged)
            mark = "FA " if flagged else "ok "
        print(
            f"  {mark} #{issue['number']:<6} want {wanted!s:<20} flagged {sorted(flagged)}",
            flush=True,
        )

    n_pos, n_neg = len(truth), len(negatives)
    flags = correct + wrong + false_alarms
    result = {
        "model": model,
        "expand": expand,
        "pool": pool,
        "n_duplicates": n_pos,
        "n_non_duplicates": n_neg,
        "candidate_recall": in_candidates / n_pos,
        "recall": correct / n_pos,
        "recall_ci95": list(wilson(correct, n_pos)),
        "precision": correct / flags if flags else 0.0,
        "false_alarm_rate": false_alarms / n_neg if n_neg else 0.0,
        "false_alarm_ci95": list(wilson(false_alarms, n_neg)),
        "correct": correct,
        "wrong_target": wrong,
        "missed": missed,
        "false_alarms": false_alarms,
        "errors": errors,
        "at": _now(),
    }
    result["scoring"] = "cluster-aware: a flag is correct if it is in the issue's duplicate cluster"
    key = f"{model}{'+expand' if expand else ''}" + ("" if pool == "fused" else f"+{pool}")
    save_result("dedupe", key, result)
    print(
        f"\ncandidates contain the original: {in_candidates}/{n_pos}\n"
        f"duplicates correctly flagged:    {correct}/{n_pos}"
        f" (wrong target {wrong}, missed {missed})\n"
        f"false alarms on non-duplicates:  {false_alarms}/{n_neg}\n"
        f"precision of 'high' flags:       {result['precision']:.0%}  (errors: {errors})"
    )
    return result


def eval_needs_info(
    name: str, predict, test: list[dict], label: str, file: str = "needs_info"
) -> dict:
    """Binary P/R/F1: does the predictor flag the issues maintainers asked for more info on?"""
    from pytest_triage.metrics import prf

    tp = fp = fn = 0
    for issue in test:
        truth, guess = label in issue["labels"], predict(issue)
        tp += truth and guess
        fp += guess and not truth
        fn += truth and not guess
    p, r, f1 = prf(tp, fp, fn)
    result = {"precision": p, "recall": r, "f1": f1, "support": tp + fn, "flagged": tp + fp}
    result |= {"n": len(test), "at": _now()}
    save_result(file, name, result)
    print(
        f"\n== needs-info: {name}  P {p:.2f}  R {r:.2f}  F1 {f1:.2f}  "
        f"(support {tp + fn}, flagged {tp + fp}, n={len(test)})"
    )
    return result


def eval_rerank(index: IssueIndex, truth: dict[int, list[int]], models: list[str]) -> dict:
    """Does a local cross-encoder pick the original out of a wide pool better than
    the union pool? Free: expansion queries are cached, rerankers run locally.

    wide pool = issue's own top 50 + expansion queries' top 50 (the recall ceiling)
    """
    import time

    from fastembed.rerank.cross_encoder import TextCrossEncoder

    from pytest_triage.dedupe import candidates, expand_queries
    from pytest_triage.metrics import duplicate_clusters
    from pytest_triage.search import rrf

    clusters = duplicate_clusters(truth)

    def text(issue: dict, chars: int) -> str:
        return f"{issue['title']}\n{(issue['body'] or '')[:chars]}"

    pools = {}
    for n in truth:
        q = index.get(n)
        queries = expand_queries(q)
        own = index.similar(q, k=50)
        extra = [x for x in rrf(index.search_texts(queries, index.allowed_for(q))) if x not in own]
        pools[n] = (q, own + extra[:50], candidates(q, index, queries))

    def found(n: int, pool: list[int]) -> bool:
        return bool((clusters[n] - {n}) & set(pool))

    result = {
        "n": len(truth),
        "union_8_8": sum(found(n, c) for n, (_, _, c) in pools.items()),
        "wide_pool_ceiling": sum(found(n, w) for n, (_, w, _) in pools.items()),
        "rerankers": {},
    }
    for name in models:
        encoder, started, top = TextCrossEncoder(name), time.time(), {8: 0, 16: 0}
        for n, (q, wide, _) in pools.items():
            scores = list(encoder.rerank(text(q, 1000), [text(index.get(c), 600) for c in wide]))
            order = [c for _, c in sorted(zip(scores, wide, strict=True), key=lambda x: -x[0])]
            for k in top:
                top[k] += found(n, order[:k])
        seconds = (time.time() - started) / len(pools)
        result["rerankers"][name] = {"top8": top[8], "top16": top[16], "s_per_issue": seconds}
        print(
            f"  {name:<34} top8 {top[8]}/{len(pools)}  top16 {top[16]}/{len(pools)}  "
            f"{seconds:.1f}s/issue",
            flush=True,
        )
    print(
        f"  union pool (8+8): {result['union_8_8']}/{len(pools)}; "
        f"wide-pool ceiling: {result['wide_pool_ceiling']}/{len(pools)}"
    )
    save_result("rerank", "candidate_recall", result | {"at": _now()})
    return result
