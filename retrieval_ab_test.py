"""
Paired significance testing for retrieval A/B runs: is the new retriever actually
better, or did you just get lucky on your eval set?

retrieval_eval.py gives you one number per metric. You swap the chunker, MRR goes
0.612 -> 0.641, and everyone ships it. On a 50-query eval set that +0.03 is very often
noise: a handful of queries flipping rank 2 <-> rank 1 moves the mean that much. This
runs both systems' per-query scores through paired tests so you get a p-value and a
confidence interval instead of a vibe.

Paired is the bit that actually matters. Query difficulty varies way more than system
quality does, so comparing two means as if they were independent samples throws away
most of your power. Here every query is scored by both systems and we test the
per-query *differences*.

Tests (per metric):
  - paired randomization test (sign flip): under H0 the A/B labels are exchangeable
    per query, so randomly flip the sign of each diff and see how often the mean is as
    extreme as what you got. Smucker, Allan & Carterette (CIKM 2007,
    https://doi.org/10.1145/1321440.1321528) found this the one to use for IR evals,
    with bootstrap and t-test agreeing closely and sign/Wilcoxon tests misleading.
  - paired bootstrap CI on the mean difference, so you see effect size, not just a p.
  - Holm-Bonferroni across metrics. Test 8 metrics at 0.05 and you'll "find"
    something on pure noise a third of the time.
  - win/tie/loss counts, because a +0.03 mean from 3 big wins and 20 small losses is a
    different story from 20 small wins.
  - queries needed to detect a given effect at 80% power, from the observed spread of
    diffs. Usually the most useful line in the output.

Input: two JSONL files in retrieval_eval.py format (query, retrieved_ids, relevant_ids),
joined on `query_id` if present, else `query`. relevant_ids come from file A.

Requires: stdlib only

Usage:
    python retrieval_ab_test.py --demo
    python retrieval_ab_test.py baseline.jsonl candidate.jsonl
    python retrieval_ab_test.py baseline.jsonl candidate.jsonl --metrics mrr ndcg@10 --mde 0.02
    python retrieval_ab_test.py a.jsonl b.jsonl --out ab_report.json

    from retrieval_ab_test import compare
    report = compare(records_a, records_b, metrics=["mrr", "recall@5"])
    report["mrr"]["p_holm"], report["mrr"]["ci"]
"""

import json
import math
import random
import logging
import argparse
from statistics import NormalDist, fmean, stdev
from typing import Any, Callable

log = logging.getLogger("retrieval_ab_test")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

DEFAULT_METRICS = ["mrr", "hit@1", "hit@5", "recall@5", "ndcg@10"]

# 10k resamples puts the Monte Carlo error on p around +-0.002 at p=0.05, plenty.
# bump it if you're reporting p values near your alpha
DEFAULT_RESAMPLES = 10_000
DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.8  # the usual convention, nothing magic about it
DEFAULT_SEED = 13


# --- metrics (copied from retrieval_eval.py so this stays standalone) ---

def _hit(ret: list[str], rel: set[str], k: int) -> float:
    return 1.0 if any(r in rel for r in ret[:k]) else 0.0


def _recall(ret: list[str], rel: set[str], k: int) -> float:
    return sum(1 for r in ret[:k] if r in rel) / len(rel)


def _rr(ret: list[str], rel: set[str]) -> float:
    return next((1.0 / (i + 1) for i, r in enumerate(ret) if r in rel), 0.0)


def _ndcg(ret: list[str], rel: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(ret[:k]) if r in rel)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / ideal


def metric_fn(name: str) -> Callable[[list[str], set[str]], float]:
    if name == "mrr":
        return _rr
    base, _, k = name.partition("@")
    fns = {"hit": _hit, "recall": _recall, "ndcg": _ndcg}
    if base not in fns or not k.isdigit():
        raise ValueError(f"unknown metric {name!r}, use mrr or hit@k / recall@k / ndcg@k")
    return lambda ret, rel, f=fns[base], k=int(k): f(ret, rel, k)


# --- the stats ---

def permutation_test(diffs: list[float], n: int = DEFAULT_RESAMPLES, seed: int = DEFAULT_SEED) -> float:
    """two-sided paired randomization test via sign flips. returns p."""
    nz = [d for d in diffs if d != 0]  # ties contribute 0 either way, skip them for speed
    if not nz:
        return 1.0
    obs = abs(sum(nz))
    rng = random.Random(seed)
    hits = 0
    for _ in range(n):
        s = sum(d if rng.random() < 0.5 else -d for d in nz)
        if abs(s) >= obs - 1e-12:
            hits += 1
    # +1 on both sides so p is never exactly 0 (you didn't run infinite resamples)
    return (hits + 1) / (n + 1)


def bootstrap_ci(diffs: list[float], alpha: float = DEFAULT_ALPHA,
                 n: int = DEFAULT_RESAMPLES, seed: int = DEFAULT_SEED) -> tuple[float, float]:
    """percentile bootstrap CI on the mean paired difference"""
    rng = random.Random(seed + 1)
    m = len(diffs)
    means = sorted(fmean(rng.choices(diffs, k=m)) for _ in range(n))
    lo = means[int(alpha / 2 * n)]
    hi = means[min(n - 1, int((1 - alpha / 2) * n))]
    return lo, hi


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p values. uniformly better than plain Bonferroni, no extra assumptions"""
    order = sorted(pvals, key=pvals.get)
    m, running, out = len(order), 0.0, {}
    for i, name in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvals[name]))
        out[name] = running
    return out


def queries_needed(diffs: list[float], effect: float, alpha: float = DEFAULT_ALPHA,
                   power: float = DEFAULT_POWER) -> int | None:
    """
    normal-approx sample size for a paired test to detect `effect` given the observed sd
    of diffs. rough (the diffs are bounded and lumpy) but it's the right order of magnitude,
    which is what you need to decide between "label 40 more" and "label 400 more".
    """
    if len(diffs) < 2 or effect <= 0:
        return None
    sd = stdev(diffs)
    if sd == 0:
        return None
    z = NormalDist().inv_cdf
    return math.ceil(((z(1 - alpha / 2) + z(power)) * sd / effect) ** 2)


# --- glue ---

def _key(rec: dict) -> str:
    return str(rec.get("query_id") or rec.get("query"))


def pair_records(a: list[dict], b: list[dict]) -> list[tuple[dict, dict]]:
    bmap = {_key(r): r for r in b}
    pairs, missing, mismatched = [], 0, 0
    for ra in a:
        rb = bmap.get(_key(ra))
        if rb is None:
            missing += 1
            continue
        if rb.get("relevant_ids") and set(rb["relevant_ids"]) != set(ra.get("relevant_ids") or []):
            mismatched += 1
        pairs.append((ra, rb))
    if missing:
        # if this is big you're probably comparing runs on different eval set versions
        log.warning(f"{missing} queries in A have no match in B, dropped")
    if mismatched:
        log.warning(f"{mismatched} queries have different relevant_ids in A and B, using A's")
    return pairs


def compare(records_a: list[dict], records_b: list[dict], metrics: list[str] = DEFAULT_METRICS,
            alpha: float = DEFAULT_ALPHA, n_resamples: int = DEFAULT_RESAMPLES,
            mde: float | None = None, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """per-metric paired comparison of B vs A. positive delta means B is better."""
    pairs = [(ra, rb) for ra, rb in pair_records(records_a, records_b) if ra.get("relevant_ids")]
    if len(pairs) < 2:
        raise ValueError("need at least 2 paired queries with relevant_ids")

    report: dict[str, Any] = {}
    for name in metrics:
        fn = metric_fn(name)
        sa, sb = [], []
        for ra, rb in pairs:
            rel = set(ra["relevant_ids"])
            sa.append(fn(ra.get("retrieved_ids") or [], rel))
            sb.append(fn(rb.get("retrieved_ids") or [], rel))
        diffs = [y - x for x, y in zip(sa, sb)]
        lo, hi = bootstrap_ci(diffs, alpha, n_resamples, seed)
        report[name] = {
            "a": fmean(sa), "b": fmean(sb), "delta": fmean(diffs), "ci": [lo, hi],
            "p": permutation_test(diffs, n_resamples, seed),
            "wins": sum(d > 0 for d in diffs), "ties": sum(d == 0 for d in diffs),
            "losses": sum(d < 0 for d in diffs),
            # default mde: whatever you observed. "how many queries to trust this delta"
            "queries_needed": queries_needed(diffs, mde or abs(fmean(diffs)), alpha),
        }

    adj = holm({m: r["p"] for m, r in report.items()})
    for m, r in report.items():
        r["p_holm"] = adj[m]
        r["significant"] = adj[m] < alpha
    report["_meta"] = {"n_queries": len(pairs), "alpha": alpha, "resamples": n_resamples,
                       "mde": mde}
    return report


def print_report(report: dict[str, Any], label_a: str = "A", label_b: str = "B") -> None:
    meta = report["_meta"]
    ci_pct = int(round((1 - meta["alpha"]) * 100))
    print(f"\n  {label_b} vs {label_a}   n={meta['n_queries']} paired queries, "
          f"{meta['resamples']} resamples, Holm-corrected at alpha={meta['alpha']}\n")
    print(f"  {'metric':<10} {'A':>7} {'B':>7} {'delta':>8}  {f'{ci_pct}% CI':<19} "
          f"{'p':>7} {'p_holm':>7}  {'W/T/L':<11} {'n_needed':>8}")
    for m, r in report.items():
        if m == "_meta":
            continue
        ci = f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]"
        wtl = f"{r['wins']}/{r['ties']}/{r['losses']}"
        need = r["queries_needed"] if r["queries_needed"] is not None else "-"
        flag = " *" if r["significant"] else ""
        print(f"  {m:<10} {r['a']:7.3f} {r['b']:7.3f} {r['delta']:+8.3f}  {ci:<19} "
              f"{r['p']:7.4f} {r['p_holm']:7.4f}  {wtl:<11} {need:>8}{flag}")
    print("\n  * = significant after correction. n_needed = queries for 80% power at "
          + (f"delta={meta['mde']}" if meta["mde"] else "the observed delta") + "\n")


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# --- demo ---

def _synthetic_run(rng: random.Random, n_queries: int, skill: list[float],
                   n_docs: int = 200, k: int = 20) -> tuple[list[dict], list[dict]]:
    """
    two systems over the same queries. each query has a difficulty, each system finds
    the relevant doc with prob ~ skill - difficulty, and lands it at a random rank.
    shared difficulty is what makes pairing pay off.
    """
    runs: list[list[dict]] = [[] for _ in skill]
    for q in range(n_queries):
        rel = [f"d{rng.randrange(n_docs)}"]
        diff = rng.uniform(-0.3, 0.3)
        noise_seed = rng.random()
        for s, sk in enumerate(skill):
            r2 = random.Random(f"{q}-{s}-{noise_seed}")
            ret = [f"d{i}" for i in r2.sample(range(n_docs), k) if f"d{i}" not in rel][:k]
            if r2.random() < sk - diff:
                # better system also tends to rank it higher
                pos = min(int(r2.expovariate(1.0 + 3 * sk)), k - 1)
                ret.insert(pos, rel[0])
            runs[s].append({"query": f"q{q}", "retrieved_ids": ret[:k], "relevant_ids": rel})
    return runs[0], runs[1]


def demo() -> None:
    rng = random.Random(7)
    metrics = ["mrr", "hit@1", "hit@5", "ndcg@10"]

    print("=" * 96)
    print("  case 1: same system twice (pure noise), 40 queries. should NOT be significant")
    a, b = _synthetic_run(rng, 40, [0.6, 0.6])
    r1 = compare(a, b, metrics, n_resamples=4000)
    print_report(r1, "run1", "run2")

    print("=" * 96)
    print("  case 2: B is genuinely a bit better (+0.1 skill), but only 30 queries")
    a, b = _synthetic_run(rng, 30, [0.55, 0.65])
    r2 = compare(a, b, metrics, n_resamples=4000, mde=0.05)
    print_report(r2, "baseline", "candidate")

    print("=" * 96)
    print("  case 3: same +0.1 skill gap, 400 queries")
    a, b = _synthetic_run(rng, 400, [0.55, 0.65])
    r3 = compare(a, b, metrics, n_resamples=4000, mde=0.05)
    print_report(r3, "baseline", "candidate")

    # self-checks
    assert not any(r1[m]["significant"] for m in metrics), "noise flagged as significant"
    assert all(r3[m]["significant"] for m in metrics), "real gap at n=400 not detected"
    assert all(r3[m]["ci"][0] > 0 for m in metrics)
    assert holm({"x": 0.01, "y": 0.04, "z": 0.03}) == {"x": 0.03, "z": 0.06, "y": 0.06}
    assert permutation_test([0.0] * 10) == 1.0
    print("  takeaway: same true gap, case 2 can't see it and case 3 can. "
          "check n_needed before trusting a delta.")
    print("  self-checks passed")


def main():
    parser = argparse.ArgumentParser(description="Paired significance test for two retrieval runs")
    parser.add_argument("a", nargs="?", help="baseline JSONL (retrieval_eval.py format)")
    parser.add_argument("b", nargs="?", help="candidate JSONL")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS,
                        help="mrr, hit@k, recall@k, ndcg@k")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--mde", type=float, default=None,
                        help="smallest delta you care about, for the n_needed column")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", help="write the full report as JSON")
    parser.add_argument("--demo", action="store_true", help="offline demo on synthetic runs")
    args = parser.parse_args()

    if args.demo:
        demo()
        return
    if not (args.a and args.b):
        parser.error("need two JSONL files (or --demo)")

    ra, rb = load_jsonl(args.a), load_jsonl(args.b)
    log.info(f"loaded {len(ra)} records from {args.a}, {len(rb)} from {args.b}")
    report = compare(ra, rb, args.metrics, args.alpha, args.resamples, args.mde, args.seed)
    print_report(report, args.a, args.b)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        log.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
