"""
Offline retrieval evaluation for RAG pipelines.

Computes standard IR metrics from a JSONL file where each line is:
  {
    "query": "what is photosynthesis?",
    "retrieved_ids": ["chunk_3", "chunk_7", "chunk_1", ...],  # ranked list
    "relevant_ids": ["chunk_3", "chunk_15"]                   # ground truth
  }

Metrics:
  - Hit@k      : was any relevant doc in top-k?
  - Precision@k: fraction of top-k that are relevant
  - Recall@k   : fraction of relevant docs found in top-k
  - MRR        : mean reciprocal rank (how high is the first relevant result?)
  - NDCG@k     : normalized discounted cumulative gain

Usage:
    python retrieval_eval.py results.jsonl --k 5 10 20
    python retrieval_eval.py results.jsonl --k 5 --verbose

You can generate a dummy test file with:
    python retrieval_eval.py --generate-sample sample.jsonl
"""

import math
import json
import sys
import logging
import argparse
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("retrieval_eval")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(r in relevant for r in retrieved[:k]) else 0.0


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 1.0  # edge case: nothing to recall
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, r in enumerate(retrieved):
        if r in relevant:
            return 1.0 / (i + 1)
    return 0.0


def dcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = 0.0
    for i, r in enumerate(retrieved[:k]):
        if r in relevant:
            # binary relevance, standard DCG formula
            dcg += 1.0 / math.log2(i + 2)
    return dcg


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    actual_dcg = dcg_at_k(retrieved, relevant, k)
    # ideal: all relevant docs at the top
    n_rel = min(len(relevant), k)
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(n_rel))
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def average_precision(retrieved: list[str], relevant: set[str]) -> float:
    """AP: area under the precision-recall curve"""
    hits = 0
    cumulative_precision = 0.0
    for i, r in enumerate(retrieved):
        if r in relevant:
            hits += 1
            cumulative_precision += hits / (i + 1)
    if not relevant:
        return 0.0
    return cumulative_precision / len(relevant)


def evaluate(records: list[dict], ks: list[int] = [1, 5, 10]) -> dict:
    """
    records: list of dicts with retrieved_ids, relevant_ids, query
    returns: dict of metric_name -> value (averaged over all queries)
    """
    accum = defaultdict(list)

    for rec in records:
        retrieved = rec.get("retrieved_ids") or []
        relevant = set(rec.get("relevant_ids") or [])

        if not relevant:
            log.debug(f"Query '{rec.get('query', '?')[:60]}' has no relevant_ids — skipping")
            continue

        accum["mrr"].append(reciprocal_rank(retrieved, relevant))
        accum["map"].append(average_precision(retrieved, relevant))

        for k in ks:
            accum[f"hit@{k}"].append(hit_at_k(retrieved, relevant, k))
            accum[f"p@{k}"].append(precision_at_k(retrieved, relevant, k))
            accum[f"recall@{k}"].append(recall_at_k(retrieved, relevant, k))
            accum[f"ndcg@{k}"].append(ndcg_at_k(retrieved, relevant, k))

    results = {}
    for metric, values in accum.items():
        results[metric] = round(sum(values) / len(values), 4) if values else 0.0

    results["n_queries"] = len(records)
    results["n_evaluated"] = len(accum.get("mrr", []))
    return results


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _generate_sample(out_path: str, n_queries: int = 20, n_chunks: int = 100, k: int = 10):
    """generate a dummy evaluation file for testing"""
    import random
    random.seed(42)
    chunk_ids = [f"chunk_{i}" for i in range(n_chunks)]
    records = []
    for i in range(n_queries):
        relevant = random.sample(chunk_ids, random.randint(1, 3))
        retrieved = random.sample(chunk_ids, min(k, n_chunks))
        # sprinkle some relevant ones in to make it interesting
        for rel in relevant:
            if rel not in retrieved and random.random() > 0.3:
                pos = random.randint(0, min(k-1, len(retrieved)))
                retrieved.insert(pos, rel)
        records.append({
            "query": f"sample query {i}",
            "retrieved_ids": retrieved[:k],
            "relevant_ids": relevant,
        })
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {n_queries} sample records to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval from JSONL")
    parser.add_argument("input", nargs="?", help="JSONL file with evaluation records")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--generate-sample", metavar="OUT",
                        help="generate a sample JSONL file for testing")
    args = parser.parse_args()

    if args.generate_sample:
        _generate_sample(args.generate_sample)
        return

    if not args.input:
        parser.print_help()
        sys.exit(1)

    records = load_jsonl(args.input)
    log.info(f"Loaded {len(records)} records from {args.input}")

    results = evaluate(records, ks=sorted(args.k))

    print(f"\n{'=' * 40}")
    print(f"  Retrieval Evaluation Results")
    print(f"  Queries evaluated: {results['n_evaluated']}/{results['n_queries']}")
    print(f"{'=' * 40}")

    # print in a somewhat readable order
    priority = ["mrr", "map"] + [f"hit@{k}" for k in sorted(args.k)] + \
               [f"ndcg@{k}" for k in sorted(args.k)] + \
               [f"p@{k}" for k in sorted(args.k)] + \
               [f"recall@{k}" for k in sorted(args.k)]

    for m in priority:
        if m in results:
            print(f"  {m:<15} {results[m]:.4f}")

    print()

    if args.verbose:
        print("Full results:")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
