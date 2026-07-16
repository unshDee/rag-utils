"""
Answer-side RAG evaluation. retrieval_eval.py scores whether you found the right
chunks; this scores what the LLM then did with them.

Metrics (LLM-judged, decomposed into atomic claims rather than a 1-10 vibes score):
  - faithfulness     : answer claims supported by the retrieved context / total claims.
                       This is your hallucination rate.
  - answer_relevance : does the answer address the question, or is it on-topic waffle?
                       Judged by reverse-engineering questions from the answer.
  - context_precision: how much of what you retrieved was actually useful
  - context_recall   : claims in the ground-truth answer that the context supports.
                       Ceiling on faithfulness — the generator can't cite what you never
                       gave it, so this splits blame between retriever and generator.
                       Skipped when ground_truth is absent.
  - citations        : if the answer has [1]-style markers, does each cited chunk really
                       support the sentence it's attached to?

Metric framing follows RAGAS (Es et al. 2023, https://arxiv.org/abs/2309.15217),
reimplemented with no framework so the judge prompts are readable and editable.

Requires: anthropic (pip install anthropic)
Set ANTHROPIC_API_KEY in env.

Input JSONL, one object per line:
  {
    "query": "what is RRF?",
    "answer": "RRF merges ranked lists by summing 1/(k+rank) [1].",
    "contexts": ["chunk text", ...],
    "ground_truth": "optional reference answer"
  }

Usage:
    python answer_eval.py --generate-sample sample.jsonl
    python answer_eval.py evals.jsonl --out report.json
"""

import os
import re
import json
import logging
import argparse
import statistics
from concurrent.futures import ThreadPoolExecutor
from typing import Any

log = logging.getLogger("answer_eval")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# judge should be at least as strong as the generator, or it rubber-stamps
# hallucinations it can't detect
DEFAULT_JUDGE = "claude-sonnet-5"

CITATION_RE = re.compile(r"\[(\d+)\]")


def _get_client():
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY not set in environment")
    return anthropic.Anthropic()


def _ask_json(client, prompt: str, model: str, max_tokens: int = 2000) -> Any:
    resp = client.messages.create(
        model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}]
    )
    raw = "".join(b.text for b in resp.content if b.type == "text").strip()

    # models sometimes wrap JSON in code fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw.strip())


def _format_contexts(contexts: list[str]) -> str:
    return "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))


def faithfulness(client, answer: str, contexts: list[str], model: str) -> dict[str, Any]:
    """
    Decompose the answer into atomic claims, then check each against the context.

    Two steps on purpose: a single "is this faithful?" call lets the judge average away
    one bad sentence in an otherwise good answer. Per-claim verdicts don't.
    """
    prompt = (
        "Break the ANSWER into atomic factual claims — one verifiable statement each, "
        "with pronouns resolved. Then decide, for each claim, whether it is directly "
        "supported by the CONTEXT.\n\n"
        "A claim is supported ONLY if the context states or entails it. Plausible, "
        "true-in-general, or common-knowledge claims that the context does not state "
        "are NOT supported.\n\n"
        f"CONTEXT:\n{_format_contexts(contexts)}\n\n"
        f"ANSWER:\n{answer}\n\n"
        'Output ONLY JSON: {"claims": [{"claim": "...", "supported": true|false, '
        '"evidence": "quote from context, or empty if unsupported"}]}'
    )
    result = _ask_json(client, prompt, model)
    claims = result.get("claims", [])
    if not claims:
        return {"score": 0.0, "claims": []}

    supported = sum(1 for c in claims if c.get("supported"))
    return {
        "score": supported / len(claims),
        "n_claims": len(claims),
        "n_unsupported": len(claims) - supported,
        "claims": claims,
    }


def answer_relevance(client, query: str, answer: str, model: str, n: int = 3) -> dict[str, Any]:
    """
    Generate N questions the answer would be a good reply to, then score how close they
    are to the real query. A padded or evasive answer generates questions that drift.
    """
    prompt = (
        f"Read the ANSWER and write {n} questions that it would be a direct, complete "
        f"reply to. Then rate 0.0-1.0 how well each generated question matches the "
        f"ORIGINAL QUESTION in intent.\n\n"
        f"Also flag the answer as evasive if it dodges the question, refuses, or says "
        f"it lacks the information.\n\n"
        f"ORIGINAL QUESTION: {query}\n\nANSWER: {answer}\n\n"
        'Output ONLY JSON: {"generated": [{"question": "...", "similarity": 0.0}], '
        '"evasive": true|false}'
    )
    result = _ask_json(client, prompt, model)
    sims = [g.get("similarity", 0.0) for g in result.get("generated", [])]
    score = statistics.mean(sims) if sims else 0.0
    if result.get("evasive"):
        score = 0.0
    return {
        "score": score,
        "evasive": bool(result.get("evasive")),
        "generated": result.get("generated", []),
    }


def context_precision(client, query: str, contexts: list[str], model: str) -> dict[str, Any]:
    """fraction of retrieved chunks that actually help answer the query"""
    if not contexts:
        return {"score": 0.0, "verdicts": []}

    prompt = (
        "For each numbered CONTEXT passage, decide whether it contains information that "
        "helps answer the QUESTION. Being on the same topic is not enough — it must "
        "contribute to an answer.\n\n"
        f"QUESTION: {query}\n\n{_format_contexts(contexts)}\n\n"
        'Output ONLY JSON: {"verdicts": [{"index": 1, "useful": true|false, "why": "..."}]}'
    )
    result = _ask_json(client, prompt, model)
    verdicts = result.get("verdicts", [])
    useful = [v for v in verdicts if v.get("useful")]
    return {
        "score": len(useful) / len(contexts),
        "n_useful": len(useful),
        "n_retrieved": len(contexts),
        "verdicts": verdicts,
    }


def context_recall(client, ground_truth: str, contexts: list[str], model: str) -> dict[str, Any]:
    """
    Claims in the reference answer that the context supports.

    Isolates blame: low faithfulness + high recall means the generator is hallucinating;
    low faithfulness + low recall means the retriever starved it.
    """
    prompt = (
        "Break the REFERENCE ANSWER into atomic claims. For each, decide whether it can "
        "be attributed to the CONTEXT.\n\n"
        f"CONTEXT:\n{_format_contexts(contexts)}\n\n"
        f"REFERENCE ANSWER:\n{ground_truth}\n\n"
        'Output ONLY JSON: {"claims": [{"claim": "...", "attributable": true|false}]}'
    )
    result = _ask_json(client, prompt, model)
    claims = result.get("claims", [])
    if not claims:
        return {"score": 0.0, "claims": []}

    hit = sum(1 for c in claims if c.get("attributable"))
    return {"score": hit / len(claims), "n_claims": len(claims), "claims": claims}


def citation_check(client, answer: str, contexts: list[str], model: str) -> dict[str, Any] | None:
    """
    Verify [n]-style citations. Returns None if the answer has none.

    Two failure modes: out_of_range (cites [7] when you gave it 5 chunks) and unsupported
    (cites a real chunk that doesn't say it — worse, since it passes a human skim).
    """
    cited = sorted({int(m) for m in CITATION_RE.findall(answer)})
    if not cited:
        return None

    out_of_range = [c for c in cited if c < 1 or c > len(contexts)]

    prompt = (
        "Each sentence in the ANSWER carries citations like [1], [2] pointing at the "
        "numbered CONTEXT passages. For every citation, decide whether the cited passage "
        "actually supports the statement it is attached to.\n\n"
        f"CONTEXT:\n{_format_contexts(contexts)}\n\n"
        f"ANSWER:\n{answer}\n\n"
        'Output ONLY JSON: {"citations": [{"marker": 1, "statement": "...", '
        '"supports": true|false}]}'
    )
    result = _ask_json(client, prompt, model)
    checks = result.get("citations", [])
    good = sum(1 for c in checks if c.get("supports"))
    total = len(checks) + len(out_of_range)

    return {
        "score": good / total if total else 0.0,
        "n_citations": len(cited),
        "out_of_range": out_of_range,
        "checks": checks,
    }


def evaluate_sample(sample: dict[str, Any], model: str = DEFAULT_JUDGE, client=None) -> dict[str, Any]:
    """score one {query, answer, contexts, ground_truth?} record"""
    client = client or _get_client()
    query = sample["query"]
    answer = sample["answer"]
    contexts = sample.get("contexts", [])

    out: dict[str, Any] = {"query": query}

    try:
        out["faithfulness"] = faithfulness(client, answer, contexts, model)
        out["answer_relevance"] = answer_relevance(client, query, answer, model)
        out["context_precision"] = context_precision(client, query, contexts, model)

        if sample.get("ground_truth"):
            out["context_recall"] = context_recall(client, sample["ground_truth"], contexts, model)

        cites = citation_check(client, answer, contexts, model)
        if cites is not None:
            out["citations"] = cites

    except Exception as e:
        log.warning(f"Judge failed on '{query[:50]}': {e}")
        out["error"] = str(e)

    return out


def evaluate_dataset(
    samples: list[dict[str, Any]], model: str = DEFAULT_JUDGE, max_workers: int = 4
) -> dict[str, Any]:
    client = _get_client()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(lambda s: evaluate_sample(s, model, client), samples))

    metrics = ["faithfulness", "answer_relevance", "context_precision", "context_recall", "citations"]
    aggregate = {}
    for m in metrics:
        scores = [r[m]["score"] for r in results if "score" in r.get(m, {})]
        if scores:
            aggregate[m] = {
                "mean": round(statistics.mean(scores), 4),
                "median": round(statistics.median(scores), 4),
                "min": round(min(scores), 4),
                "n": len(scores),
            }

    return {"aggregate": aggregate, "samples": results}


def print_report(report: dict[str, Any]):
    print(f"\n{'=' * 56}")
    print("  Answer Evaluation Results")
    print(f"{'=' * 56}")
    print(f"  {'metric':<20} {'mean':>7} {'median':>7} {'min':>7} {'n':>5}")
    print(f"{'-' * 56}")
    for name, s in report["aggregate"].items():
        print(f"  {name:<20} {s['mean']:>7.3f} {s['median']:>7.3f} {s['min']:>7.3f} {s['n']:>5}")
    print(f"{'=' * 56}")

    worst = sorted(
        (r for r in report["samples"] if "faithfulness" in r),
        key=lambda r: r["faithfulness"]["score"],
    )[:3]

    unfaithful = [r for r in worst if r["faithfulness"]["score"] < 1.0]
    if unfaithful:
        print("\nLeast faithful answers:")
        for r in unfaithful:
            print(f"\n  [{r['faithfulness']['score']:.2f}] {r['query'][:70]}")
            for c in r["faithfulness"]["claims"]:
                if not c.get("supported"):
                    print(f"      unsupported: {c['claim'][:90]}")
    print()


SAMPLE = [
    {
        "query": "What is reciprocal rank fusion?",
        "answer": "Reciprocal Rank Fusion merges several ranked lists by summing 1/(k+rank) for each document, with k=60 by default [1]. It was invented at Google in 2015 and is the default fusion method in Elasticsearch.",
        "contexts": [
            "Reciprocal Rank Fusion (RRF) combines multiple ranked lists by summing 1/(k + rank) over the lists in which a document appears. The constant k is typically set to 60.",
            "Dense retrieval encodes queries and documents into a shared vector space.",
        ],
        "ground_truth": "RRF fuses ranked lists by summing 1/(k+rank) across retrievers, typically with k=60, requiring no score normalization.",
    },
    {
        "query": "Why use a cross-encoder for reranking?",
        "answer": "A cross-encoder encodes the query and the document together, so attention runs across both and it produces much more accurate relevance scores than a bi-encoder [1]. The cost is that it cannot be precomputed, so it is only run over a shortlist [1].",
        "contexts": [
            "Cross-encoders jointly encode query and document, enabling full cross-attention and yielding markedly better relevance estimates than bi-encoders. Because scores cannot be precomputed, cross-encoders are applied only to a small candidate set produced by a first-stage retriever.",
        ],
        "ground_truth": "Cross-encoders jointly attend over query and document, giving better relevance than bi-encoders, but must be limited to a shortlist since they cannot be precomputed.",
    },
]


def _generate_sample(out_path: str):
    """generate a dummy evaluation file for testing — the first answer hallucinates"""
    with open(out_path, "w") as f:
        for s in SAMPLE:
            f.write(json.dumps(s) + "\n")
    print(f"Wrote {len(SAMPLE)} sample records to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="LLM-judged RAG answer evaluation")
    parser.add_argument("input", nargs="?", help="JSONL file with evaluation records")
    parser.add_argument("--model", default=DEFAULT_JUDGE, help="judge model")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", help="write the full JSON report here")
    parser.add_argument("--generate-sample", metavar="OUT",
                        help="generate a sample JSONL file for testing")
    args = parser.parse_args()

    if args.generate_sample:
        _generate_sample(args.generate_sample)
        return

    if not args.input:
        parser.print_help()
        return

    samples = [json.loads(line) for line in open(args.input) if line.strip()]
    log.info(f"Loaded {len(samples)} samples from {args.input} (judge={args.model})")

    report = evaluate_dataset(samples, model=args.model, max_workers=args.workers)
    print_report(report)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        log.info(f"Saved report to {args.out}")


if __name__ == "__main__":
    main()
