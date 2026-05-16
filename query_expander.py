"""
Query expansion for RAG using two techniques:

1. Multi-query expansion: rephrase the user query in N different ways,
   run retrieval for each, merge + dedup results. Catches synonyms and
   paraphrase variations that single-query retrieval misses.

2. HyDE (Hypothetical Document Embeddings): generate a hypothetical answer
   to the query, embed that instead of (or alongside) the query itself.
   The intuition: the hypothetical answer lives in the same embedding space
   as real answers, so it's a better retrieval query than the raw question.

   Gao et al. 2022: https://arxiv.org/abs/2212.10496

Requires: anthropic (pip install anthropic)
Set ANTHROPIC_API_KEY in env.

Usage:
    from query_expander import expand_query, hyde_document

    variants = expand_query("what causes neural network overfitting?", n=3)
    hypo_doc = hyde_document("what causes neural network overfitting?")

    # or run standalone demo:
    python query_expander.py --query "what is RAG?" --mode both
"""

import os
import json
import time
import logging
import argparse
from typing import Literal

log = logging.getLogger("query_expander")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# tweak these if you want cheaper/faster expansions
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS_EXPAND = 400
MAX_TOKENS_HYDE = 600


def _get_client():
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set in environment")

    return anthropic.Anthropic(api_key=api_key)


def expand_query(
    query: str,
    n: int = 3,
    model: str = DEFAULT_MODEL,
    domain_hint: str = "",
) -> list[str]:
    """
    Generate N rephrased versions of the query for multi-query retrieval.

    Returns list of query strings (including the original as first element).
    """
    client = _get_client()

    domain_str = f" The queries are for a {domain_hint} knowledge base." if domain_hint else ""
    prompt = (
        f"Generate {n} different phrasings of the following search query.{domain_str}\n"
        f"Each phrasing should capture the same information need but use different words, "
        f"angles, or levels of specificity. Output ONLY a JSON array of strings, no explanation.\n\n"
        f"Query: {query}"
    )

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS_EXPAND,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()

        # parse the JSON array
        # sometimes the model wraps it in code fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        variants = json.loads(raw)
        if not isinstance(variants, list):
            raise ValueError("Expected a JSON array")

        log.info(f"Generated {len(variants)} query variants")
        return [query] + [v for v in variants if v != query]

    except Exception as e:
        log.warning(f"Query expansion failed ({e}), returning original query only")
        return [query]


def hyde_document(
    query: str,
    model: str = DEFAULT_MODEL,
    domain_hint: str = "",
    length_hint: str = "2-3 paragraphs",
) -> str:
    """
    Generate a hypothetical document that would answer the query.
    Embed this document instead of (or alongside) the raw query for retrieval.

    The idea: a hypothetical answer is closer in embedding space to real answers
    than the original question is, so retrieval precision improves.
    """
    client = _get_client()

    domain_str = f" from a {domain_hint} knowledge base" if domain_hint else ""
    prompt = (
        f"Write a hypothetical passage{domain_str} that directly and completely "
        f"answers the following question. Write as if you are the document, not as "
        f"an AI explaining the answer. Be specific and factual. Length: {length_hint}.\n\n"
        f"Question: {query}\n\nPassage:"
    )

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS_HYDE,
            messages=[{"role": "user", "content": prompt}],
        )
        hypo_doc = resp.content[0].text.strip()
        log.info(f"Generated HyDE document ({len(hypo_doc.split())} words)")
        return hypo_doc

    except Exception as e:
        log.warning(f"HyDE generation failed ({e}), falling back to original query")
        return query


def expand_and_merge_results(
    query: str,
    retriever_fn,
    top_k: int = 10,
    n_variants: int = 3,
    use_hyde: bool = False,
    domain_hint: str = "",
) -> list[dict]:
    """
    Run multi-query expansion + optional HyDE, call retriever_fn for each query,
    merge with deduplication, return top_k results.

    retriever_fn: callable(query: str, top_k: int) -> list[dict]
                  each dict should have a 'chunk_id' or 'id' field
    """
    queries = expand_query(query, n=n_variants, domain_hint=domain_hint)

    if use_hyde:
        hypo = hyde_document(query, domain_hint=domain_hint)
        queries.append(hypo)

    seen_ids = {}  # chunk_id -> (chunk_dict, best_rank)

    for q_idx, q in enumerate(queries):
        log.debug(f"Retrieving for variant {q_idx+1}: {q[:80]!r}")
        try:
            results = retriever_fn(q, top_k)
        except Exception as e:
            log.warning(f"Retrieval failed for variant {q_idx}: {e}")
            continue

        for rank, chunk in enumerate(results):
            cid = chunk.get("chunk_id") or chunk.get("id") or str(id(chunk))
            if cid not in seen_ids:
                seen_ids[cid] = (chunk, rank)
            else:
                # keep the occurrence with best (lowest) rank
                _, existing_rank = seen_ids[cid]
                if rank < existing_rank:
                    seen_ids[cid] = (chunk, rank)

    # sort by best rank across all queries
    merged = sorted(seen_ids.values(), key=lambda x: x[1])
    return [chunk for chunk, _ in merged[:top_k]]


def main():
    parser = argparse.ArgumentParser(description="Query expansion for RAG")
    parser.add_argument("--query", required=True, help="query to expand")
    parser.add_argument("--mode", choices=["multi", "hyde", "both"], default="both")
    parser.add_argument("--n", type=int, default=3, help="number of variants for multi-query")
    parser.add_argument("--domain", default="", help="domain hint (e.g. 'medical research')")
    args = parser.parse_args()

    print(f"\nOriginal query: {args.query!r}\n")

    if args.mode in ("multi", "both"):
        print("=== Multi-Query Variants ===")
        variants = expand_query(args.query, n=args.n, domain_hint=args.domain)
        for i, v in enumerate(variants):
            prefix = "(original)" if i == 0 else f"variant {i}"
            print(f"  [{prefix}] {v}")

    if args.mode in ("hyde", "both"):
        print("\n=== HyDE Document ===")
        hypo = hyde_document(args.query, domain_hint=args.domain)
        print(hypo)

    print()


if __name__ == "__main__":
    main()
