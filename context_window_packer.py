"""
Given a list of retrieved chunks + a token budget, pick the best subset
to fit inside a context window. Uses a greedy fractional-knapsack approach
(sorted by score/token_cost ratio) which is near-optimal in practice and
runs in O(n log n).

Also handles:
- minimum score threshold (skip garbage retrievals)
- deduplication by chunk_id
- summary of what got packed vs dropped

Standalone — only stdlib + tiktoken (optional; falls back to char estimate).

Usage:
    from context_window_packer import pack_context

    chunks_with_scores = [
        {"chunk_id": "a", "text": "...", "score": 0.91},
        {"chunk_id": "b", "text": "...", "score": 0.74},
    ]
    packed = pack_context(chunks_with_scores, budget=3000)
"""

import re
import sys
import json
import logging
import argparse
from typing import Any

log = logging.getLogger("ctx_packer")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


def _count_tokens_tiktoken(text: str, enc) -> int:
    return len(enc.encode(text))


def _count_tokens_approx(text: str) -> int:
    # rough heuristic: ~4 chars per token for English
    return max(1, len(text) // 4)


def _get_token_counter(model: str = "gpt-4o"):
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model(model)
        return lambda t: _count_tokens_tiktoken(t, enc)
    except Exception:
        log.debug("tiktoken not available, using char-based estimate")
        return _count_tokens_approx


def pack_context(
    chunks: list[dict[str, Any]],
    budget: int = 4000,
    score_key: str = "score",
    text_key: str = "text",
    min_score: float = 0.0,
    model: str = "gpt-4o",
    system_prompt_tokens: int = 0,
) -> dict[str, Any]:
    """
    Pack chunks greedily into a token budget.

    Args:
        chunks: list of dicts, each must have `text` and `score` (or your keys)
        budget: max tokens available (subtract system prompt tokens first)
        score_key: key for retrieval score
        text_key: key for text content
        min_score: drop chunks below this score before packing
        model: used for tiktoken encoding (falls back to char estimate)
        system_prompt_tokens: tokens already used by system prompt

    Returns:
        dict with keys: packed_chunks, total_tokens, dropped_count, utilization
    """
    count_tokens = _get_token_counter(model)
    remaining = budget - system_prompt_tokens

    if remaining <= 0:
        raise ValueError("budget is entirely consumed by system_prompt_tokens")

    # dedup by chunk_id, keep highest score
    seen_ids: dict[str, dict] = {}
    for c in chunks:
        cid = c.get("chunk_id") or c.get("id") or id(c)
        existing = seen_ids.get(str(cid))
        if existing is None or c.get(score_key, 0) > existing.get(score_key, 0):
            seen_ids[str(cid)] = c

    candidates = list(seen_ids.values())

    # filter by min score
    if min_score > 0:
        before = len(candidates)
        candidates = [c for c in candidates if c.get(score_key, 0) >= min_score]
        if before > len(candidates):
            log.debug(f"min_score filter removed {before - len(candidates)} chunks")

    # attach token counts
    for c in candidates:
        c["_tokens"] = count_tokens(c.get(text_key) or "")

    # sort by score/token_cost — this is the greedy fractional knapsack key
    # chunks with high score per token go first
    candidates.sort(
        key=lambda c: c.get(score_key, 0) / max(c["_tokens"], 1),
        reverse=True,
    )

    packed = []
    total_tokens = 0
    dropped = []

    for c in candidates:
        tok = c["_tokens"]
        if total_tokens + tok <= remaining:
            packed.append(c)
            total_tokens += tok
        else:
            # try to squeeze it in if it's small (partial fill heuristic)
            if tok <= 50 and total_tokens + tok <= remaining + 50:
                packed.append(c)
                total_tokens += tok
            else:
                dropped.append(c)

    utilization = total_tokens / remaining if remaining > 0 else 0

    log.info(
        f"Packed {len(packed)}/{len(candidates)} chunks "
        f"({total_tokens}/{remaining} tokens, {utilization:.1%} utilization)"
    )

    if dropped:
        log.debug(f"Dropped chunks: {[c.get('chunk_id', '?') for c in dropped]}")

    # clean up internal key before returning
    for c in packed + dropped:
        c.pop("_tokens", None)

    return {
        "packed_chunks": packed,
        "total_tokens": total_tokens,
        "budget": remaining,
        "dropped_count": len(dropped),
        "utilization": round(utilization, 4),
    }


def build_context_string(
    packed_chunks: list[dict],
    text_key: str = "text",
    separator: str = "\n\n---\n\n",
    include_source: bool = True,
) -> str:
    """Convert packed chunks to a single context string for the LLM prompt."""
    parts = []
    for c in packed_chunks:
        text = c.get(text_key) or ""
        if include_source:
            src = c.get("doc_name") or c.get("source_path") or c.get("chunk_id") or ""
            if src:
                text = f"[Source: {src}]\n{text}"
        parts.append(text)
    return separator.join(parts)


def main():
    parser = argparse.ArgumentParser(description="Pack retrieved chunks into a token budget")
    parser.add_argument("input", help="JSON file with chunks (must have 'text' and 'score')")
    parser.add_argument("--budget", type=int, default=4000)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--system-tokens", type=int, default=0)
    parser.add_argument("--print-context", action="store_true")
    args = parser.parse_args()

    with open(args.input) as f:
        chunks = json.load(f)

    result = pack_context(
        chunks,
        budget=args.budget,
        min_score=args.min_score,
        model=args.model,
        system_prompt_tokens=args.system_tokens,
    )

    print(json.dumps({
        "total_tokens": result["total_tokens"],
        "budget": result["budget"],
        "packed": len(result["packed_chunks"]),
        "dropped": result["dropped_count"],
        "utilization": result["utilization"],
    }, indent=2))

    if args.print_context:
        print("\n=== PACKED CONTEXT ===\n")
        print(build_context_string(result["packed_chunks"]))


if __name__ == "__main__":
    main()
