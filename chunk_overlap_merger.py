"""
Detect and merge overlapping consecutive chunks.

When you process documents with different chunkers, or use sliding-window
chunking, you end up with chunks that partially repeat each other. This
script finds those overlaps and stitches them back together.

Algorithm:
  For each pair of consecutive chunks (ordered by doc + chunk_index):
    1. Compute the longest common suffix of chunk[i] that matches a
       prefix of chunk[i+1] (approximate, character-level)
    2. If overlap exceeds threshold, merge by concatenating without the
       repeated portion

Works purely on text — no embeddings needed.

Usage:
    python chunk_overlap_merger.py chunks.json --out merged.json
    python chunk_overlap_merger.py chunks.json --overlap-chars 80 --show-merges
"""

import re
import sys
import json
import logging
import argparse
from pathlib import Path
from itertools import groupby
from typing import Any

log = logging.getLogger("overlap_merger")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

DEFAULT_OVERLAP_CHARS = 60     # minimum overlap in characters to trigger merge
DEFAULT_MAX_OVERLAP_CHARS = 600  # if overlap is bigger than this, something's wrong


def _normalize(text: str) -> str:
    """strip whitespace noise for comparison"""
    return re.sub(r"\s+", " ", text).strip()


def _find_overlap_length(a: str, b: str, min_len: int = 20, max_check: int = 800) -> int:
    """
    Find the length of the longest suffix of `a` that is a prefix of `b`.
    Returns 0 if no overlap >= min_len found.

    Uses a sliding window: for each possible overlap length (from max down to min),
    check if suffix(a, L) == prefix(b, L).
    """
    a_norm = _normalize(a)
    b_norm = _normalize(b)

    # only check up to max_check chars to stay O(n) per pair
    a_check = a_norm[-max_check:]
    b_check = b_norm[:max_check]

    best = 0
    for length in range(min(len(a_check), len(b_check)), min_len - 1, -1):
        if a_check[-length:] == b_check[:length]:
            best = length
            break  # longest found, stop

    return best


def _merge_two(a: dict, b: dict, overlap_len: int) -> dict:
    """
    Merge chunk b into chunk a by removing the overlap from the start of b.
    Keeps metadata from a, updates text and chunk_id.
    """
    a_text = a.get("text") or ""
    b_text = b.get("text") or ""

    # find the actual split point in b_text (un-normalized)
    # we'll use the normalized overlap length as an approximation
    # and trim by word boundary to avoid mid-word cuts
    b_norm = _normalize(b_text)
    trimmed_b_norm = b_norm[overlap_len:].lstrip()

    # try to find where trimmed_b_norm starts in original b_text
    # by looking for the first word of trimmed_b_norm
    if trimmed_b_norm:
        first_words = " ".join(trimmed_b_norm.split()[:3])
        idx = b_text.find(first_words[:30])
        if idx > 0:
            b_tail = b_text[idx:]
        else:
            b_tail = trimmed_b_norm
    else:
        b_tail = ""

    merged_text = (a_text.rstrip() + " " + b_tail.lstrip()).strip()

    merged = dict(a)
    merged["text"] = merged_text
    merged["raw_text"] = merged_text
    # mark as merged for debugging
    merged["chunk_id"] = a.get("chunk_id", "") + "+merged"
    merged["_merged_from"] = [
        a.get("chunk_id", "?"),
        b.get("chunk_id", "?"),
    ]
    return merged


def merge_overlapping_chunks(
    chunks: list[dict[str, Any]],
    min_overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    max_overlap_chars: int = DEFAULT_MAX_OVERLAP_CHARS,
    group_by_doc: bool = True,
) -> tuple[list[dict], int]:
    """
    Main entry point.

    Returns:
        (merged_chunks, n_merges_performed)
    """
    if not chunks:
        return [], 0

    # sort by doc + chunk_index for consecutive comparison to make sense
    def sort_key(c):
        return (
            c.get("doc_name") or "",
            c.get("chunk_index") if c.get("chunk_index") is not None else 9999,
        )

    sorted_chunks = sorted(chunks, key=sort_key)

    result = []
    n_merges = 0
    i = 0

    while i < len(sorted_chunks):
        current = sorted_chunks[i]

        if i + 1 >= len(sorted_chunks):
            result.append(current)
            break

        nxt = sorted_chunks[i + 1]

        # don't merge across documents
        if group_by_doc and current.get("doc_name") != nxt.get("doc_name"):
            result.append(current)
            i += 1
            continue

        text_a = current.get("text") or current.get("raw_text") or ""
        text_b = nxt.get("text") or nxt.get("raw_text") or ""

        overlap = _find_overlap_length(
            text_a, text_b, min_len=min_overlap_chars, max_check=max_overlap_chars
        )

        if overlap >= min_overlap_chars:
            merged = _merge_two(current, nxt, overlap)
            log.debug(
                f"Merged {current.get('chunk_id')} + {nxt.get('chunk_id')} "
                f"(overlap={overlap} chars)"
            )
            n_merges += 1
            # replace current with merged, skip nxt, re-check the merged chunk
            # against the one after nxt (chain merging)
            sorted_chunks[i] = merged
            sorted_chunks.pop(i + 1)
        else:
            result.append(current)
            i += 1

    if sorted_chunks and (not result or result[-1] is not sorted_chunks[-1]):
        result.append(sorted_chunks[i] if i < len(sorted_chunks) else sorted_chunks[-1])

    return result, n_merges


def main():
    parser = argparse.ArgumentParser(description="Merge overlapping consecutive chunks")
    parser.add_argument("input", help="chunks JSON file")
    parser.add_argument("--overlap-chars", type=int, default=DEFAULT_OVERLAP_CHARS,
                        help="minimum character overlap to trigger merge")
    parser.add_argument("--out", default=None)
    parser.add_argument("--show-merges", action="store_true",
                        help="print chunks that were merged")
    parser.add_argument("--no-group-by-doc", action="store_true",
                        help="merge across document boundaries (usually wrong)")
    args = parser.parse_args()

    p = Path(args.input)
    with p.open() as f:
        chunks = json.load(f)
    log.info(f"Loaded {len(chunks)} chunks")

    merged, n_merges = merge_overlapping_chunks(
        chunks,
        min_overlap_chars=args.overlap_chars,
        group_by_doc=not args.no_group_by_doc,
    )

    log.info(f"Performed {n_merges} merges. {len(chunks)} → {len(merged)} chunks")

    if args.show_merges:
        for c in merged:
            if "_merged_from" in c:
                print(f"  merged: {c['_merged_from']} → {c['chunk_id']}")
                print(f"    text preview: {(c.get('text') or '')[:120]!r}")
                print()

    out_path = args.out or str(p.with_stem(p.stem + "_merged"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    log.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
