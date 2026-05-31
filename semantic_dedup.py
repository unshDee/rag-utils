"""
Near-duplicate chunk removal using embedding cosine similarity.

Useful when you've ingested the same document multiple times with different
chunkers, or when your corpus has overlapping sources (wikis, mirrors, etc.).

Strategy: embed all chunks, build a similarity graph, then greedily keep
the highest-scoring chunk from each cluster. Falls back to text-based
Jaccard similarity if no embedder is provided.

Requires: numpy sentence-transformers (for embedding mode)
Jaccard mode: stdlib only

Usage:
    # embedding mode (better, slower)
    python semantic_dedup.py chunks.json --threshold 0.92 --out deduped.json

    # fast jaccard mode (no deps beyond stdlib)
    python semantic_dedup.py chunks.json --mode jaccard --threshold 0.7
"""

import json
import time
import logging
import argparse
import hashlib
from pathlib import Path
from typing import Callable

import numpy as np

log = logging.getLogger("semantic_dedup")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _shingle(text: str, k=5) -> set[str]:
    """character k-shingles — fast and decent for near-dup detection"""
    text = " ".join(text.lower().split())
    return {text[i:i+k] for i in range(len(text) - k + 1)}


def jaccard_dedup(chunks: list[dict], threshold: float = 0.7) -> list[dict]:
    log.info(f"Jaccard dedup on {len(chunks)} chunks (threshold={threshold})")
    shingles = [_shingle(c.get("text") or c.get("raw_text") or "") for c in chunks]
    kept = []
    dropped = 0

    # O(n^2) but fine for <10k chunks; can swap to MinHash if needed
    for i, (chunk, shin_i) in enumerate(zip(chunks, shingles)):
        dup = False
        for j in range(i):
            if j in {k_idx for k_idx, _ in kept}:
                if _jaccard(shin_i, shingles[j]) >= threshold:
                    dup = True
                    break
        if not dup:
            kept.append((i, chunk))
        else:
            dropped += 1

    log.info(f"Dropped {dropped} duplicates, kept {len(kept)}")
    return [c for _, c in kept]


def embedding_dedup(
    chunks: list[dict],
    threshold: float = 0.92,
    embedder: Callable | None = None,
    batch_size: int = 64,
) -> list[dict]:
    """
    Embed all chunks, then do greedy dedup:
    - sort by quality score if available, else by index
    - mark a chunk as duplicate if cosine sim to any already-kept chunk >= threshold
    """
    texts = [c.get("text") or c.get("raw_text") or "" for c in chunks]

    if embedder is None:
        log.info("Loading sentence-transformers model...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embedder = lambda txts: model.encode(txts, batch_size=batch_size, show_progress_bar=True)

    t0 = time.time()
    embeddings = embedder(texts)
    embeddings = np.array(embeddings, dtype=np.float32)

    # normalize for cosine sim via dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    embeddings = embeddings / norms

    log.info(f"Embedded {len(chunks)} chunks in {time.time()-t0:.1f}s, shape={embeddings.shape}")

    # prefer chunks with higher quality scores
    quality_scores = [
        c.get("_quality", {}).get("final_score", 0.5) for c in chunks
    ]
    order = sorted(range(len(chunks)), key=lambda i: quality_scores[i], reverse=True)

    kept_indices = []
    kept_embs = []
    dropped = 0

    for idx in order:
        emb = embeddings[idx]
        if kept_embs:
            sims = np.dot(np.stack(kept_embs), emb)
            if sims.max() >= threshold:
                dropped += 1
                continue
        kept_indices.append(idx)
        kept_embs.append(emb)

    # restore original order
    kept_set = set(kept_indices)
    result = [c for i, c in enumerate(chunks) if i in kept_set]

    log.info(f"Dropped {dropped} near-duplicates, kept {len(result)}/{len(chunks)}")
    return result


def _hash_chunk(c: dict) -> str:
    text = (c.get("text") or c.get("raw_text") or "").strip()
    return hashlib.md5(text.encode()).hexdigest()


def exact_dedup(chunks: list[dict]) -> list[dict]:
    """fast pass first — exact duplicates by text hash"""
    seen = set()
    out = []
    for c in chunks:
        h = _hash_chunk(c)
        if h not in seen:
            seen.add(h)
            out.append(c)
    removed = len(chunks) - len(out)
    if removed:
        log.info(f"Exact dedup removed {removed} identical chunks")
    return out


def main():
    parser = argparse.ArgumentParser(description="Near-duplicate chunk removal")
    parser.add_argument("input", help="chunks JSON file")
    parser.add_argument("--threshold", type=float, default=0.92,
                        help="similarity threshold (higher = stricter dedup)")
    parser.add_argument("--mode", choices=["embedding", "jaccard"], default="embedding")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    p = Path(args.input)
    with p.open() as f:
        chunks = json.load(f)
    log.info(f"Loaded {len(chunks)} chunks from {p}")

    # always do exact dedup first, it's free
    chunks = exact_dedup(chunks)

    if args.mode == "jaccard":
        result = jaccard_dedup(chunks, threshold=args.threshold)
    else:
        result = embedding_dedup(chunks, threshold=args.threshold)

    out_path = args.out or str(p.with_stem(p.stem + "_deduped"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log.info(f"Saved {len(result)} chunks to {out_path}")


if __name__ == "__main__":
    main()
