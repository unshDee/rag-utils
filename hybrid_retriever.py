"""
Hybrid retrieval: BM25 (sparse) + dense vector search, fused with
Reciprocal Rank Fusion (RRF).

RRF was introduced by Cormack et al. 2009 and is surprisingly hard to beat
for combining ranked lists without needing to tune score scales.

  RRF_score(d) = sum_r [ 1 / (k + rank_r(d)) ]

where k=60 is the standard constant (empirically best on TREC benchmarks).

Requires: rank-bm25  numpy  (faiss-cpu for vector search, optional)

Usage:
    # index some chunks and query interactively
    python hybrid_retriever.py --demo

    # or import and use in your pipeline:
    from hybrid_retriever import HybridRetriever
    retriever = HybridRetriever(chunks)
    results = retriever.search("what is RAG?", top_k=5)
"""

import math
import json
import time
import logging
import argparse
from typing import Any, Callable

import numpy as np

log = logging.getLogger("hybrid_retriever")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

RRF_K = 60  # standard constant, rarely worth tuning


def rrf_fuse(
    ranked_lists: list[list[str]],
    k: int = RRF_K,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """
    Fuse multiple ranked lists via RRF.
    Returns [(doc_id, rrf_score), ...] sorted by score descending.
    weights: per-list multipliers (default: equal). len must match ranked_lists.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    assert len(weights) == len(ranked_lists)

    scores: dict[str, float] = {}
    for ranked, w in zip(ranked_lists, weights):
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class BM25Index:
    def __init__(self, corpus: list[str]):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("pip install rank-bm25")

        tokenized = [self._tokenize(doc) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized)
        self.corpus_size = len(corpus)

    def _tokenize(self, text: str) -> list[str]:
        import re
        return re.findall(r"\b[a-z']+\b", text.lower())

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        q_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        # get indices sorted by score
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]


class DenseIndex:
    """FAISS-backed dense retrieval. Falls back to numpy brute-force if faiss unavailable."""

    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings.astype(np.float32)
        # normalize for cosine sim
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / np.where(norms == 0, 1e-9, norms)

        self._faiss_index = None
        try:
            import faiss
            dim = self.embeddings.shape[1]
            idx = faiss.IndexFlatIP(dim)  # inner product = cosine after normalization
            idx.add(self.embeddings)
            self._faiss_index = idx
            log.debug("Using FAISS index")
        except ImportError:
            log.debug("faiss not installed, falling back to numpy brute-force")

    def search(self, query_emb: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        q = query_emb.astype(np.float32)
        q = q / max(np.linalg.norm(q), 1e-9)

        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(q.reshape(1, -1), top_k)
            return [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]

        # numpy fallback
        sims = self.embeddings @ q
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(int(i), float(sims[i])) for i in top_indices]


class HybridRetriever:
    def __init__(
        self,
        chunks: list[dict[str, Any]],
        embedder: Callable | None = None,
        text_key: str = "text",
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
    ):
        self.chunks = chunks
        self.text_key = text_key
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self._embedder = embedder
        self._dense_index: DenseIndex | None = None

        texts = [c.get(text_key) or "" for c in chunks]
        log.info(f"Building BM25 index over {len(texts)} chunks...")
        self.bm25 = BM25Index(texts)

        if embedder is not None:
            self._build_dense(texts, embedder)

    def _build_dense(self, texts: list[str], embedder: Callable):
        log.info("Embedding chunks for dense index...")
        t0 = time.time()
        embs = np.array(embedder(texts), dtype=np.float32)
        log.info(f"Embedded in {time.time()-t0:.1f}s")
        self._dense_index = DenseIndex(embs)

    def search(
        self,
        query: str,
        top_k: int = 10,
        bm25_candidates: int = 50,
        dense_candidates: int = 50,
    ) -> list[dict[str, Any]]:
        bm25_results = self.bm25.search(query, top_k=bm25_candidates)
        bm25_ranked = [str(i) for i, _ in bm25_results]

        ranked_lists = [bm25_ranked]
        weights = [self.bm25_weight]

        if self._dense_index is not None and self._embedder is not None:
            q_emb = np.array(self._embedder([query])[0])
            dense_results = self._dense_index.search(q_emb, top_k=dense_candidates)
            dense_ranked = [str(i) for i, _ in dense_results]
            ranked_lists.append(dense_ranked)
            weights.append(self.dense_weight)

        fused = rrf_fuse(ranked_lists, weights=weights)

        results = []
        for idx_str, rrf_score in fused[:top_k]:
            idx = int(idx_str)
            chunk = dict(self.chunks[idx])
            chunk["_rrf_score"] = round(rrf_score, 6)
            results.append(chunk)

        return results


def _demo():
    """run a small demo with synthetic chunks"""
    sample_chunks = [
        {"chunk_id": "c1", "text": "Retrieval augmented generation combines search with LLMs to answer questions from documents."},
        {"chunk_id": "c2", "text": "BM25 is a bag-of-words ranking function used in information retrieval."},
        {"chunk_id": "c3", "text": "Dense retrieval uses neural embeddings to find semantically similar documents."},
        {"chunk_id": "c4", "text": "Reciprocal Rank Fusion merges multiple ranked lists without score normalization."},
        {"chunk_id": "c5", "text": "Chunking splits documents into smaller pieces for embedding and retrieval."},
        {"chunk_id": "c6", "text": "Vector databases like FAISS, Pinecone, and Weaviate store embeddings efficiently."},
        {"chunk_id": "c7", "text": "Hybrid search outperforms pure sparse or pure dense retrieval on most benchmarks."},
        {"chunk_id": "c8", "text": "The context window of a language model limits how much text can be processed at once."},
    ]

    # demo without dense (BM25 only since no embedder)
    retriever = HybridRetriever(sample_chunks)

    queries = [
        "how does RAG work?",
        "combining ranked lists",
        "neural search embeddings",
    ]

    for q in queries:
        print(f"\nQuery: {q!r}")
        results = retriever.search(q, top_k=3)
        for r in results:
            print(f"  [{r['_rrf_score']:.4f}] {r['chunk_id']}: {r['text'][:80]}")


def main():
    parser = argparse.ArgumentParser(description="Hybrid BM25 + dense retrieval with RRF")
    parser.add_argument("--demo", action="store_true", help="run demo with synthetic data")
    parser.add_argument("--index", help="chunks JSON file to index")
    parser.add_argument("--query", help="query string")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    if args.demo:
        _demo()
        return

    if args.index and args.query:
        with open(args.index) as f:
            chunks = json.load(f)
        retriever = HybridRetriever(chunks)
        results = retriever.search(args.query, top_k=args.top_k)
        for r in results:
            print(f"[{r['_rrf_score']:.4f}] {r.get('chunk_id', '?')}: {(r.get('text') or '')[:120]}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
