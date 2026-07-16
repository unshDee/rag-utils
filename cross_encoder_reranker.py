"""
Reranking stage: sits between retrieval and context packing.

Bi-encoders (what your vector DB uses) embed query and document separately, so the
model compresses a document into one vector before knowing what will be asked of it.
A cross-encoder concatenates [query, doc] and runs attention across both, then emits
one relevance score. Much more accurate, way too slow to run over a corpus — so
retrieve top-100 cheaply, rerank down to top-5 here.

Three tools:
  CrossEncoderReranker : local model (bge-reranker-v2-m3). Free, fast, no network.
  LLMReranker          : listwise rerank via Claude — reads the whole candidate list
                         and returns a permutation. RankGPT-style, Sun et al. 2023
                         (https://arxiv.org/abs/2304.09542). Slower, costs money,
                         catches intent a 400M-param cross-encoder misses.
  mmr()                : diversity filter, not a reranker. Trades relevance against
                         redundancy so top-k isn't 5 paraphrases of one paragraph.
                         Carbonell & Goldstein 1998.

Requires: numpy  sentence-transformers (cross-encoder)  anthropic (LLM rerank)

Usage:
    python cross_encoder_reranker.py --demo
    python cross_encoder_reranker.py results.json --query "what is RRF?" --top-k 5

    from cross_encoder_reranker import CrossEncoderReranker, mmr
    reranker = CrossEncoderReranker()
    top = reranker.rerank(query, candidates, top_k=5)
"""

import os
import json
import time
import logging
import argparse
from typing import Any

import numpy as np

log = logging.getLogger("reranker")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# bge-reranker-v2-m3: strongest general-purpose open reranker (multilingual, 8k ctx).
# Swap to cross-encoder/ms-marco-MiniLM-L-6-v2 for ~10x speed, English only.
DEFAULT_CROSS_ENCODER = "BAAI/bge-reranker-v2-m3"
DEFAULT_LLM = "claude-haiku-4-5"


class CrossEncoderReranker:
    """Local cross-encoder. Scores every (query, chunk) pair jointly."""

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
    ):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError("pip install sentence-transformers")

        log.info(f"Loading cross-encoder {model_name}...")
        t0 = time.time()
        self.model = CrossEncoder(model_name, device=device, max_length=max_length)
        self.batch_size = batch_size
        log.info(f"Loaded in {time.time()-t0:.1f}s")

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        pairs = [[query, t] for t in texts]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return np.asarray(scores, dtype=np.float32)

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
        text_key: str = "text",
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Returns top_k candidates by cross-encoder score, each with a _rerank_score field.

        min_score: absolute cutoff, so the pipeline can return nothing when the corpus
        has no answer instead of handing the LLM 5 irrelevant chunks. Calibrate on your
        own data — bge-reranker logits are roughly centered on 0.
        """
        if not candidates:
            return []

        # each pair is a full forward pass, so keep candidates to ~50-100
        texts = [(c.get(text_key) or "") for c in candidates]
        scores = self.score(query, texts)

        order = np.argsort(scores)[::-1]
        out = []
        for i in order[:top_k]:
            if min_score is not None and scores[i] < min_score:
                break
            chunk = dict(candidates[i])
            chunk["_rerank_score"] = float(scores[i])
            out.append(chunk)

        return out


class LLMReranker:
    """
    Listwise reranking with Claude (RankGPT-style).

    The model sees the whole candidate list at once and returns an ordering, so it
    can reason about relative relevance — "passage 3 answers it, passage 7 only
    mentions the topic" — which pointwise scorers can't.
    """

    def __init__(self, model: str = DEFAULT_LLM, window: int = 20, stride: int = 10):
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY not set in environment")

        self.client = anthropic.Anthropic()
        self.model = model
        self.window = window
        self.stride = stride

    def _rank_window(self, query: str, texts: list[str]) -> list[int]:
        """rank one window, returns local indices best-first"""
        listing = "\n\n".join(f"[{i}] {t[:1200]}" for i, t in enumerate(texts))
        prompt = (
            f"Rank the passages below by how well each one answers the query.\n\n"
            f"Query: {query}\n\n"
            f"{listing}\n\n"
            f"Output ONLY a JSON array of passage numbers, most relevant first, "
            f"including every number exactly once. Example: [3, 0, 5, 1, 2, 4]"
        )

        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = "".join(b.text for b in resp.content if b.type == "text").strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            order = json.loads(raw)

            # the model can drop, repeat, or invent indices — repair instead of trusting
            seen, clean = set(), []
            for i in order:
                if isinstance(i, int) and 0 <= i < len(texts) and i not in seen:
                    seen.add(i)
                    clean.append(i)
            clean += [i for i in range(len(texts)) if i not in seen]
            return clean

        except Exception as e:
            log.warning(f"LLM rerank failed ({e}), keeping original order")
            return list(range(len(texts)))

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
        text_key: str = "text",
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        ranked = list(candidates)

        if len(ranked) <= self.window:
            order = self._rank_window(query, [(c.get(text_key) or "") for c in ranked])
            ranked = [ranked[i] for i in order]
        else:
            # slide bottom -> top so a promoted doc keeps rising through each window
            end = len(ranked)
            while end > 0:
                start = max(0, end - self.window)
                window = ranked[start:end]
                order = self._rank_window(query, [(c.get(text_key) or "") for c in window])
                ranked[start:end] = [window[i] for i in order]
                if start == 0:
                    break
                end -= self.stride

        out = []
        for rank, chunk in enumerate(ranked[:top_k]):
            c = dict(chunk)
            c["_rerank_score"] = round(1.0 / (rank + 1), 4)  # reciprocal rank as score
            out.append(c)
        return out


def mmr(
    query_emb: np.ndarray,
    doc_embs: np.ndarray,
    top_k: int = 5,
    lambda_mult: float = 0.7,
) -> list[int]:
    """
    Maximal Marginal Relevance. Returns selected indices in selection order.

      MMR = argmax_d [ lambda * sim(d, q) - (1-lambda) * max_{s in selected} sim(d, s) ]

    lambda_mult=1.0 is pure relevance, 0.0 is pure diversity; 0.5-0.7 is the useful
    range. Worth it when the corpus has near-duplicate passages — the LLM gains nothing
    from redundancy and every dupe steals a slot from a chunk covering something else.
    """
    if len(doc_embs) == 0:
        return []

    def _norm(x):
        return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)

    q = _norm(np.asarray(query_emb, dtype=np.float32))
    d = _norm(np.asarray(doc_embs, dtype=np.float32))

    query_sim = d @ q
    doc_sim = d @ d.T

    selected: list[int] = []
    remaining = set(range(len(d)))

    while len(selected) < min(top_k, len(d)):
        best_i, best_score = None, -np.inf
        for i in remaining:
            redundancy = max((doc_sim[i][j] for j in selected), default=0.0)
            score = lambda_mult * query_sim[i] - (1 - lambda_mult) * redundancy
            if score > best_score:
                best_i, best_score = i, score
        selected.append(best_i)
        remaining.remove(best_i)

    return selected


def _demo():
    chunks = [
        {"chunk_id": "c1", "text": "Reciprocal Rank Fusion merges ranked lists from multiple retrievers by summing 1/(k+rank). It needs no score normalization."},
        {"chunk_id": "c2", "text": "RRF combines several ranked result lists using rank-based votes rather than raw scores, which avoids score calibration."},
        {"chunk_id": "c3", "text": "BM25 is a sparse lexical ranking function based on term frequency and inverse document frequency."},
        {"chunk_id": "c4", "text": "The context window of a language model bounds how many tokens can be processed in a single request."},
        {"chunk_id": "c5", "text": "Cross-encoders jointly encode the query and document, giving much better relevance estimates than bi-encoders."},
        {"chunk_id": "c6", "text": "Vector databases such as FAISS and Qdrant store dense embeddings and support approximate nearest neighbour search."},
    ]
    query = "how does reciprocal rank fusion work?"

    print(f"Query: {query!r}\n")
    try:
        reranker = CrossEncoderReranker()
        for r in reranker.rerank(query, chunks, top_k=3):
            print(f"  [{r['_rerank_score']:+.3f}] {r['chunk_id']}: {r['text'][:70]}...")
    except ImportError as e:
        print(f"  (skipping cross-encoder: {e})")

    # c1 and c2 say the same thing — MMR should keep only one
    print("\nMMR diversity (random embeddings, illustrative only):")
    rng = np.random.default_rng(0)
    q_emb = rng.normal(size=384)
    d_embs = rng.normal(size=(len(chunks), 384))
    d_embs[1] = d_embs[0] + 0.01 * rng.normal(size=384)  # make c2 a near-dup of c1
    for i in mmr(q_emb, d_embs, top_k=3, lambda_mult=0.6):
        print(f"  {chunks[i]['chunk_id']}: {chunks[i]['text'][:70]}...")


def main():
    parser = argparse.ArgumentParser(description="Cross-encoder / LLM reranking")
    parser.add_argument("candidates", nargs="?", help="retrieved chunks JSON")
    parser.add_argument("--query", help="query string")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", choices=["cross", "llm"], default="cross")
    parser.add_argument("--model", help="cross-encoder or Claude model id")
    parser.add_argument("--min-score", type=float, help="drop candidates below this score")
    parser.add_argument("--out", default=None)
    parser.add_argument("--demo", action="store_true", help="run demo with synthetic data")
    args = parser.parse_args()

    if args.demo:
        _demo()
        return

    if not (args.candidates and args.query):
        parser.print_help()
        return

    with open(args.candidates) as f:
        candidates = json.load(f)
    log.info(f"Loaded {len(candidates)} candidates from {args.candidates}")

    if args.mode == "cross":
        reranker = CrossEncoderReranker(args.model or DEFAULT_CROSS_ENCODER)
        results = reranker.rerank(args.query, candidates, top_k=args.top_k, min_score=args.min_score)
    else:
        results = LLMReranker(args.model or DEFAULT_LLM).rerank(
            args.query, candidates, top_k=args.top_k
        )

    for r in results:
        print(f"[{r['_rerank_score']:+.4f}] {r.get('chunk_id', '?')}: {(r.get('text') or '')[:110]}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        log.info(f"Saved {len(results)} chunks to {args.out}")


if __name__ == "__main__":
    main()
