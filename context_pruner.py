"""
Query-aware context pruning: drop the sentences in each retrieved chunk that don't
help answer the query, before they reach the LLM.

Reranking (cross_encoder_reranker.py) decides which *chunks* to keep. But a kept chunk
is still mostly filler — a 400-token chunk that answers the question usually does it
in one or two sentences, and the rest costs tokens, latency, and attention. Worse,
irrelevant sentences next to the answer actively distract the generator.

Provence (Chirkova et al., NAVER Labs, ICLR 2025 — https://arxiv.org/abs/2501.16214)
treats pruning as sequence labeling: a cross-encoder reads [query, chunk] and tags
each token keep/drop, then sentences are kept by majority vote. Three things make it
practical where earlier compressors (LLMLingua, RECOMP) weren't:
  - it's sentence-level, so output stays readable and quotable
  - the amount pruned is *dynamic* — 0 sentences if the chunk is irrelevant, all of
    them if every sentence matters. No fixed compression ratio to tune.
  - the same forward pass emits a rerank score, so rerank + prune costs one model call
Reported up to ~80% compression with equal-or-better answer quality.

Two backends:
  provence : naver/provence-reranker-debertav3-v1 (English) or
             naver/xprovence-reranker-bgem3-v1 (multilingual). The real thing.
  sentence : fallback with no custom model code — scores every sentence against the
             query with the cross-encoder you already have, keeps sentences above an
             adaptive threshold relative to the chunk's best sentence. Cruder (sentences
             are judged without their neighbours), but needs no trust_remote_code.

Requires: transformers (provence, trust_remote_code)  or  sentence-transformers (sentence)

Usage:
    python context_pruner.py --demo
    python context_pruner.py results.json --query "what is RRF?" --out pruned.json
    python context_pruner.py results.json --query "what is RRF?" --backend sentence

    from context_pruner import ProvencePruner
    pruner = ProvencePruner()
    pruned = pruner.prune(query, chunks)   # same chunks, shorter text, _rerank_score added
"""

import re
import json
import math
import time
import logging
import argparse
from typing import Any

log = logging.getLogger("context_pruner")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

DEFAULT_PROVENCE = "naver/provence-reranker-debertav3-v1"
DEFAULT_CROSS_ENCODER = "BAAI/bge-reranker-v2-m3"

# provence: 0.1 is conservative (paper default), 0.5 prunes hard. Start low — a dropped
# answer sentence is unrecoverable, a kept filler sentence just costs tokens.
DEFAULT_THRESHOLD = 0.1

# naive splitter: good enough for prose; swap in nltk/spacy/pysbd for abbreviations
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_sentences(text: str) -> list[str]:
    return [s for s in SENT_RE.split(text.strip()) if s.strip()]


def _logit(p: float) -> float:
    p = min(max(p, 1e-7), 1 - 1e-7)
    return math.log(p / (1 - p))


def _stats(before: list[str], after: list[str]) -> dict[str, Any]:
    b = sum(len(t) for t in before)
    a = sum(len(t) for t in after)
    return {
        "chars_before": b,
        "chars_after": a,
        "compression": round(1 - a / b, 3) if b else 0.0,
        "chunks_emptied": sum(1 for t in after if not t.strip()),
    }


class ProvencePruner:
    """Provence: one forward pass per chunk -> pruned text + rerank score."""

    def __init__(
        self,
        model_name: str = DEFAULT_PROVENCE,
        threshold: float = DEFAULT_THRESHOLD,
        always_select_title: bool = False,
        batch_size: int = 32,
    ):
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError("pip install transformers")

        log.info(f"Loading {model_name} (trust_remote_code — the pruning head is custom code)...")
        t0 = time.time()
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        log.info(f"Loaded in {time.time()-t0:.1f}s")
        self.threshold = threshold
        # True keeps each chunk's first sentence regardless — set it if your chunks
        # start with a title/heading line (docx_to_chunks.py output does)
        self.always_select_title = always_select_title
        self.batch_size = batch_size

    def prune(self, query: str, chunks: list[dict[str, Any]], text_key: str = "text") -> list[dict[str, Any]]:
        if not chunks:
            return []
        texts = [(c.get(text_key) or "") for c in chunks]

        out = self.model.process(
            query,
            [texts],
            threshold=self.threshold,
            always_select_title=self.always_select_title,
            batch_size=self.batch_size,
            reorder=False,
            enable_warnings=False,
        )
        # process() takes lists of lists (one per query); unwrap our single query
        pruned_texts = out["pruned_context"][0]
        scores = out["reranking_score"][0]

        result = []
        for chunk, pruned, score in zip(chunks, pruned_texts, scores):
            c = dict(chunk)
            c[text_key] = pruned
            c["_original_text"] = chunk.get(text_key) or ""
            c["_rerank_score"] = float(score)
            result.append(c)

        log.info(f"Pruned {len(chunks)} chunks: {_stats(texts, pruned_texts)}")
        return result


class SentencePruner:
    """
    Fallback: cross-encoder scores each sentence independently against the query.

    Keep rule is relative, not absolute — a sentence survives if its logit is within
    `margin` of the best sentence in the same chunk AND above `floor`. Relative because
    raw cross-encoder logits shift with query phrasing; the floor is what lets a fully
    irrelevant chunk prune to nothing (Provence's best property, approximated).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER,
        margin: float = 3.0,
        floor: float = -6.0,
        neighbour_window: int = 0,
        batch_size: int = 64,
    ):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError("pip install sentence-transformers")

        log.info(f"Loading cross-encoder {model_name}...")
        self.model = CrossEncoder(model_name, max_length=256)
        self.margin = margin
        self.floor = floor
        # also keep N sentences either side of each kept one — cheap fix for the
        # "kept the answer, dropped the sentence that says what 'it' refers to" problem
        self.neighbour_window = neighbour_window
        self.batch_size = batch_size

    def prune(self, query: str, chunks: list[dict[str, Any]], text_key: str = "text") -> list[dict[str, Any]]:
        if not chunks:
            return []
        texts = [(c.get(text_key) or "") for c in chunks]
        sents = [split_sentences(t) for t in texts]

        # one batched predict over every sentence of every chunk
        flat = [s for ss in sents for s in ss]
        if not flat:
            return [dict(c) for c in chunks]
        all_scores = self.model.predict(
            [[query, s] for s in flat], batch_size=self.batch_size, show_progress_bar=False
        )
        # sentence-transformers >= 3 sigmoids single-label rerankers into (0, 1), where
        # every irrelevant sentence is ~0.000 and a margin is meaningless. Go back to
        # logits so margin/floor work the same whatever version is installed.
        if all(0.0 <= x <= 1.0 for x in all_scores):
            all_scores = [_logit(float(x)) for x in all_scores]

        result, pruned_texts, pos = [], [], 0
        for chunk, ss in zip(chunks, sents):
            scores = [float(x) for x in all_scores[pos : pos + len(ss)]]
            pos += len(ss)

            best = max(scores, default=self.floor - 1)
            keep = {i for i, s in enumerate(scores) if s >= self.floor and s >= best - self.margin}
            for i in list(keep):
                for j in range(i - self.neighbour_window, i + self.neighbour_window + 1):
                    if 0 <= j < len(ss):
                        keep.add(j)

            pruned = " ".join(ss[i] for i in sorted(keep))
            pruned_texts.append(pruned)

            c = dict(chunk)
            c[text_key] = pruned
            c["_original_text"] = chunk.get(text_key) or ""
            # chunk relevance = its best sentence; usable as a rerank score
            c["_rerank_score"] = best
            result.append(c)

        log.info(f"Pruned {len(chunks)} chunks: {_stats(texts, pruned_texts)}")
        return result


def drop_empty(chunks: list[dict[str, Any]], text_key: str = "text") -> list[dict[str, Any]]:
    """Chunks pruned to nothing were irrelevant — drop them before context packing."""
    return [c for c in chunks if (c.get(text_key) or "").strip()]


def _demo(backend: str):
    query = "How does reciprocal rank fusion combine ranked lists?"
    chunks = [
        {
            "chunk_id": "c1",
            "text": (
                "Hybrid search has become the default in production RAG systems. "
                "Reciprocal Rank Fusion scores each document by summing 1/(k + rank) across "
                "every ranked list it appears in, with k usually set to 60. "
                "Because it only uses ranks, RRF never has to normalize BM25 and cosine scores "
                "onto the same scale. "
                "The technique was published by Cormack, Clarke and Buettcher in 2009. "
                "Many vector databases now ship it as a built-in option."
            ),
        },
        {
            "chunk_id": "c2",
            "text": (
                "Vector databases store embeddings in approximate nearest neighbour indexes. "
                "HNSW builds a layered proximity graph and is the most common choice. "
                "IVF partitions the space into clusters and only searches the closest ones."
            ),
        },
        {
            "chunk_id": "c3",
            "text": (
                "Our team migrated to a new cluster in March. "
                "After the migration, fusion of the keyword and dense result lists was done "
                "with RRF instead of a weighted sum of scores, which removed a fragile tuning step. "
                "Latency stayed roughly the same."
            ),
        },
    ]

    pruner = ProvencePruner() if backend == "provence" else SentencePruner()
    pruned = pruner.prune(query, chunks)

    print(f"\nQuery: {query!r}\n")
    for c in pruned:
        print(f"--- {c['chunk_id']}  score={c['_rerank_score']:.3f}")
        print(f"  before ({len(c['_original_text'])} chars): {c['_original_text'][:100]}...")
        print(f"  after  ({len(c['text'])} chars): {c['text'] or '(pruned to nothing — chunk dropped)'}")
    s = _stats([c["_original_text"] for c in pruned], [c["text"] for c in pruned])
    print(f"\ncompression: {s['compression']:.0%}, chunks emptied: {s['chunks_emptied']}")


def main():
    parser = argparse.ArgumentParser(description="Query-aware sentence-level context pruning (Provence)")
    parser.add_argument("chunks", nargs="?", help="retrieved/reranked chunks JSON")
    parser.add_argument("--query", help="the user query the chunks were retrieved for")
    parser.add_argument("--out", default="pruned.json")
    parser.add_argument("--backend", choices=["provence", "sentence"], default="provence")
    parser.add_argument("--model", help="override model name for the chosen backend")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="provence keep threshold")
    parser.add_argument("--keep-empty", action="store_true", help="keep chunks pruned to nothing")
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--demo", action="store_true", help="run demo with synthetic data")
    args = parser.parse_args()

    if args.demo:
        _demo(args.backend)
        return

    if not (args.chunks and args.query):
        parser.print_help()
        return

    with open(args.chunks) as f:
        chunks = json.load(f)
    log.info(f"Loaded {len(chunks)} chunks from {args.chunks}")

    if args.backend == "provence":
        pruner = ProvencePruner(args.model or DEFAULT_PROVENCE, threshold=args.threshold)
    else:
        pruner = SentencePruner(args.model or DEFAULT_CROSS_ENCODER)

    pruned = pruner.prune(args.query, chunks, text_key=args.text_key)
    if not args.keep_empty:
        pruned = drop_empty(pruned, args.text_key)

    with open(args.out, "w") as f:
        json.dump(pruned, f, indent=2, ensure_ascii=False)
    log.info(f"Wrote {len(pruned)} chunks to {args.out}")


if __name__ == "__main__":
    main()
