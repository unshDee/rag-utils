"""
Small-to-big expansion: retrieve on small chunks, hand the LLM the text around them.

Chunk size is a tradeoff you can't win with one number. Small chunks (a sentence, a
short paragraph) embed one idea each, so they match queries precisely. Big chunks keep
the context needed to actually answer, but their embedding is an average of several
topics and they rank worse. Dense X Retrieval (Chen et al. 2023,
https://arxiv.org/abs/2312.06648) showed retrieving at a finer granularity (propositions)
beats passages on recall. The catch nobody mentions: the retrieved sentence is often
"It gives each document 1/(k + rank)...", and the generator has no idea what "it" is.

So decouple the two. Index small, then at query time, after rerank and before packing,
expand each hit back into a bigger span from the source document:
  - window   : grow each hit by N neighbours either side (LlamaIndex calls this
               sentence-window retrieval). capped by a token budget per span.
  - merge    : windows from the same doc that touch or overlap become one span, so two
               hits on adjacent sentences don't put the same text in the prompt twice.
  - auto-merge: if enough children of one parent section got hit, swap in the whole
               parent (LlamaIndex AutoMergingRetriever, LangChain ParentDocumentRetriever).
               parents are either fixed runs of N children or runs of the same value of
               a key, e.g. `headings` from docx_to_chunks.py.

Output spans keep `chunk_id` / `text` / `score`, so they go straight into
context_window_packer.py. score is the best hit in the span, n_hits is there if you want
to boost spans that several hits agreed on.

Requires: stdlib only

Usage:
    python small_to_big.py --demo
    python small_to_big.py reranked.json --chunks chunks.json --window 2 --out expanded.json
    python small_to_big.py reranked.json --chunks chunks.json --parent-key headings --merge-ratio 0.5

    from small_to_big import SmallToBig
    s2b = SmallToBig(corpus_chunks, window=1, max_tokens=400)
    spans = s2b.expand(hits)   # hits: [{"chunk_id": ..., "score": ...}, ...]
"""

import re
import json
import math
import logging
import argparse
from collections import Counter, defaultdict
from typing import Any

log = logging.getLogger("small_to_big")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# 1 neighbour each side fixes most dangling "it"/"this" sentences. go to 2-3 if your
# children are single sentences and answers tend to span a few of them
DEFAULT_WINDOW = 1

# cap per window, so a hit inside a run of huge paragraphs doesn't eat the whole budget.
# merged spans can end up bigger than this, that's fine, the text isn't duplicated
DEFAULT_MAX_TOKENS = 400

# LlamaIndex default. half the children of a section hit means the section is the answer,
# lower it and you start pulling whole sections in on two lucky matches
DEFAULT_MERGE_RATIO = 0.5

# a parent bigger than this never auto-merges, a 6k token section is not "context"
DEFAULT_MAX_PARENT_TOKENS = 1200

# first one that exists on the chunks decides which doc a chunk belongs to
DOC_KEYS = ("doc_id", "doc_name", "source", "source_path")

JOINER = "\n"


def estimate_tokens(text: str) -> int:
    # same ~4 chars per token heuristic as context_window_packer.py
    return max(1, len(text) // 4)


class SmallToBig:
    """
    corpus: every small chunk you indexed, not just the retrieved ones, since we need the
    neighbours. order within a doc is `chunk_index` if present, else file order.

    if your chunker used overlapping windows, run chunk_overlap_merger.py first or the
    joined spans will repeat the overlap text.
    """

    def __init__(
        self,
        corpus: list[dict[str, Any]],
        window: int = DEFAULT_WINDOW,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        parent_key: str | None = None,
        parent_size: int = 0,
        merge_ratio: float = DEFAULT_MERGE_RATIO,
        max_parent_tokens: int = DEFAULT_MAX_PARENT_TOKENS,
        text_key: str = "text",
        joiner: str = JOINER,
    ):
        self.window = window
        self.max_tokens = max_tokens
        self.merge_ratio = merge_ratio
        self.max_parent_tokens = max_parent_tokens
        self.text_key = text_key
        self.joiner = joiner
        # with a real section key, a window shouldn't spill into the next section. fixed
        # size parents are arbitrary cuts, so windows are allowed to cross them
        self.clip_to_parent = parent_key is not None

        doc_key = next((k for k in DOC_KEYS if any(k in c for c in corpus)), None)
        by_doc: dict[str, list[dict]] = defaultdict(list)
        for c in corpus:
            by_doc[str(c.get(doc_key, "_")) if doc_key else "_"].append(c)

        self.docs: dict[str, list[dict]] = {}
        self.pos: dict[str, tuple[str, int]] = {}
        self.tokens: dict[str, list[int]] = {}
        # parent_of[doc][i] = (start, end) of the parent range child i sits in, inclusive
        self.parent_of: dict[str, list[tuple[int, int]]] = {}

        for doc, cs in by_doc.items():
            if any("chunk_index" in c for c in cs):
                cs = sorted(cs, key=lambda c: c.get("chunk_index", 0))  # stable, keeps file order on ties
            self.docs[doc] = cs
            self.tokens[doc] = [estimate_tokens(c.get(text_key) or "") for c in cs]
            for i, c in enumerate(cs):
                self.pos[str(c["chunk_id"])] = (doc, i)
            self.parent_of[doc] = self._parents(cs, parent_key, parent_size)

        log.info(f"Indexed {len(self.pos)} children across {len(self.docs)} docs")

    @staticmethod
    def _parents(cs: list[dict], key: str | None, size: int) -> list[tuple[int, int]]:
        n = len(cs)
        if key:
            # consecutive runs of the same value. str() so list-valued keys like headings work
            labels = [str(c.get(key)) for c in cs]
            out, start = [], 0
            for i in range(1, n + 1):
                if i == n or labels[i] != labels[start]:
                    out += [(start, i - 1)] * (i - start)
                    start = i
            return out
        if size > 1:
            return [(i // size * size, min(n, i // size * size + size) - 1) for i in range(n)]
        return [(i, i) for i in range(n)]  # no parents, auto-merge is off

    def _grow(self, doc: str, i: int) -> tuple[int, int]:
        lo_bound, hi_bound = self.parent_of[doc][i] if self.clip_to_parent else (0, len(self.docs[doc]) - 1)
        toks = self.tokens[doc]
        lo = hi = i
        used = toks[i]
        for _ in range(self.window):
            grew = False
            # before first: the previous sentence is usually what "it" points at
            for j, side in ((lo - 1, "lo"), (hi + 1, "hi")):
                if lo_bound <= j <= hi_bound and used + toks[j] <= self.max_tokens:
                    used += toks[j]
                    lo, hi = (j, hi) if side == "lo" else (lo, j)
                    grew = True
            if not grew:
                break
        return lo, hi

    def expand(self, hits: list[dict[str, Any]], score_key: str = "score") -> list[dict[str, Any]]:
        best: dict[str, float] = {}
        for h in hits:
            cid = str(h["chunk_id"])
            if cid not in self.pos:
                log.warning(f"hit {cid} not in corpus, skipping (corpus and index out of sync?)")
                continue
            s = float(h.get(score_key, 0.0))
            best[cid] = max(best.get(cid, -math.inf), s)

        hits_by_doc: dict[str, list[int]] = defaultdict(list)
        for cid in best:
            doc, i = self.pos[cid]
            hits_by_doc[doc].append(i)

        spans = []
        for doc, idxs in hits_by_doc.items():
            ranges: list[tuple[int, int, bool]] = []

            # auto-merge first, a hit inside a merged parent doesn't need its own window
            parent_hits = Counter(self.parent_of[doc][i] for i in idxs)
            merged_parents = set()
            for (a, b), k in parent_hits.items():
                size = b - a + 1
                if size > 1 and k / size >= self.merge_ratio and sum(self.tokens[doc][a:b + 1]) <= self.max_parent_tokens:
                    merged_parents.add((a, b))
                    ranges.append((a, b, True))
            for i in idxs:
                if self.parent_of[doc][i] not in merged_parents:
                    ranges.append((*self._grow(doc, i), False))

            # merge ranges that overlap or touch, adjacent text reads as one passage anyway
            ranges.sort()
            merged: list[list] = []
            for a, b, p in ranges:
                if merged and a <= merged[-1][1] + 1:
                    merged[-1][1] = max(merged[-1][1], b)
                    merged[-1][2] = merged[-1][2] or p
                else:
                    merged.append([a, b, p])

            cs = self.docs[doc]
            hit_set = set(idxs)
            for a, b, p in merged:
                kids = cs[a:b + 1]
                hit_ids = [str(cs[i]["chunk_id"]) for i in range(a, b + 1) if i in hit_set]
                text = self.joiner.join(c.get(self.text_key) or "" for c in kids)
                spans.append({
                    "chunk_id": f"{doc}#{a}-{b}",
                    "text": text,
                    "score": max(best[h] for h in hit_ids),
                    "doc": doc,
                    "child_ids": [str(c["chunk_id"]) for c in kids],
                    "hit_ids": hit_ids,
                    "n_hits": len(hit_ids),
                    "auto_merged": p,
                    "n_tokens_est": estimate_tokens(text),
                })

        spans.sort(key=lambda s: -s["score"])
        log.info(
            f"{len(best)} hits -> {len(spans)} spans "
            f"({sum(s['auto_merged'] for s in spans)} auto-merged), "
            f"~{sum(s['n_tokens_est'] for s in spans)} tokens"
        )
        return spans


# --- demo ------------------------------------------------------------------------

_DEMO_DOCS = {
    "rrf": [
        "Hybrid search runs a keyword retriever and a dense retriever side by side.",
        "Their scores live on different scales, so adding them directly is fragile.",
        "Reciprocal Rank Fusion sidesteps this by ignoring the raw scores.",
        "It gives each document 1/(k + rank) from every list and sums the votes.",
        "The constant k is usually 60, which damps the influence of the very top ranks.",
        "Cormack, Clarke and Buettcher introduced the method at SIGIR 2009.",
        "Most vector databases now ship it as a built in option.",
        "Weighted score fusion can beat it, but only after careful per corpus tuning.",
    ],
    "hnsw": [
        "HNSW is the most common approximate nearest neighbour index in vector databases.",
        "It builds a layered proximity graph over the embeddings.",
        "Search starts at the sparse top layer and greedily descends to denser layers.",
        "The ef_search parameter controls how many candidates are kept during that descent.",
        "Raising it improves recall at the cost of latency.",
        "Memory is the main downside of HNSW, since every vector and its edges stay in RAM.",
        "Product quantization is the usual fix once the index stops fitting.",
        "It compresses each vector into a handful of byte codes.",
    ],
    "chunking": [
        "Chunk size is the first knob most teams tune in a RAG pipeline.",
        "Small chunks match queries precisely because one chunk carries one idea.",
        "But a small chunk often lacks the context needed to answer the question.",
        "Large chunks keep that context, yet their embeddings blur several topics together.",
        "Small-to-big retrieval splits the difference.",
        "It searches over small chunks and hands the generator their surrounding text.",
        "Parent document retrieval is the same idea with fixed parent sections.",
        "Auto-merging swaps in the whole parent when enough of its children match.",
    ],
}

# (query, children the answer actually needs). in each one the keywords sit in one
# sentence and the answer is in its neighbour, which is the failure this script is for
_DEMO_QUERIES = [
    ("how does reciprocal rank fusion work", ["rrf_2", "rrf_3"]),
    ("what does the ef_search parameter trade off", ["hnsw_3", "hnsw_4"]),
    ("how do you fix hnsw memory", ["hnsw_5", "hnsw_6"]),
    ("what is small-to-big retrieval", ["chunking_4", "chunking_5"]),
    ("why do large chunks retrieve worse", ["chunking_2", "chunking_3"]),
]

_STOP = set("a an the is are of to in and or how what why do does you it its by on at for with".split())


def _terms(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9_]+", text.lower()) if t not in _STOP]


def _lexical_search(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """tiny idf-weighted overlap scorer, stand-in for your real retriever"""
    df = Counter(t for c in chunks for t in set(_terms(c["text"])))
    n = len(chunks)
    q = set(_terms(query))
    scored = []
    for c in chunks:
        ts = _terms(c["text"])
        s = sum(math.log(1 + n / df[t]) for t in q & set(ts)) / math.sqrt(len(ts) or 1)
        scored.append({"chunk_id": c["chunk_id"], "text": c["text"], "score": round(s, 4)})
    return sorted(scored, key=lambda x: -x["score"])[:top_k]


def demo() -> None:
    small = [
        {"chunk_id": f"{d}_{i}", "doc_id": d, "chunk_index": i, "text": s}
        for d, ss in _DEMO_DOCS.items() for i, s in enumerate(ss)
    ]
    # the "just use bigger chunks" baseline: 4 sentences each
    big = []
    for d, ss in _DEMO_DOCS.items():
        for j in range(0, len(ss), 4):
            big.append({"chunk_id": f"{d}_big{j}", "text": " ".join(ss[j:j + 4]),
                        "children": {f"{d}_{i}" for i in range(j, min(j + 4, len(ss)))}})
    big_children = {b["chunk_id"]: b["children"] for b in big}

    s2b = SmallToBig(small, window=1, max_tokens=120)
    top_k = 2
    rows = {"small top-2": [], "big(4) top-2": [], "small-to-big": []}
    toks = {k: 0 for k in rows}

    logging.getLogger("small_to_big").setLevel(logging.WARNING)
    print(f"\n{'query':45s} {'small':>6s} {'big':>6s} {'s2b':>6s}")
    for q, need in _DEMO_QUERIES:
        need = set(need)
        hits = _lexical_search(q, small, top_k)
        got_small = {h["chunk_id"] for h in hits}
        bhits = _lexical_search(q, big, top_k)
        got_big = set().union(*(big_children[h["chunk_id"]] for h in bhits))
        spans = s2b.expand(hits)
        got_s2b = {c for s in spans for c in s["child_ids"]}

        cov = [len(need & g) / len(need) for g in (got_small, got_big, got_s2b)]
        for (k, v) in zip(rows, cov):
            rows[k].append(v)
        toks["small top-2"] += sum(estimate_tokens(h["text"]) for h in hits)
        toks["big(4) top-2"] += sum(estimate_tokens(h["text"]) for h in bhits)
        toks["small-to-big"] += sum(s["n_tokens_est"] for s in spans)
        print(f"{q:45s} {cov[0]:6.2f} {cov[1]:6.2f} {cov[2]:6.2f}")

    n = len(_DEMO_QUERIES)
    print(f"\n{'':22s} {'answer coverage':>16s} {'tokens/query':>13s}")
    for k in rows:
        print(f"{k:22s} {sum(rows[k]) / n:16.2f} {toks[k] / n:13.1f}")

    # merge + auto-merge behaviour on hand-picked hits
    s2b_p = SmallToBig(small, window=1, parent_size=4, merge_ratio=0.5)
    spans = s2b_p.expand([{"chunk_id": "rrf_0", "score": 0.9}, {"chunk_id": "rrf_2", "score": 0.5},
                          {"chunk_id": "rrf_5", "score": 0.7}, {"chunk_id": "rrf_7", "score": 0.4}])
    print("\nauto-merge (parents of 4, ratio 0.5), hits rrf_0 rrf_2 rrf_5 rrf_7:")
    for s in spans:
        print(f"  {s['chunk_id']:10s} score={s['score']}  hits={s['hit_ids']}  auto_merged={s['auto_merged']}")

    # self-checks
    avg = {k: sum(v) / n for k, v in rows.items()}
    assert avg["small-to-big"] == 1.0, "window=1 should cover every adjacent-sentence answer"
    assert avg["small-to-big"] > avg["small top-2"]
    assert toks["small-to-big"] < toks["big(4) top-2"], "s2b should be cheaper than big chunks here"
    # both parents of rrf have 2/4 hits, so both auto-merge, and they touch, so one span
    assert len(spans) == 1 and spans[0]["chunk_id"] == "rrf#0-7" and spans[0]["score"] == 0.9
    # adjacent windows merge instead of duplicating text
    sp = s2b.expand([{"chunk_id": "hnsw_3", "score": 1}, {"chunk_id": "hnsw_5", "score": 0.5}])
    assert len(sp) == 1 and sp[0]["child_ids"] == [f"hnsw_{i}" for i in range(2, 7)]
    # token cap stops the window growing
    tight = SmallToBig(small, window=3, max_tokens=20).expand([{"chunk_id": "rrf_3", "score": 1}])
    assert tight[0]["child_ids"] == ["rrf_3"]
    # section key: window doesn't spill into the next section
    sec = [dict(c, section=c["chunk_index"] // 4) for c in small]
    sp = SmallToBig(sec, window=1, parent_key="section", merge_ratio=1.0).expand([{"chunk_id": "rrf_3", "score": 1}])
    assert sp[0]["child_ids"] == ["rrf_2", "rrf_3"]
    print("\nself-checks passed")


def main():
    parser = argparse.ArgumentParser(description="Expand small-chunk hits into windows / parent sections")
    parser.add_argument("hits", nargs="?", help="retrieved/reranked hits JSON (needs chunk_id, score)")
    parser.add_argument("--chunks", help="full corpus chunks JSON the hits came from")
    parser.add_argument("--out", default="expanded.json")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="neighbours each side")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="cap per window")
    parser.add_argument("--parent-key", help="chunk field that defines sections, e.g. headings")
    parser.add_argument("--parent-size", type=int, default=0, help="fixed parent size in children (0 = off)")
    parser.add_argument("--merge-ratio", type=float, default=DEFAULT_MERGE_RATIO)
    parser.add_argument("--max-parent-tokens", type=int, default=DEFAULT_MAX_PARENT_TOKENS)
    parser.add_argument("--score-key", default="score")
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--demo", action="store_true", help="offline demo + self-checks")
    args = parser.parse_args()

    if args.demo:
        demo()
        return

    if not (args.hits and args.chunks):
        parser.print_help()
        return

    with open(args.chunks) as f:
        corpus = json.load(f)
    with open(args.hits) as f:
        hits = json.load(f)
    log.info(f"Loaded {len(corpus)} corpus chunks, {len(hits)} hits")

    s2b = SmallToBig(
        corpus,
        window=args.window,
        max_tokens=args.max_tokens,
        parent_key=args.parent_key,
        parent_size=args.parent_size,
        merge_ratio=args.merge_ratio,
        max_parent_tokens=args.max_parent_tokens,
        text_key=args.text_key,
    )
    spans = s2b.expand(hits, score_key=args.score_key)

    with open(args.out, "w") as f:
        json.dump(spans, f, indent=2, ensure_ascii=False)
    log.info(f"Wrote {len(spans)} spans to {args.out}")


if __name__ == "__main__":
    main()
