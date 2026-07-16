"""
Contextual Retrieval: prepend an LLM-written context blurb to each chunk before
you embed or index it.

A chunk that reads "revenue grew 3% over the previous quarter" has no company and
no quarter in it, so it never matches a query naming either. This rewrites it as
"This chunk is from an SEC filing on ACME Corp's Q2 2023 performance... revenue
grew 3% over the previous quarter."

Anthropic reported ~35% fewer retrieval failures from contextual embeddings, ~49%
combined with contextual BM25, ~67% with a reranker on top.
https://www.anthropic.com/news/contextual-retrieval

Sending the whole document with every chunk is expensive, so the document goes in
a cached prompt prefix — later chunks read it at ~0.1x input price instead of full.
Cache hit rate and cost are printed at the end.

Requires: anthropic (pip install anthropic)
Set ANTHROPIC_API_KEY in env.

Usage:
    python contextual_retrieval.py chunks.json --out contextual_chunks.json
    python contextual_retrieval.py chunks.json --limit 20   # dry run, check cost first

    from contextual_retrieval import contextualize_document
    chunks = contextualize_document(doc_text, chunks)
"""

import os
import json
import time
import logging
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

log = logging.getLogger("contextual_retrieval")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

DEFAULT_MODEL = "claude-haiku-4-5"
MAX_TOKENS = 300

# USD per million tokens. Cache writes bill at 1.25x input, reads at 0.1x.
# Only used for the cost printout — update if pricing moves.
PRICING = {
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-sonnet-5": {"input": 3.00, "output": 15.00},
}

# The document block is the cache prefix — it must be byte-identical on every
# call or the cache misses. Keep it first, keep it frozen.
DOC_PROMPT = """<document>
{doc}
</document>"""

CHUNK_PROMPT = """Here is a chunk taken from the document above:

<chunk>
{chunk}
</chunk>

Write a short, standalone context blurb (50-100 tokens) that situates this
chunk within the overall document, so that a search engine can retrieve it
correctly. Resolve any pronouns, acronyms, and relative references (dates,
"the company", "this section") using the document.

Answer with ONLY the context blurb. No preamble, no quotes, no restating the
chunk itself."""

# Caching has a minimum cacheable prefix (4096 tokens on Haiku 4.5). Below that
# it silently no-ops and you still pay the 1.25x write premium, so skip it.
MIN_CACHEABLE_CHARS = 4096 * 3  # ~3 chars/token, conservative


def _get_client():
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY not set in environment")

    return anthropic.Anthropic()


class _UsageTracker:
    """accumulates token usage across threads so we can report cache efficiency"""

    def __init__(self):
        self.counts = defaultdict(int)
        self.calls = 0

    def add(self, usage):
        self.counts["input"] += usage.input_tokens
        self.counts["output"] += usage.output_tokens
        self.counts["cache_write"] += usage.cache_creation_input_tokens or 0
        self.counts["cache_read"] += usage.cache_read_input_tokens or 0
        self.calls += 1

    def report(self, model: str):
        c = self.counts
        cacheable = c["cache_read"] + c["cache_write"]
        hit_rate = c["cache_read"] / cacheable if cacheable else 0.0

        log.info(
            f"{self.calls} calls | uncached in: {c['input']:,} | "
            f"cache write: {c['cache_write']:,} | cache read: {c['cache_read']:,} "
            f"({hit_rate:.0%} hit) | out: {c['output']:,}"
        )

        price = PRICING.get(model)
        if not price:
            return

        cost = (
            c["input"] * price["input"]
            + c["cache_write"] * price["input"] * 1.25
            + c["cache_read"] * price["input"] * 0.10
            + c["output"] * price["output"]
        ) / 1_000_000
        # what it would have cost re-sending the doc uncached every time
        naive = (
            (c["input"] + c["cache_write"] + c["cache_read"]) * price["input"]
            + c["output"] * price["output"]
        ) / 1_000_000
        log.info(f"Cost: ${cost:.4f} (without caching: ${naive:.4f})")


def _context_for_chunk(
    client,
    doc_text: str,
    chunk_text: str,
    model: str,
    usage: _UsageTracker,
    max_retries: int = 4,
) -> str:
    doc_block: dict[str, Any] = {"type": "text", "text": DOC_PROMPT.format(doc=doc_text)}
    if len(doc_text) >= MIN_CACHEABLE_CHARS:
        doc_block["cache_control"] = {"type": "ephemeral"}

    content = [doc_block, {"type": "text", "text": CHUNK_PROMPT.format(chunk=chunk_text)}]

    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": content}],
            )
            usage.add(resp.usage)
            return "".join(b.text for b in resp.content if b.type == "text").strip()
        except Exception as e:
            if attempt == max_retries - 1:
                log.warning(f"Context generation failed after {max_retries} tries: {e}")
                return ""
            time.sleep(2 ** attempt)

    return ""


def contextualize_document(
    doc_text: str,
    chunks: list[dict[str, Any]],
    model: str = DEFAULT_MODEL,
    text_key: str = "text",
    max_workers: int = 8,
    warm_cache: bool = True,
) -> list[dict[str, Any]]:
    """
    Adds `context` and `contextual_text` to every chunk.

    Embed and index contextual_text; keep the original text for what you actually
    show the user / feed the LLM at generation time.

    warm_cache: run the first chunk alone before fanning out. A cache entry is only
    readable once the request that wrote it starts responding, so N cold parallel
    requests would each pay full document price.
    """
    client = _get_client()
    usage = _UsageTracker()
    t0 = time.time()

    def work(i: int) -> tuple[int, str]:
        return i, _context_for_chunk(client, doc_text, chunks[i].get(text_key) or "", model, usage)

    contexts: dict[int, str] = {}
    pending = list(range(len(chunks)))

    if warm_cache and pending and len(doc_text) >= MIN_CACHEABLE_CHARS:
        i, ctx = work(pending.pop(0))
        contexts[i] = ctx

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(work, i) for i in pending]
        for fut in as_completed(futures):
            i, ctx = fut.result()
            contexts[i] = ctx

    out = []
    for i, chunk in enumerate(chunks):
        ctx = contexts.get(i, "")
        raw = chunk.get(text_key) or ""
        new = dict(chunk)
        new["context"] = ctx
        new["contextual_text"] = f"{ctx}\n\n{raw}" if ctx else raw
        out.append(new)

    log.info(f"Contextualized {len(out)} chunks in {time.time()-t0:.1f}s")
    usage.report(model)
    return out


def contextualize_corpus(
    chunks: list[dict[str, Any]],
    doc_key: str = "source",
    text_key: str = "text",
    doc_texts: dict[str, str] | None = None,
    model: str = DEFAULT_MODEL,
    max_workers: int = 8,
) -> list[dict[str, Any]]:
    """
    Multi-document version — groups by doc_key, each document gets its own cache prefix.

    doc_texts: optional {doc_id -> full text}. If missing, the document is rebuilt by
    concatenating its own chunks in order, which is fine when the chunks tile it.
    """
    by_doc: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(chunks):
        by_doc[str(c.get(doc_key, "__unknown__"))].append(i)

    log.info(f"{len(chunks)} chunks across {len(by_doc)} documents")

    out = [dict(c) for c in chunks]
    for doc_id, indices in by_doc.items():
        doc_chunks = [chunks[i] for i in indices]
        doc_text = (
            doc_texts[doc_id]
            if doc_texts and doc_id in doc_texts
            else "\n\n".join((c.get(text_key) or "") for c in doc_chunks)
        )
        log.info(f"[{doc_id}] {len(doc_chunks)} chunks, {len(doc_text):,} chars")
        done = contextualize_document(
            doc_text, doc_chunks, model=model, text_key=text_key, max_workers=max_workers
        )
        for slot, c in zip(indices, done):
            out[slot] = c

    return out


def main():
    parser = argparse.ArgumentParser(description="Contextual Retrieval chunk enrichment")
    parser.add_argument("chunks", help="chunks JSON file")
    parser.add_argument("--out", default="contextual_chunks.json")
    parser.add_argument("--doc-key", default="source", help="field grouping chunks into documents")
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, help="only process the first N chunks")
    args = parser.parse_args()

    with open(args.chunks) as f:
        chunks = json.load(f)
    if args.limit:
        chunks = chunks[: args.limit]
    log.info(f"Loaded {len(chunks)} chunks from {args.chunks}")

    enriched = contextualize_corpus(
        chunks,
        doc_key=args.doc_key,
        text_key=args.text_key,
        model=args.model,
        max_workers=args.workers,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)
    log.info(f"Saved {len(enriched)} chunks to {args.out}")

    for c in enriched[:2]:
        print(f"\n--- {c.get('chunk_id', '?')} ---")
        print(f"context: {c['context']}")
        print(f"text:    {(c.get(args.text_key) or '')[:160]}...")


if __name__ == "__main__":
    main()
