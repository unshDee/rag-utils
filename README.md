# rag-utils
scripts and code useful for rag, a collection of what I have learned and have used while working on rag pipelines

Each script is standalone — no framework, no shared package to install, copy the one you need. Roughly, they slot into a pipeline like this:

```
ingest      docx_to_chunks.py · pdf_to_markdown.py
chunk       semantic_chunker.py · chunk_overlap_merger.py · chunk_quality_scorer.py · semantic_dedup.py
enrich      contextual_retrieval.py · late_chunking.py · embedding_cache.py
retrieve    hybrid_retriever.py · query_expander.py
rerank      cross_encoder_reranker.py
pack        context_window_packer.py
evaluate    eval_set_generator.py → retrieval_eval.py (did we find it?) · answer_eval.py (did we use it?)
observe     rag_tracer.py
```

---

### `docx_to_chunks.py`
Converts `.docx` files into chunks ready for vector database ingestion. Uses [Docling](https://github.com/DS4SD/docling) for document parsing and its `HybridChunker` for splitting. Extracts images (PNG) and tables (CSV + JSON) into a `media/` directory and embeds references to them in each chunk's metadata. Optionally uses a HuggingFace tokenizer so chunk sizes stay aligned with the target embedding model's token limits.

**Deps:** `docling`, `docling-core`, `pandas`, `pillow`, `transformers` (optional)
```
python docx_to_chunks.py   # reads docs/, writes chunks_output.json
```

### `retrieval_eval.py`
Offline retrieval evaluation. Reads a JSONL file where each line has `query`, `retrieved_ids` (ranked), and `relevant_ids` (ground truth). Computes Hit@k, Precision@k, Recall@k, MRR, NDCG@k, and MAP. Includes a `--generate-sample` flag to create test data.

**Deps:** stdlib only
```
python retrieval_eval.py --generate-sample sample.jsonl
python retrieval_eval.py sample.jsonl --k 1 5 10
```

### `chunk_quality_scorer.py`
Scores chunks on five text-quality signals: token count (prefers 60–400), type-token ratio (information density), stopword ratio, sentence completeness (starts capital, ends punctuation), and heading-text overlap. Weighted average → a 0–1 quality score. Useful for filtering noise before indexing.

**Deps:** stdlib only
```
python chunk_quality_scorer.py chunks_output.json --min-score 0.4 --out filtered.json
```

### `chunk_overlap_merger.py`
Finds and merges consecutive chunks that share overlapping text — common when using sliding-window chunking or ingesting the same document through multiple pipelines. Detects overlaps by matching suffixes of chunk N against prefixes of chunk N+1, then stitches them without the repeated portion.

**Deps:** stdlib only
```
python chunk_overlap_merger.py chunks.json --overlap-chars 80 --show-merges
```

### `context_window_packer.py`
Given retrieved chunks with scores and a token budget, greedily selects the best subset to fit in the context window. Sorts by score-per-token (fractional knapsack) so high-value, low-cost chunks always win. Also deduplicates by `chunk_id` and supports a minimum score threshold. Falls back to character-based token estimation if tiktoken isn't installed.

**Deps:** `tiktoken` (optional)
```
python context_window_packer.py results.json --budget 4000 --min-score 0.5 --print-context
```

### `query_expander.py`
Two query expansion techniques to improve retrieval recall. **Multi-query:** rephrases the query N ways (different angles/specificity), runs retrieval for each, and merges results. **HyDE:** generates a hypothetical answer document and uses its embedding as the retrieval query instead — the answer lives closer to real answers in embedding space than the question does. Both use the Claude API (Haiku by default for speed).

**Deps:** `anthropic`, `ANTHROPIC_API_KEY` env var
```
python query_expander.py --query "what causes gradient vanishing?" --mode both
```

### `pdf_to_markdown.py`
Converts PDFs to clean Markdown using `pdfplumber`. Detects two-column layouts (splits and processes each column), converts tables to Markdown table syntax, infers headings from font size ratios, and removes repeated header/footer lines. Does not handle scanned PDFs (no OCR).

**Deps:** `pdfplumber`
```
python pdf_to_markdown.py paper.pdf --out paper.md
python pdf_to_markdown.py *.pdf --out-dir markdown/
```

### `rag_tracer.py`
Lightweight span-based tracer for RAG pipelines. Records named spans with latency, token counts, and custom metadata. Works as both a context manager and a `@traced()` decorator. Writes JSON Lines to a trace file for analysis. Thread-safe, zero overhead when `TRACING_ENABLED=false`. Includes a `print_summary()` that renders a timing tree.

**Deps:** stdlib only
```python
with tracer.span("retrieval", query=q) as s:
    results = search(q)
    s.set(n_results=len(results))

tracer.print_summary()
tracer.flush()   # writes to rag_traces.jsonl
```

### `embedding_cache.py`
SQLite-backed disk cache for embeddings. Hashes `(model_id, text)` pairs so the same text is never re-embedded twice across runs. Supports batch encoding with cache hits, configurable TTL, and any embedder callable (sentence-transformers, OpenAI, Cohere, etc.). Thread-safe via SQLite WAL mode + thread-local connections.

**Deps:** `numpy`, `sentence-transformers` (or any embedder)
```python
cache = EmbeddingCache("embeddings.db", model_id="all-MiniLM-L6-v2")
embeddings = cache.encode(model.encode, texts)   # hits cache on second call
```

### `semantic_dedup.py`
Near-duplicate chunk removal. First does a free exact-dedup pass (MD5 hash). Then either embeds all chunks and does greedy cosine-similarity dedup (prefers higher-quality chunks, skips anything above the similarity threshold), or falls back to character k-shingle Jaccard similarity for a dep-free mode. Threshold is configurable — 0.92 for near-dups, lower for aggressive dedup.

**Deps:** `numpy`, `sentence-transformers` (embedding mode) or stdlib (jaccard mode)
```
python semantic_dedup.py chunks.json --threshold 0.92 --out deduped.json
python semantic_dedup.py chunks.json --mode jaccard --threshold 0.7
```

### `hybrid_retriever.py`
Combines BM25 sparse retrieval (via `rank-bm25`) with dense vector search (FAISS or numpy fallback), fused using Reciprocal Rank Fusion. RRF avoids needing to normalize scores across retrieval systems — each system contributes rank-based votes. Exposes a `HybridRetriever` class importable into any pipeline, plus a standalone `rrf_fuse()` function for custom fusion.

**Deps:** `numpy`, `rank-bm25`, `faiss-cpu` (optional), `sentence-transformers` (for dense)
```
python hybrid_retriever.py --demo
python hybrid_retriever.py --index chunks.json --query "what is RAG?" --top-k 5
```
```python
retriever = HybridRetriever(chunks, embedder=model.encode)
results = retriever.search("how does chunking work?", top_k=10)
```

### `contextual_retrieval.py`
Implements [Anthropic's Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) (Sept 2024): before indexing, each chunk gets a short LLM-written blurb prepended that situates it in its parent document — resolving the pronouns, acronyms and relative dates that make an isolated chunk unretrievable. Reported ~49% fewer retrieval failures combined with contextual BM25, ~67% with a reranker on top. The naive implementation is ruinously expensive (you resend the whole document per chunk), so this one puts the document in a **prompt-cache prefix** — chunk N+1 reads it at 0.1x input price — warms the cache with a single serial call before fanning out, and prints the cache hit rate and cost saved so you can prove it worked.

**Deps:** `anthropic`, `ANTHROPIC_API_KEY` env var
```
python contextual_retrieval.py chunks.json --out contextual_chunks.json
python contextual_retrieval.py chunks.json --limit 20   # dry run, check the cost first
```

### `cross_encoder_reranker.py`
The stage between `hybrid_retriever.py` and `context_window_packer.py`. Bi-encoders embed query and document separately, so the model must compress a document into a vector *before* knowing what will be asked of it; a cross-encoder concatenates the pair and runs full attention across both. Far more accurate, far too slow to run over a corpus — so retrieve top-100 cheaply, rerank to top-5 here. Three tools: `CrossEncoderReranker` (local `bge-reranker-v2-m3`, free and fast — the default), `LLMReranker` (listwise permutation via Claude, [RankGPT](https://arxiv.org/abs/2304.09542)-style with a sliding window so a good doc at rank 40 can reach rank 1), and `mmr()` for diversity — so your top-5 isn't five paraphrases of the same paragraph. Supports an absolute `--min-score` cutoff, which lets the pipeline return *nothing* rather than hand the LLM five irrelevant chunks.

**Deps:** `numpy`, `sentence-transformers` (cross-encoder), `anthropic` (LLM rerank)
```
python cross_encoder_reranker.py --demo
python cross_encoder_reranker.py results.json --query "what is RRF?" --top-k 5 --min-score 0
```

### `semantic_chunker.py`
Splits on meaning instead of character count. Embeds each sentence (with a context buffer, so a short sentence isn't judged alone), measures cosine distance between consecutive sentences, and cuts where the topic shifts. The threshold is derived from the document's own distance distribution — a fixed number never transfers between corpora — via `percentile`, `stddev`, or `gradient` (cut at inflection points; for dense single-topic docs where absolute distance stays low and the *change* is the signal). Merges undersized chunks and hard-splits oversized ones so output still respects your embedding model's token limit. Ships with a stdlib hashing embedder so it runs with no model download.

**Deps:** `numpy`, `sentence-transformers` (optional — falls back to lexical hashing)
```
python semantic_chunker.py doc.md --model all-MiniLM-L6-v2 --out chunks.json
python semantic_chunker.py doc.md --method gradient --threshold 90 --max-tokens 400
```

### `late_chunking.py`
[Late Chunking](https://arxiv.org/abs/2409.04701) (Günther et al. 2024) — chunk *after* embedding, not before. A chunk reading *"It was founded in 1209"* has no idea it's about Cambridge, so no query about Cambridge retrieves it. This runs the whole document through a long-context encoder, keeps the **per-token** embeddings (each has already attended over the entire document), and only then applies chunk boundaries by mean-pooling the tokens inside each chunk's character span. Same chunks, same model, same vector count — but conditioned on the full document, and one forward pass instead of N. Fixes the same disease as `contextual_retrieval.py`, for free, with no LLM. Documents past the context limit use overlapping macro-windows with cross-window pooling. `--demo` reproduces the paper's result: chunks that never say "Cambridge" go from 0.69–0.72 similarity to 0.83–0.84 (spread against the naming chunk collapses from 0.157 to 0.037).

**Deps:** `torch`, `transformers`, `numpy` (needs a fast tokenizer for offset mapping)
```
python late_chunking.py --demo
python late_chunking.py chunks.json --doc source.md --out embeddings.npz
```

### `answer_eval.py`
`retrieval_eval.py` scores whether you **found** the right chunks; this scores what the LLM then **did** with them. Five LLM-judged metrics, each decomposed into atomic claims rather than asking a model for a vibes-based 1–10 score: **faithfulness** (share of answer claims actually supported by the retrieved context — your hallucination rate), **answer_relevance** (reverse-engineers questions from the answer and compares to the real one; catches evasion), **context_precision** (how much of what you retrieved was useful), **context_recall** (ceiling on faithfulness — the generator can't cite what you never gave it, so this isolates blame between retriever and generator), and **citation verification** (does `[2]` actually say that?). Metric framing follows [RAGAS](https://arxiv.org/abs/2309.15217), reimplemented with no framework so you can read and edit the judge prompts.

**Deps:** `anthropic`, `ANTHROPIC_API_KEY` env var
```
python answer_eval.py --generate-sample sample.jsonl
python answer_eval.py evals.jsonl --out report.json
```

### `eval_set_generator.py`
Nobody hand-labels 200 queries, so most RAG systems ship with no evaluation and "did that change help?" gets answered by vibes. This builds the golden set from the corpus you already have — and because every question is generated *from* known chunks, the `relevant_ids` labels come free. Four question types: `simple`, `multi_hop` (needs two chunks from one document — breaks naive top-1 retrieval, which is the point), `reasoning` (kills lexical-only retrievers), and `unanswerable` (plausible, on-topic, not in the corpus — the only way to measure whether your system abstains instead of confabulating; most eval sets omit these and they're the ones that catch real production failures). Every question then passes an independent **critique gate** scoring groundedness / standalone-ness / specificity, and anything below threshold is dropped — ungated synthetic questions are worse than no eval set, since they encode the generator's confusion as ground truth. Rejects are returned rather than discarded: a high rejection rate means your *chunks* are too fragmented to answer anything, and no retriever will save you. Output feeds straight into `retrieval_eval.py`.

**Deps:** `anthropic`, `ANTHROPIC_API_KEY` env var
```
python eval_set_generator.py chunks.json --n 50 --out golden.jsonl
python retrieval_eval.py golden.jsonl --k 1 5 10     # ...then measure
```
