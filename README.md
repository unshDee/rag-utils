# rag-utils
scripts and code useful for rag, a collection of what I have learned and have used while working on rag pipelines

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
