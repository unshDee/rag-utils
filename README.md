# rag-utils
scripts and code useful for rag

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

