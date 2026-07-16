"""
Late Chunking — chunk after embedding, not before.

The standard pipeline embeds each chunk in isolation, which throws away everything
the chunk needed from its neighbours. A chunk reading "It was founded in 1209. The
city has a population of ~145,000." never says Cambridge, so no query about Cambridge
retrieves it. The embedding cannot encode what the text never says.

Late chunking flips the order:
  1. run the whole document through a long-context embedding model
  2. keep the per-token output embeddings — each one has already attended over the
     entire document, so the token "It" carries Cambridge in its vector
  3. only now apply chunk boundaries: mean-pool the token embeddings falling inside
     each chunk's character span

Same chunks, same model, same vector count — the chunk embeddings are just conditioned
on the full document. One forward pass instead of N. Fixes the same problem as
contextual_retrieval.py, without an LLM.

  Günther et al. 2024 — https://arxiv.org/abs/2409.04701

Documents longer than the model's context use overlapping macro-windows, with each
chunk pooled across every window that covers it.

Requires: torch  transformers  numpy
Needs a fast tokenizer (offset mapping). Runs on CPU, slowly.

Usage:
    python late_chunking.py --demo
    python late_chunking.py chunks.json --doc source.md --out embeddings.npz

    from late_chunking import LateChunker, spans_from_chunks
    chunker = LateChunker()
    embeddings = chunker.embed_chunks(doc_text, spans)   # spans: [(start, end), ...]
"""

import json
import logging
import argparse

import numpy as np

log = logging.getLogger("late_chunking")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# needs a long-context encoder that exposes token-level hidden states
# jina-embeddings-v2-small-en: 8192 ctx, 33M params, fine on a laptop CPU
DEFAULT_MODEL = "jinaai/jina-embeddings-v2-small-en"


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


class LateChunker:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_length: int = 8192,
        overlap_tokens: int = 512,
        device: str | None = None,
    ):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError("pip install torch transformers")

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        log.info(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if not self.tokenizer.is_fast:
            raise ValueError(
                "needs a fast tokenizer — the character offset mapping is what maps "
                "chunk spans back onto token positions"
            )
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(self.device).eval()

        self.max_length = max_length
        self.overlap_tokens = overlap_tokens

    def _encode_window(self, text: str, char_offset: int = 0):
        """one forward pass — returns (token_embeddings, char_span_per_token)"""
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length,
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with self.torch.no_grad():
            hidden = self.model(**enc).last_hidden_state[0]  # (T, D)

        token_embs = hidden.cpu().numpy().astype(np.float32)

        # (0, 0) marks special tokens (CLS/SEP) — they belong to no chunk
        spans = [(s + char_offset, e + char_offset) if e > s else (-1, -1) for s, e in offsets]
        return token_embs, spans

    def _windows(self, text: str) -> list[tuple[str, int]]:
        """split a too-long document into overlapping (text, char_offset) windows"""
        enc = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc["offset_mapping"]
        n = len(offsets)

        budget = self.max_length - 2  # room for CLS/SEP
        if n <= budget:
            return [(text, 0)]

        step = max(1, budget - self.overlap_tokens)
        windows = []
        for start in range(0, n, step):
            end = min(start + budget, n)
            c0 = offsets[start][0]
            c1 = offsets[end - 1][1]
            windows.append((text[c0:c1], c0))
            if end == n:
                break

        log.info(f"Document is {n:,} tokens, using {len(windows)} overlapping windows")
        return windows

    def embed_chunks(self, text: str, spans: list[tuple[int, int]]) -> np.ndarray:
        """
        spans: character (start, end) per chunk, into `text`.
        Returns (n_chunks, dim), L2-normalized.

        Pools by accumulating sums and counts, so a chunk covered by two overlapping
        windows is averaged rather than double-counted.
        """
        if not spans:
            return np.zeros((0, 0), dtype=np.float32)

        sums: np.ndarray | None = None
        counts = np.zeros(len(spans), dtype=np.float32)

        for window_text, char_offset in self._windows(text):
            token_embs, token_spans = self._encode_window(window_text, char_offset)
            if sums is None:
                sums = np.zeros((len(spans), token_embs.shape[1]), dtype=np.float32)

            for t, (ts, te) in enumerate(token_spans):
                if ts < 0:
                    continue
                for c, (cs, ce) in enumerate(spans):
                    if ts >= cs and te <= ce:
                        sums[c] += token_embs[t]
                        counts[c] += 1
                        break

        empty = int((counts == 0).sum())
        if empty:
            log.warning(f"{empty} chunk(s) matched no tokens — check that spans index into `text`")

        pooled = sums / np.maximum(counts[:, None], 1.0)
        return _l2(pooled)

    def embed_naive(self, texts: list[str]) -> np.ndarray:
        """baseline: embed each chunk in isolation — this is what you're replacing"""
        out = []
        for t in texts:
            embs, spans = self._encode_window(t)
            mask = np.array([s >= 0 for s, _ in spans])
            out.append(embs[mask].mean(axis=0) if mask.any() else embs.mean(axis=0))
        return _l2(np.stack(out))


def spans_from_chunks(text: str, chunk_texts: list[str]) -> list[tuple[int, int]]:
    """
    Locate each chunk's character span in the source document.

    Scans forward from the previous match so repeated boilerplate (a recurring header)
    doesn't collapse onto the first hit.
    """
    spans, cursor = [], 0
    for ct in chunk_texts:
        i = text.find(ct, cursor)
        if i < 0:
            i = text.find(ct)  # chunker may have normalized whitespace, try anywhere
        if i < 0:
            log.warning(f"Chunk not found verbatim in document: {ct[:60]!r}")
            spans.append((0, 0))
            continue
        spans.append((i, i + len(ct)))
        cursor = i + len(ct)
    return spans


def _demo():
    """does 'It' know it means Cambridge?"""
    doc = (
        "Cambridge is a city on the River Cam in England, about 55 miles north of London. "
        "It is home to the University of Cambridge, one of the world's oldest universities. "
        "It was founded in 1209 and consistently ranks among the top universities globally. "
        "The city has a population of around 145,000 residents. "
        "Its economy is centred on technology and biotechnology research, a cluster often "
        "referred to as Silicon Fen."
    )
    chunk_texts = [s.strip() + "." for s in doc.split(". ") if s.strip()]
    spans = spans_from_chunks(doc, [c.rstrip(".") for c in chunk_texts])

    chunker = LateChunker()
    late = chunker.embed_chunks(doc, spans)
    naive = chunker.embed_naive([doc[s:e] for s, e in spans])

    query = "Cambridge"
    q = chunker.embed_naive([query])[0]

    print(f"\nQuery: {query!r}\n")
    print(f"{'chunk':<62} {'naive':>7} {'late':>7}")
    print("-" * 80)
    for i, ct in enumerate(chunk_texts):
        print(f"{ct[:60]:<62} {float(naive[i] @ q):>7.3f} {float(late[i] @ q):>7.3f}")

    ns, ls = naive @ q, late @ q
    print(f"\nnaive spread: {ns.min():.3f}-{ns.max():.3f} (range {np.ptp(ns):.3f})")
    print(f"late  spread: {ls.min():.3f}-{ls.max():.3f} (range {np.ptp(ls):.3f})")
    print(
        "\nThe chunks starting with 'It' / 'Its' never say Cambridge, and score well below\n"
        "the first chunk under naive embedding. Late chunking pulls them up to nearly the\n"
        "same score — those tokens attended over the first sentence before being pooled."
    )


def main():
    parser = argparse.ArgumentParser(description="Late chunking (Günther et al. 2024)")
    parser.add_argument("chunks", nargs="?", help="chunks JSON (all from one document)")
    parser.add_argument("--doc", help="path to the source document text")
    parser.add_argument("--out", default="late_embeddings.npz")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--demo", action="store_true", help="run demo with synthetic data")
    args = parser.parse_args()

    if args.demo or not args.chunks:
        _demo()
        return

    with open(args.chunks) as f:
        chunks = json.load(f)
    log.info(f"Loaded {len(chunks)} chunks from {args.chunks}")

    texts = [(c.get(args.text_key) or "") for c in chunks]
    doc = open(args.doc).read() if args.doc else "\n\n".join(texts)

    chunker = LateChunker(args.model)
    spans = spans_from_chunks(doc, texts)
    embs = chunker.embed_chunks(doc, spans)

    ids = [c.get("chunk_id", str(i)) for i, c in enumerate(chunks)]
    np.savez(args.out, embeddings=embs, chunk_ids=np.array(ids))
    log.info(f"Saved {embs.shape[0]} x {embs.shape[1]} embeddings to {args.out}")


if __name__ == "__main__":
    main()
