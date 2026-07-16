"""
Semantic chunking — split on meaning instead of character count.

Fixed-size chunking cuts sentences in half and glues the tail onto an unrelated
topic. This embeds each sentence, walks the document, and cuts where consecutive
sentences stop looking alike:

  1. split into sentences
  2. embed each one with a small context buffer, so a short sentence ("It doesn't.")
     is judged in company rather than alone
  3. distance[i] = 1 - cos(emb[i], emb[i+1])
  4. cut wherever distance[i] crosses a threshold
  5. merge undersized chunks, hard-split oversized ones

Threshold is derived from the document's own distance distribution, since a fixed
number never transfers between corpora:
  percentile : cut at the top N% of distances (default, predictable chunk count)
  stddev     : cut at mean + N*sigma (adapts to how coherent the doc is)
  gradient   : cut at inflection points — for dense single-topic docs where absolute
               distance stays low and the change in distance is the real signal

Requires: numpy  sentence-transformers (optional)
Falls back to a stdlib hashing embedder so it runs with no model download.

Usage:
    python semantic_chunker.py doc.md --model all-MiniLM-L6-v2 --out chunks.json
    python semantic_chunker.py doc.md --method stddev --threshold 1.0 --max-tokens 400

    from semantic_chunker import SemanticChunker
    chunker = SemanticChunker(embedder=model.encode)
    chunks = chunker.chunk(text, source="doc.md")
"""

import re
import json
import logging
import argparse
from pathlib import Path
from typing import Any, Callable

import numpy as np

log = logging.getLogger("semantic_chunker")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# sentence-final punctuation + whitespace + capital/quote/digit, guarding the usual
# abbreviation traps. Swap in spaCy/pysbd if your text is gnarly.
_ABBREV = r"(?<!\bDr)(?<!\bMr)(?<!\bMrs)(?<!\bMs)(?<!\bProf)(?<!\bFig)(?<!\bEq)(?<!\bNo)(?<!\bvs)(?<!\bet al)(?<!\be\.g)(?<!\bi\.e)"
SENTENCE_RE = re.compile(rf"{_ABBREV}(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]*[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    sentences = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        for s in SENTENCE_RE.split(para):
            s = s.strip()
            if s:
                sentences.append(s)
    return sentences


def estimate_tokens(text: str) -> int:
    """~4 chars/token — fine for chunk sizing, not for billing"""
    return max(1, len(text) // 4)


class _HashingEmbedder:
    """
    dependency-free fallback: character n-gram hashing + L2 norm

    Bag-of-ngrams, so it only sees lexical overlap. Fine for demos and CI; use a real
    sentence encoder in production or your breakpoints stay surface-level.
    """

    def __init__(self, dim: int = 512, ngram: int = 4):
        self.dim = dim
        self.ngram = ngram

    def __call__(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            t = re.sub(r"\s+", " ", text.lower())
            for j in range(max(1, len(t) - self.ngram + 1)):
                gram = t[j : j + self.ngram]
                out[i, hash(gram) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1e-9, norms)


class SemanticChunker:
    def __init__(
        self,
        embedder: Callable[[list[str]], Any] | None = None,
        method: str = "percentile",
        threshold: float = 95.0,
        buffer_size: int = 1,
        min_tokens: int = 60,
        max_tokens: int = 500,
    ):
        """
        embedder    : callable(list[str]) -> array (n, d). Defaults to hashing fallback.
        method      : percentile | stddev | gradient
        threshold   : percentile (0-100) | number of stddevs | gradient percentile
        buffer_size : sentences of context on each side when embedding a sentence
        min_tokens  : chunks below this get merged into a neighbour
        max_tokens  : chunks above this get hard-split — a semantic cut that never came
                      still has to respect the embedding model's token limit
        """
        self.embedder = embedder or _HashingEmbedder()
        self.method = method
        self.threshold = threshold
        self.buffer_size = buffer_size
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    def _windowed(self, sentences: list[str]) -> list[str]:
        if self.buffer_size == 0:
            return sentences
        out = []
        for i in range(len(sentences)):
            lo = max(0, i - self.buffer_size)
            hi = min(len(sentences), i + self.buffer_size + 1)
            out.append(" ".join(sentences[lo:hi]))
        return out

    def _distances(self, sentences: list[str]) -> np.ndarray:
        embs = np.asarray(self.embedder(self._windowed(sentences)), dtype=np.float32)

        # normalize for cosine sim via dot product
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / np.where(norms == 0, 1e-9, norms)

        sims = np.sum(embs[:-1] * embs[1:], axis=1)
        return 1.0 - sims

    def _breakpoints(self, distances: np.ndarray) -> list[int]:
        if len(distances) == 0:
            return []

        if self.method == "percentile":
            cutoff = float(np.percentile(distances, self.threshold))
            return [int(i) for i in np.where(distances > cutoff)[0]]

        if self.method == "stddev":
            cutoff = float(distances.mean() + self.threshold * distances.std())
            return [int(i) for i in np.where(distances > cutoff)[0]]

        if self.method == "gradient":
            # cut where the distance curve rises fastest, not where it's highest —
            # catches topic drift inside uniformly-similar text
            if len(distances) < 3:
                return []
            grad = np.gradient(distances)
            cutoff = float(np.percentile(grad, self.threshold))
            return [int(i) for i in np.where(grad > cutoff)[0]]

        raise ValueError(f"unknown method: {self.method}")

    def _postprocess(self, groups: list[list[str]]) -> list[list[str]]:
        """merge undersized groups forward, hard-split oversized ones"""
        merged: list[list[str]] = []
        for group in groups:
            if merged and estimate_tokens(" ".join(merged[-1])) < self.min_tokens:
                merged[-1].extend(group)
            else:
                merged.append(list(group))

        # a trailing runt has no successor to merge into — fold it backwards
        if len(merged) > 1 and estimate_tokens(" ".join(merged[-1])) < self.min_tokens:
            tail = merged.pop()
            merged[-1].extend(tail)

        out: list[list[str]] = []
        for group in merged:
            if estimate_tokens(" ".join(group)) <= self.max_tokens:
                out.append(group)
                continue
            current: list[str] = []
            for sent in group:
                if current and estimate_tokens(" ".join(current + [sent])) > self.max_tokens:
                    out.append(current)
                    current = [sent]
                else:
                    current.append(sent)
            if current:
                out.append(current)

        return out

    def chunk(self, text: str, source: str = "") -> list[dict[str, Any]]:
        sentences = split_sentences(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return [self._make(0, sentences, source)]

        distances = self._distances(sentences)
        breakpoints = self._breakpoints(distances)

        groups, start = [], 0
        for bp in breakpoints:
            groups.append(sentences[start : bp + 1])
            start = bp + 1
        if start < len(sentences):
            groups.append(sentences[start:])

        groups = self._postprocess(groups)

        log.info(
            f"{len(sentences)} sentences -> {len(breakpoints)} breakpoints -> "
            f"{len(groups)} chunks (mean distance {distances.mean():.3f})"
        )
        return [self._make(i, g, source) for i, g in enumerate(groups)]

    def _make(self, i: int, sentences: list[str], source: str) -> dict[str, Any]:
        text = " ".join(sentences)
        return {
            "chunk_id": f"{Path(source).stem or 'chunk'}_{i:04d}",
            "text": text,
            "source": source,
            "n_sentences": len(sentences),
            "n_tokens_est": estimate_tokens(text),
        }


def main():
    parser = argparse.ArgumentParser(description="Semantic (embedding-breakpoint) chunking")
    parser.add_argument("files", nargs="+", help="text/markdown files to chunk")
    parser.add_argument("--out", default="semantic_chunks.json")
    parser.add_argument("--method", choices=["percentile", "stddev", "gradient"], default="percentile")
    parser.add_argument("--threshold", type=float, default=95.0)
    parser.add_argument("--buffer", type=int, default=1, help="context sentences each side")
    parser.add_argument("--min-tokens", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--model", help="sentence-transformers model (default: hashing fallback)")
    args = parser.parse_args()

    embedder = None
    if args.model:
        from sentence_transformers import SentenceTransformer

        log.info(f"Loading {args.model}...")
        embedder = SentenceTransformer(args.model).encode
    else:
        log.warning("No --model given, using the hashing fallback (lexical only, demo quality)")

    chunker = SemanticChunker(
        embedder=embedder,
        method=args.method,
        threshold=args.threshold,
        buffer_size=args.buffer,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )

    all_chunks = []
    for path in args.files:
        text = Path(path).read_text()
        log.info(f"[{path}] {len(text):,} chars")
        all_chunks.extend(chunker.chunk(text, source=path))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    log.info(f"Saved {len(all_chunks)} chunks to {args.out}")

    sizes = [c["n_tokens_est"] for c in all_chunks]
    if sizes:
        log.info(f"Token estimate: min {min(sizes)} / median {int(np.median(sizes))} / max {max(sizes)}")


if __name__ == "__main__":
    main()
