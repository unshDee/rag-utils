"""
Binary-quantized dense index with float rescoring — 32x smaller vectors, ~same recall.

A float32 embedding of a 1024-dim model is 4 KB. Ten million chunks is 40 GB of RAM
just to hold the index, before you've stored any text. Two tricks cut that down and
they stack, because they compress along different axes:

  1. Matryoshka truncation (Kusupati et al. 2022, https://arxiv.org/abs/2205.13147)
     MRL-trained models front-load information into the leading dimensions, so you
     can keep the first 256 of 1024 dims and lose surprisingly little. Shrinks the
     *number* of dims. Only works on models trained for it (nomic-embed, mxbai,
     jina-v3, OpenAI text-embedding-3, Voyage, most 2025+ open models) — on a
     non-MRL model truncation is just damage.

  2. Binary quantization: keep only the sign of each dim, 1 bit instead of 32, and
     search by Hamming distance (XOR + popcount — very fast). Shrinks the *size* of
     each dim. Alone it loses a lot of recall.

The part that makes binary actually usable is rescoring (first used in the Binary
Passage Retriever, Yamada et al. 2021): oversample candidates from the binary index
(say 4x top_k), then rescore just those with the full-precision *query* against the
binary *docs*. Query stays float, so the dot product still carries magnitude
information the doc side threw away. Reported to keep ~96% of float retrieval quality
at 1/32 the memory. If you keep float docs on disk (not RAM), a final exact rescore
of the top few gets you to ~100%.

Run --demo to see the recall/memory tradeoff on your own embeddings or synthetic ones.

Requires: numpy

Usage:
    python binary_quant_index.py --demo
    python binary_quant_index.py --embeddings late_embeddings.npz --dims 256 --oversample 4

    from binary_quant_index import BinaryIndex
    index = BinaryIndex(doc_embeddings, dims=256)
    hits = index.search(query_embedding, top_k=10, oversample=4)   # [(idx, score), ...]
"""

import time
import logging
import argparse

import numpy as np

log = logging.getLogger("binary_quant_index")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

DEFAULT_OVERSAMPLE = 4  # 4x is the usual sweet spot; past ~10x recall barely moves

# popcount lookup for one byte, so Hamming distance is a table lookup + sum
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def truncate(embs: np.ndarray, dims: int | None) -> np.ndarray:
    """
    Matryoshka truncation: keep the leading dims, then re-normalize.
    Re-normalizing matters — the truncated vector's norm is < 1, and cosine scores
    across differently-truncated vectors stop being comparable without it.
    """
    if dims is None or dims >= embs.shape[-1]:
        return _l2(embs.astype(np.float32))
    return _l2(embs[..., :dims].astype(np.float32))


def binarize(embs: np.ndarray) -> np.ndarray:
    """sign -> bits, packed 8 per byte. (n, d) float -> (n, ceil(d/8)) uint8"""
    return np.packbits(embs > 0, axis=-1)


def hamming(q_bits: np.ndarray, doc_bits: np.ndarray) -> np.ndarray:
    """Hamming distance from one packed query to every packed doc."""
    return _POPCOUNT[np.bitwise_xor(doc_bits, q_bits)].sum(axis=1, dtype=np.int32)


class BinaryIndex:
    """
    Stores only packed bits (plus optional float docs for an exact final rescore).

    keep_float=False is the point of the thing — 1 bit/dim in memory. Set it True
    only if you're measuring, or the floats live in a memory-mapped file anyway.
    """

    def __init__(self, embeddings: np.ndarray, dims: int | None = None, keep_float: bool = False):
        self.dims = dims
        docs = truncate(np.asarray(embeddings), dims)
        self.bits = binarize(docs)
        self.n, self.d = docs.shape
        self._float = docs if keep_float else None

        log.info(
            f"Indexed {self.n} docs: {self.d} dims, "
            f"{self.bits.nbytes / 1e6:.2f} MB binary vs {self.n * self.d * 4 / 1e6:.2f} MB float32"
        )

    def _unpack(self, idx: np.ndarray) -> np.ndarray:
        # only ever unpack the candidate set, so memory stays at 1 bit/dim
        signs = np.unpackbits(self.bits[idx], axis=-1)[:, : self.d].astype(np.float32)
        return signs * 2 - 1  # {0,1} -> {-1,+1}

    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
        oversample: int = DEFAULT_OVERSAMPLE,
        exact_rescore: bool = False,
    ) -> list[tuple[int, float]]:
        """
        1. Hamming search over bits for top_k * oversample candidates
        2. rescore candidates: float query · ±1 doc (asymmetric — recovers most recall)
        3. optional: exact float rescore if keep_float=True
        """
        q = truncate(np.asarray(query).reshape(1, -1), self.dims)[0]
        n_cand = min(self.n, top_k * max(oversample, 1))

        dist = hamming(binarize(q), self.bits)
        cand = np.argpartition(dist, n_cand - 1)[:n_cand] if n_cand < self.n else np.arange(self.n)

        if exact_rescore and self._float is not None:
            scores = self._float[cand] @ q
        elif oversample > 1:
            scores = self._unpack(cand) @ q
        else:
            # no rescoring: rank by Hamming alone, score as normalized agreement
            scores = 1 - 2 * dist[cand] / self.d

        order = np.argsort(scores)[::-1][:top_k]
        return [(int(cand[i]), float(scores[i])) for i in order]


def evaluate(index: BinaryIndex, docs: np.ndarray, queries: np.ndarray, sources: np.ndarray, k: int, **kw):
    """
    Two numbers per config:
      hit@k   : is the doc each query was made from in the top-k? (retrieval still works)
      overlap : share of the exact float top-k reproduced (how much the ranking drifted)
    Overlap is the harsher one — near-ties among the top-k get reshuffled by quantization
    without hurting whether the right doc is found.
    """
    exact_docs = truncate(docs, None)
    hits = overlap = 0
    for q, src in zip(queries, sources):
        truth = set(np.argsort(exact_docs @ truncate(q, None))[::-1][:k].tolist())
        got = {i for i, _ in index.search(q, top_k=k, **kw)}
        hits += int(src) in got
        overlap += len(truth & got)
    return hits / len(queries), overlap / (k * len(queries))


def _synthetic(n_docs: int = 20000, n_queries: int = 200, dim: int = 768, seed: int = 0):
    """
    Clustered, MRL-ish embeddings: variance decays across dims, so leading dims carry
    most of the signal — roughly what a Matryoshka-trained model's output looks like.
    Queries are heavy-noise copies of a known source doc. Uniform random vectors would
    make every method look equally bad.
    """
    rng = np.random.default_rng(seed)
    decay = 1.0 / np.sqrt(1 + np.arange(dim) / 32)
    centers = rng.standard_normal((2000, dim)) * decay
    docs = centers[rng.integers(0, 2000, n_docs)] + 0.5 * rng.standard_normal((n_docs, dim)) * decay
    sources = rng.integers(0, n_docs, n_queries)
    queries = docs[sources] + 2.0 * rng.standard_normal((n_queries, dim)) * decay
    return docs.astype(np.float32), queries.astype(np.float32), sources


def _benchmark(docs, queries, sources, k: int, dims_list: list[int | None], oversample: int):
    full = docs.shape[1]
    float_mb = docs.shape[0] * full * 4 / 1e6
    print(f"\n{len(docs)} docs, {len(queries)} queries, k={k}. float32 baseline: {float_mb:.1f} MB\n")
    print(f"{'dims':>6} {'MB':>7} {'compress':>9} {'hit@k':>7} {'+rescore':>9} {'overlap':>8} {'ms/q':>6}")
    print("-" * 58)
    for dims in dims_list:
        idx = BinaryIndex(docs, dims=dims)
        hit_raw, _ = evaluate(idx, docs, queries, sources, k, oversample=1)
        t0 = time.time()
        hit_res, ovl = evaluate(idx, docs, queries, sources, k, oversample=oversample)
        ms = (time.time() - t0) * 1000 / len(queries)
        mb = idx.bits.nbytes / 1e6
        print(f"{idx.d:>6} {mb:>7.2f} {float_mb / mb:>8.0f}x {hit_raw:>7.3f} {hit_res:>9.3f} {ovl:>8.3f} {ms:>6.2f}")
    print(
        f"\n'hit@k' = bits only. '+rescore' = {oversample}x oversample, float query vs ±1 docs.\n"
        "'overlap' = share of the exact float top-k reproduced (with rescore). The rescore\n"
        "step is free — no extra memory — and is what makes heavy truncation usable."
    )


def main():
    parser = argparse.ArgumentParser(description="Binary quantization + Matryoshka truncation with rescoring")
    parser.add_argument("--demo", action="store_true", help="benchmark on synthetic MRL-style embeddings")
    parser.add_argument("--embeddings", help=".npz (key 'embeddings') or .npy of doc embeddings to benchmark")
    parser.add_argument("--dims", type=int, nargs="+", default=None, help="truncation dims to try (default: full, 1/4, 1/8)")
    parser.add_argument("--oversample", type=int, default=DEFAULT_OVERSAMPLE)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-queries", type=int, default=200, help="queries sampled from the docs (with noise)")
    args = parser.parse_args()

    if args.embeddings:
        data = np.load(args.embeddings)
        docs = (data["embeddings"] if hasattr(data, "files") else data).astype(np.float32)
        # no real queries on hand — noisy copies of known docs stand in for them
        rng = np.random.default_rng(0)
        sources = rng.integers(0, len(docs), min(args.n_queries, len(docs)))
        noise = rng.standard_normal((len(sources), docs.shape[1])) * docs.std(axis=0)
        queries = docs[sources] + noise
    elif args.demo:
        docs, queries, sources = _synthetic(n_queries=args.n_queries)
    else:
        parser.print_help()
        return

    full = docs.shape[1]
    dims_list = args.dims or [None, full // 4, full // 8]
    _benchmark(docs, queries, sources, args.k, dims_list, args.oversample)


if __name__ == "__main__":
    main()
