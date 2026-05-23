"""
Disk-backed embedding cache using SQLite.

Caches embeddings by a hash of the (model_id, text) pair so you never
re-embed the same text twice. Significant speedup when iterating on
downstream code (retrieval, eval) without changing the corpus.

Features:
  - batch encode with cache hits
  - configurable TTL (default: no expiry)
  - thread-safe (SQLite WAL mode)
  - works with any embedder (sentence-transformers, OpenAI, Cohere, etc.)

Usage:
    from embedding_cache import EmbeddingCache
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    cache = EmbeddingCache("embeddings.db", model_id="all-MiniLM-L6-v2")

    # first call encodes, subsequent calls hit cache
    embeddings = cache.encode(model.encode, texts)

    # also works as a drop-in wrapper
    encode = cache.wrap(model.encode)
    embs = encode(["hello world", "foo bar"])
"""

import io
import time
import sqlite3
import hashlib
import logging
import argparse
import threading
from pathlib import Path
from typing import Callable, Any

import numpy as np

log = logging.getLogger("embedding_cache")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    key         TEXT PRIMARY KEY,
    model_id    TEXT NOT NULL,
    embedding   BLOB NOT NULL,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_model ON embeddings(model_id);
"""


def _text_key(model_id: str, text: str) -> str:
    h = hashlib.sha256(f"{model_id}\x00{text}".encode()).hexdigest()
    return h


def _serialize(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def _deserialize(blob: bytes) -> np.ndarray:
    return np.load(io.BytesIO(blob))


class EmbeddingCache:
    def __init__(
        self,
        db_path: str | Path = "embeddings.db",
        model_id: str = "unknown",
        ttl_seconds: float | None = None,
    ):
        self.db_path = str(db_path)
        self.model_id = model_id
        self.ttl = ttl_seconds
        self._local = threading.local()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript(SCHEMA)
            conn.commit()
            self._local.conn = conn
        return self._local.conn

    def _init_db(self):
        # just make sure the schema exists
        self._conn()

    def get(self, text: str) -> np.ndarray | None:
        key = _text_key(self.model_id, text)
        conn = self._conn()
        row = conn.execute(
            "SELECT embedding, created_at FROM embeddings WHERE key = ?", (key,)
        ).fetchone()

        if row is None:
            return None

        blob, created_at = row
        if self.ttl is not None and (time.time() - created_at) > self.ttl:
            conn.execute("DELETE FROM embeddings WHERE key = ?", (key,))
            conn.commit()
            return None

        return _deserialize(blob)

    def set(self, text: str, embedding: np.ndarray):
        key = _text_key(self.model_id, text)
        blob = _serialize(embedding)
        conn = self._conn()
        conn.execute(
            "INSERT OR REPLACE INTO embeddings (key, model_id, embedding, created_at) VALUES (?,?,?,?)",
            (key, self.model_id, blob, time.time()),
        )
        conn.commit()

    def encode(
        self,
        embedder_fn: Callable[[list[str]], Any],
        texts: list[str],
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        Encode a list of texts, using cache for any that have been seen before.
        Returns numpy array of shape (len(texts), embedding_dim).
        """
        results: dict[int, np.ndarray] = {}
        uncached_indices = []
        uncached_texts = []

        for i, text in enumerate(texts):
            cached = self.get(text)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        cache_hits = len(texts) - len(uncached_texts)
        log.info(
            f"Cache: {cache_hits}/{len(texts)} hits, encoding {len(uncached_texts)} new texts"
        )

        if uncached_texts:
            # batch encode in chunks to avoid OOM on large corpora
            all_new_embs = []
            for start in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[start:start + batch_size]
                embs = embedder_fn(batch)
                all_new_embs.extend(embs)

            for i, (orig_idx, text) in enumerate(zip(uncached_indices, uncached_texts)):
                emb = np.array(all_new_embs[i])
                self.set(text, emb)
                results[orig_idx] = emb

        return np.stack([results[i] for i in range(len(texts))])

    def wrap(self, embedder_fn: Callable, batch_size: int = 64) -> Callable:
        """
        Returns a drop-in wrapper around embedder_fn that transparently caches.
        The wrapped function accepts list[str] and returns np.ndarray.
        """
        def wrapped(texts: list[str]) -> np.ndarray:
            return self.encode(embedder_fn, texts, batch_size=batch_size)
        return wrapped

    def stats(self) -> dict:
        conn = self._conn()
        total = conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE model_id = ?", (self.model_id,)
        ).fetchone()[0]
        all_time = conn.execute(
            "SELECT MIN(created_at), MAX(created_at) FROM embeddings WHERE model_id = ?",
            (self.model_id,)
        ).fetchone()
        db_size = Path(self.db_path).stat().st_size if Path(self.db_path).exists() else 0
        return {
            "model_id": self.model_id,
            "cached_texts": total,
            "db_size_mb": round(db_size / 1e6, 2),
            "oldest": all_time[0],
            "newest": all_time[1],
        }

    def clear(self, model_id: str | None = None):
        conn = self._conn()
        if model_id:
            conn.execute("DELETE FROM embeddings WHERE model_id = ?", (model_id,))
        else:
            conn.execute("DELETE FROM embeddings WHERE model_id = ?", (self.model_id,))
        conn.commit()
        log.info("Cache cleared")

    def purge_expired(self):
        if self.ttl is None:
            log.info("No TTL set, nothing to purge")
            return
        cutoff = time.time() - self.ttl
        conn = self._conn()
        conn.execute("DELETE FROM embeddings WHERE created_at < ?", (cutoff,))
        conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Embedding cache management")
    parser.add_argument("--db", default="embeddings.db")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--demo", action="store_true", help="run a quick encode demo")
    args = parser.parse_args()

    cache = EmbeddingCache(args.db, model_id=args.model)

    if args.stats:
        import json
        print(json.dumps(cache.stats(), indent=2))

    if args.clear:
        cache.clear()

    if args.demo:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("pip install sentence-transformers to run the demo")
            return

        model = SentenceTransformer(args.model)
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "RAG combines retrieval with language model generation",
            "Embeddings map text into a high-dimensional vector space",
            "The quick brown fox jumps over the lazy dog",  # duplicate — should hit cache
        ]

        print("First encode (0 cache hits expected):")
        embs1 = cache.encode(model.encode, texts)
        print(f"  shape: {embs1.shape}")

        print("\nSecond encode (all cache hits expected):")
        embs2 = cache.encode(model.encode, texts)

        diff = np.max(np.abs(embs1 - embs2))
        print(f"  max diff from first run: {diff:.2e} (should be ~0)")
        print(f"\nCache stats: {cache.stats()}")


if __name__ == "__main__":
    main()
