"""
Lightweight tracing for RAG pipelines.

Records named spans with latency, token counts, and custom metadata.
Outputs JSON Lines for easy analysis in pandas / grep / jq.

Design goals:
  - zero-overhead when disabled (TRACING_ENABLED=false)
  - works as context manager and decorator
  - thread-safe (each thread gets its own span stack)
  - no external dependencies

Usage:
    from rag_tracer import tracer, traced

    with tracer.span("retrieval", query="what is X?") as s:
        results = vector_db.search(query)
        s.set(n_results=len(results), top_score=results[0]["score"])

    @traced("rerank")
    def rerank(chunks, query):
        ...

    # at the end of a request:
    tracer.flush()          # writes spans to JSONL file
    tracer.print_summary()  # pretty-print timing tree

Example output (in rag_traces.jsonl):
    {"trace_id": "abc123", "span": "retrieval", "latency_ms": 42.1, "query": "..."}
    {"trace_id": "abc123", "span": "rerank", "latency_ms": 8.3}

Env vars:
    TRACING_ENABLED=false    disable all tracing (production default)
    TRACING_FILE=traces.jsonl  where to write spans
    TRACING_PRINT=true       also print spans to stderr
"""

import os
import sys
import json
import time
import uuid
import logging
import threading
import functools
from pathlib import Path
from typing import Any, Callable
from contextlib import contextmanager
from datetime import datetime, timezone

log = logging.getLogger("rag_tracer")

_ENABLED = os.environ.get("TRACING_ENABLED", "true").lower() not in ("false", "0", "no")
_TRACE_FILE = os.environ.get("TRACING_FILE", "rag_traces.jsonl")
_PRINT = os.environ.get("TRACING_PRINT", "false").lower() in ("true", "1", "yes")


class Span:
    def __init__(self, name: str, trace_id: str, parent_id: str | None = None, **meta):
        self.name = name
        self.span_id = uuid.uuid4().hex[:8]
        self.trace_id = trace_id
        self.parent_id = parent_id
        self.start_time = time.perf_counter()
        self.wall_start = datetime.now(timezone.utc).isoformat()
        self._meta: dict[str, Any] = dict(meta)
        self._end_time: float | None = None
        self._error: str | None = None

    def set(self, **kwargs):
        """attach extra metadata mid-span"""
        self._meta.update(kwargs)
        return self

    def set_tokens(self, input_tokens: int = 0, output_tokens: int = 0):
        self._meta["input_tokens"] = input_tokens
        self._meta["output_tokens"] = output_tokens
        self._meta["total_tokens"] = input_tokens + output_tokens
        return self

    def finish(self, error: str | None = None):
        self._end_time = time.perf_counter()
        self._error = error

    @property
    def latency_ms(self) -> float:
        end = self._end_time or time.perf_counter()
        return round((end - self.start_time) * 1000, 2)

    def to_dict(self) -> dict:
        d = {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "span": self.name,
            "latency_ms": self.latency_ms,
            "wall_start": self.wall_start,
        }
        if self.parent_id:
            d["parent_id"] = self.parent_id
        if self._error:
            d["error"] = self._error
        d.update(self._meta)
        return d


class Tracer:
    def __init__(self):
        self._lock = threading.Lock()
        self._local = threading.local()
        self._completed: list[Span] = []
        self._current_trace_id: str = uuid.uuid4().hex[:12]
        self._file_handle = None

    def _ensure_file(self):
        if self._file_handle is None:
            try:
                self._file_handle = open(_TRACE_FILE, "a", encoding="utf-8")
            except Exception as e:
                log.warning(f"Could not open trace file {_TRACE_FILE}: {e}")

    def new_trace(self) -> str:
        self._current_trace_id = uuid.uuid4().hex[:12]
        # clear the span stack for this thread
        self._local.stack = []
        return self._current_trace_id

    @property
    def _stack(self) -> list[Span]:
        if not hasattr(self._local, "stack"):
            self._local.stack = []
        return self._local.stack

    @contextmanager
    def span(self, name: str, **meta):
        if not _ENABLED:
            yield _NoopSpan()
            return

        parent_id = self._stack[-1].span_id if self._stack else None
        s = Span(name, self._current_trace_id, parent_id=parent_id, **meta)
        self._stack.append(s)

        try:
            yield s
            s.finish()
        except Exception as e:
            s.finish(error=str(e))
            raise
        finally:
            if self._stack and self._stack[-1] is s:
                self._stack.pop()
            self._record(s)

    def _record(self, span: Span):
        with self._lock:
            self._completed.append(span)

        if _PRINT:
            depth = 0  # we've already popped from stack
            indent = "  " * depth
            status = f"ERROR: {span._error}" if span._error else "ok"
            print(
                f"[trace] {span.trace_id}/{span.span_id} {span.name} "
                f"{span.latency_ms}ms {status}",
                file=sys.stderr,
            )

    def flush(self, clear: bool = True) -> list[dict]:
        with self._lock:
            spans = list(self._completed)
            if clear:
                self._completed.clear()

        if not spans:
            return []

        self._ensure_file()
        records = [s.to_dict() for s in spans]

        if self._file_handle:
            for rec in records:
                try:
                    self._file_handle.write(json.dumps(rec) + "\n")
                except Exception as e:
                    log.warning(f"Failed to write trace: {e}")
            try:
                self._file_handle.flush()
            except Exception:
                pass

        return records

    def print_summary(self, min_ms: float = 0.0):
        with self._lock:
            spans = list(self._completed)

        if not spans:
            print("No spans recorded")
            return

        total_ms = sum(s.latency_ms for s in spans if s.parent_id is None)
        print(f"\n{'─' * 50}")
        print(f"  Trace: {self._current_trace_id}  |  Total: {total_ms:.1f}ms")
        print(f"{'─' * 50}")

        for s in sorted(spans, key=lambda x: x.start_time):
            if s.latency_ms < min_ms:
                continue
            pct = (s.latency_ms / total_ms * 100) if total_ms > 0 else 0
            indent = "    " if s.parent_id else "  "
            tokens = s._meta.get("total_tokens", "")
            tokens_str = f"  [{tokens} tok]" if tokens else ""
            err_str = f"  !! {s._error}" if s._error else ""
            print(f"{indent}{s.name:<30} {s.latency_ms:>8.1f}ms  ({pct:4.1f}%){tokens_str}{err_str}")
        print()

    def close(self):
        self.flush()
        if self._file_handle:
            try:
                self._file_handle.close()
            except Exception:
                pass
            self._file_handle = None


class _NoopSpan:
    """Returned when tracing is disabled — all ops are no-ops"""
    def set(self, **kwargs): return self
    def set_tokens(self, **kwargs): return self
    def finish(self, **kwargs): pass
    @property
    def latency_ms(self): return 0.0


# module-level singleton — import and use directly
tracer = Tracer()


def traced(span_name: str | None = None, **default_meta):
    """
    Decorator to wrap a function in a trace span.

    @traced("embedding")
    def embed(texts):
        ...

    @traced()  # uses function name
    def rerank(chunks, query):
        ...
    """
    def decorator(fn: Callable) -> Callable:
        name = span_name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with tracer.span(name, **default_meta):
                return fn(*args, **kwargs)

        return wrapper
    return decorator


# demo / test
if __name__ == "__main__":
    import random

    os.environ["TRACING_PRINT"] = "true"
    tracer.new_trace()

    print("Running demo RAG pipeline trace...\n")

    with tracer.span("rag_request", query="what is retrieval augmented generation?") as root:

        with tracer.span("query_expansion") as s:
            time.sleep(0.05)  # simulate API call
            s.set(n_variants=3)

        with tracer.span("retrieval") as s:
            time.sleep(random.uniform(0.02, 0.08))
            s.set(n_results=10, top_score=0.91, index="main_index")

        with tracer.span("rerank") as s:
            time.sleep(0.015)
            s.set(n_input=10, n_output=5)

        with tracer.span("context_packing") as s:
            time.sleep(0.003)
            s.set(tokens_used=3200, budget=4000)

        with tracer.span("llm_generation") as s:
            time.sleep(random.uniform(0.3, 0.8))
            s.set_tokens(input_tokens=3450, output_tokens=280)
            s.set(model="claude-sonnet-4-6", finish_reason="end_turn")

    tracer.print_summary()

    records = tracer.flush()
    print(f"Flushed {len(records)} spans to {_TRACE_FILE}")
    print("\nFirst span:")
    print(json.dumps(records[0], indent=2))
