"""
Scores RAG chunks on a handful of quality signals that tend to correlate
with retrieval performance. Useful for filtering junk chunks before indexing
or for debugging a chunking pipeline.

Signals:
  - token_count       : prefer chunks in a reasonable range
  - type_token_ratio  : proxy for information density (unique words / total)
  - stopword_ratio    : high stopword % → low information content
  - sentence_complete : does the chunk start/end like a real sentence?
  - heading_overlap   : does the chunk text repeat words from its heading?

Standalone — only needs: python stdlib + (optionally) a stopword list.
No sentence-transformers, no API calls.

Usage:
    python chunk_quality_scorer.py chunks_output.json
    python chunk_quality_scorer.py chunks_output.json --min-score 0.5 --out filtered.json
"""

import re
import sys
import json
import math
import argparse
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("chunk_scorer")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# rough english stopwords — good enough without nltk
_STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","are","was","were","be","been","being","have","has","had",
    "do","does","did","will","would","could","should","may","might","shall",
    "that","this","these","those","it","its","i","we","you","he","she","they",
    "as","if","then","than","so","not","no","nor","yet","both","either","each",
    "about","up","out","into","over","after","before","between","through",
}

# chunks shorter than this are likely header/footer noise
MIN_TOKENS = 20
# above this and you probably have a chunker misconfiguration
MAX_TOKENS = 600
IDEAL_RANGE = (60, 400)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z']+\b", text.lower())


def score_token_count(tokens: list[str]) -> float:
    n = len(tokens)
    if n < MIN_TOKENS:
        return 0.1
    if n > MAX_TOKENS:
        # penalize but don't zero out
        return max(0.2, 1.0 - (n - MAX_TOKENS) / MAX_TOKENS)
    lo, hi = IDEAL_RANGE
    if lo <= n <= hi:
        return 1.0
    if n < lo:
        return 0.5 + 0.5 * (n - MIN_TOKENS) / (lo - MIN_TOKENS)
    return 0.7  # a bit long but ok


def score_type_token_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    ttr = len(set(tokens)) / len(tokens)
    # ideal TTR for English prose is around 0.4-0.7
    # very low → repetitive boilerplate; very high → too short or jargon-heavy
    if ttr < 0.2:
        return 0.3
    if ttr > 0.85:
        return 0.6  # not necessarily bad, just unusual
    return min(1.0, ttr / 0.55)


def score_stopword_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    ratio = sum(1 for t in tokens if t in _STOPWORDS) / len(tokens)
    # natural prose: ~40-55% stopwords. Pure stopwords = navigation/header junk.
    if ratio > 0.75:
        return 0.2
    if ratio < 0.1:
        # almost no stopwords → probably a list of codes or table data, not prose
        return 0.5
    return 1.0 - abs(ratio - 0.45) / 0.45


def score_sentence_completeness(text: str) -> float:
    """
    Heuristic: good chunks start with a capital letter and end with punctuation.
    Doesn't catch everything but catches obvious truncation artifacts.
    """
    text = text.strip()
    if not text:
        return 0.0

    score = 0.5  # baseline
    if text and text[0].isupper():
        score += 0.25
    if text and text[-1] in ".!?:\"'":
        score += 0.25
    return score


def score_heading_overlap(text: str, headings: list[str]) -> float:
    """
    If chunk has headings metadata, check whether the chunk text actually
    contains words from those headings. Helps catch chunks that got mis-attached
    to a heading during hierarchical chunking.
    """
    if not headings:
        return 1.0  # no heading to compare against, neutral

    heading_words = set()
    for h in headings:
        heading_words.update(_tokenize(h))
    heading_words -= _STOPWORDS

    if not heading_words:
        return 1.0

    chunk_words = set(_tokenize(text))
    overlap = len(heading_words & chunk_words) / len(heading_words)
    # some overlap expected; zero overlap is suspicious
    if overlap == 0:
        return 0.3
    return min(1.0, 0.4 + overlap * 1.2)


def score_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    text = chunk.get("text") or chunk.get("raw_text") or ""
    headings = chunk.get("headings") or []
    tokens = _tokenize(text)

    scores = {
        "token_count":          score_token_count(tokens),
        "type_token_ratio":     score_type_token_ratio(tokens),
        "stopword_ratio":       score_stopword_ratio(tokens),
        "sentence_completeness": score_sentence_completeness(text),
        "heading_overlap":      score_heading_overlap(text, headings),
    }

    # weights — can tune these based on your corpus
    weights = {
        "token_count": 0.25,
        "type_token_ratio": 0.20,
        "stopword_ratio": 0.20,
        "sentence_completeness": 0.15,
        "heading_overlap": 0.20,
    }

    final = sum(scores[k] * weights[k] for k in scores)

    return {
        **chunk,
        "_quality": {
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "final_score": round(final, 4),
            "token_count": len(tokens),
        }
    }


def score_all(chunks: list[dict]) -> list[dict]:
    scored = [score_chunk(c) for c in chunks]
    scores = [c["_quality"]["final_score"] for c in scored]

    if scores:
        avg = sum(scores) / len(scores)
        lo, hi = min(scores), max(scores)
        log.info(f"Scored {len(scored)} chunks. avg={avg:.3f} min={lo:.3f} max={hi:.3f}")

    return scored


def main():
    parser = argparse.ArgumentParser(description="Score RAG chunks by quality")
    parser.add_argument("input", help="path to chunks JSON file")
    parser.add_argument("--min-score", type=float, default=0.0,
                        help="filter out chunks below this score (0-1)")
    parser.add_argument("--out", default=None,
                        help="output path (default: <input>_scored.json)")
    parser.add_argument("--show-worst", type=int, default=5,
                        help="print N worst chunks for inspection")
    args = parser.parse_args()

    p = Path(args.input)
    if not p.exists():
        print(f"file not found: {p}", file=sys.stderr)
        sys.exit(1)

    with p.open() as f:
        chunks = json.load(f)

    scored = score_all(chunks)
    scored.sort(key=lambda c: c["_quality"]["final_score"])

    if args.show_worst > 0:
        print(f"\n--- {args.show_worst} lowest-scoring chunks ---")
        for c in scored[:args.show_worst]:
            q = c["_quality"]
            text_preview = (c.get("text") or "")[:120].replace("\n", " ")
            print(f"  score={q['final_score']:.3f}  tokens={q['token_count']}")
            print(f"  scores: {q['scores']}")
            print(f"  text: {text_preview!r}")
            print()

    if args.min_score > 0:
        before = len(scored)
        scored = [c for c in scored if c["_quality"]["final_score"] >= args.min_score]
        log.info(f"Filtered {before - len(scored)} chunks below score {args.min_score}. Kept {len(scored)}.")

    out_path = args.out or str(p.with_stem(p.stem + "_scored"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scored, f, ensure_ascii=False, indent=2)
    log.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
