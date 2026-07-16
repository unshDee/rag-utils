"""
Synthetic golden-dataset generation — turn a corpus into a labelled eval set.

Nobody hand-labels 200 queries, so most RAG systems ship with no evaluation and
"did that change help?" gets answered by vibes. This generates questions from known
chunks, which means the relevant_ids labels come for free.

Question types:
  simple       : answerable from one chunk
  multi_hop    : needs two chunks from the same document — breaks naive top-1 retrieval
  reasoning    : needs inference over the chunk, not string matching — kills lexical-only
  unanswerable : plausible, on-topic, not in the corpus. The only way to measure whether
                 the system abstains instead of confabulating. Most eval sets omit these.

Every generated question then goes through a critique pass — an independent call scoring
groundedness / standalone-ness / specificity — and anything below threshold is dropped.
Ungated synthetic questions are worse than no eval set, since they encode the generator's
confusion as ground truth.

Rejects are returned rather than discarded: a high rejection rate means the chunks
themselves are too fragmented to answer anything, and no retriever will fix that.

Requires: anthropic (pip install anthropic)
Set ANTHROPIC_API_KEY in env.

Output is JSONL that feeds straight into retrieval_eval.py:
  {"query": ..., "relevant_ids": [...], "answer": ..., "type": ...}

Usage:
    python eval_set_generator.py chunks.json --n 50 --out golden.jsonl
    python retrieval_eval.py golden.jsonl --k 1 5 10
"""

import os
import json
import random
import logging
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

log = logging.getLogger("eval_set_generator")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

DEFAULT_MODEL = "claude-sonnet-5"

# ~15% unanswerable is enough to catch a system that never abstains
DEFAULT_MIX = {"simple": 0.4, "multi_hop": 0.25, "reasoning": 0.2, "unanswerable": 0.15}

# critique thresholds (1-5). 4 is strict; drop to 3 if too much is being discarded.
MIN_GROUNDEDNESS = 4
MIN_STANDALONE = 4
MIN_SPECIFICITY = 3

TYPE_INSTRUCTIONS = {
    "simple": (
        "Write one factual question that is fully answerable from the passage below. "
        "It must be answerable WITHOUT seeing the passage — so name the entities "
        "explicitly instead of saying 'the document' or 'this system'."
    ),
    "multi_hop": (
        "Write one question that requires combining information from BOTH passages "
        "below. Answering from either passage alone must be impossible. Do not "
        "reference the passages themselves."
    ),
    "reasoning": (
        "Write one question that requires reasoning or inference over the passage — "
        "comparing, deducing a consequence, or applying a stated rule to a case. It "
        "must NOT be answerable by copying a sentence verbatim."
    ),
    "unanswerable": (
        "Write one question that is closely related to the topic of the passage, sounds "
        "entirely plausible to someone browsing this corpus, but is NOT answerable from "
        "the passage or anything it implies. It must be a realistic question a user would "
        "actually ask — not absurd, not off-topic."
    ),
}


def _get_client():
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY not set in environment")
    return anthropic.Anthropic()


def _ask_json(client, prompt: str, model: str, max_tokens: int = 1000) -> Any:
    resp = client.messages.create(
        model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}]
    )
    raw = "".join(b.text for b in resp.content if b.type == "text").strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw.strip())


def _generate(client, qtype: str, chunks: list[dict[str, Any]], model: str) -> dict[str, Any] | None:
    passages = "\n\n".join(
        f"<passage id='{c['chunk_id']}'>\n{c['text']}\n</passage>" for c in chunks
    )
    answer_rule = (
        '"answer": "the correct answer, grounded in the passage(s)"'
        if qtype != "unanswerable"
        else '"answer": "This is not answerable from the provided context."'
    )
    prompt = (
        f"{TYPE_INSTRUCTIONS[qtype]}\n\n{passages}\n\n"
        f'Output ONLY JSON: {{"question": "...", {answer_rule}}}'
    )

    try:
        result = _ask_json(client, prompt, model)
    except Exception as e:
        log.warning(f"Generation failed ({qtype}): {e}")
        return None

    if not result.get("question"):
        return None

    return {
        "query": result["question"].strip(),
        "answer": result.get("answer", "").strip(),
        "type": qtype,
        # unanswerable questions have no relevant chunks — that IS the label
        "relevant_ids": [] if qtype == "unanswerable" else [c["chunk_id"] for c in chunks],
        "source_chunks": [c["chunk_id"] for c in chunks],
    }


def _critique(
    client, sample: dict[str, Any], chunks: list[dict[str, Any]], model: str
) -> dict[str, Any]:
    """independent quality gate — attaches a `critique` field and a pass/fail"""
    passages = "\n\n".join(c["text"] for c in chunks)

    if sample["type"] == "unanswerable":
        prompt = (
            "This question was written to be UNANSWERABLE from the passage. Verify that.\n\n"
            f"PASSAGE:\n{passages}\n\nQUESTION: {sample['query']}\n\n"
            "Rate 1-5:\n"
            "- groundedness: 5 means genuinely NOT answerable from the passage; 1 means it "
            "actually is answerable (a bad unanswerable).\n"
            "- standalone: is it understandable without seeing the passage?\n"
            "- specificity: is it a real, specific question a user would ask (not vague or absurd)?\n\n"
            'Output ONLY JSON: {"groundedness": n, "standalone": n, "specificity": n, "reason": "..."}'
        )
    else:
        prompt = (
            "Rate this generated evaluation question 1-5 on each axis.\n\n"
            f"PASSAGE(S):\n{passages}\n\n"
            f"QUESTION: {sample['query']}\nPROPOSED ANSWER: {sample['answer']}\n\n"
            "- groundedness: can the question be answered unambiguously and completely from "
            "the passage(s)? (5 = yes, 1 = the passage doesn't really answer it)\n"
            "- standalone: is it understandable on its own, without the passage in front of "
            "you? References like 'the document', 'this method', 'the above' score 1.\n"
            "- specificity: is it a specific, useful question rather than a generic one like "
            "'what is discussed here'?\n\n"
            'Output ONLY JSON: {"groundedness": n, "standalone": n, "specificity": n, "reason": "..."}'
        )

    try:
        scores = _ask_json(client, prompt, model, max_tokens=500)
    except Exception as e:
        log.warning(f"Critique failed: {e}")
        scores = {"groundedness": 0, "standalone": 0, "specificity": 0, "reason": str(e)}

    sample["critique"] = scores
    sample["passed"] = (
        scores.get("groundedness", 0) >= MIN_GROUNDEDNESS
        and scores.get("standalone", 0) >= MIN_STANDALONE
        and scores.get("specificity", 0) >= MIN_SPECIFICITY
    )
    return sample


def generate_eval_set(
    chunks: list[dict[str, Any]],
    n: int = 50,
    model: str = DEFAULT_MODEL,
    mix: dict[str, float] | None = None,
    doc_key: str = "source",
    min_chunk_chars: int = 200,
    max_workers: int = 6,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """returns (kept, rejected)"""
    client = _get_client()
    rng = random.Random(seed)
    mix = mix or DEFAULT_MIX

    usable = [c for c in chunks if len((c.get("text") or "")) >= min_chunk_chars]
    if not usable:
        raise ValueError(f"No chunks with >= {min_chunk_chars} chars")
    log.info(f"{len(usable)}/{len(chunks)} chunks usable as question sources")

    by_doc: dict[str, list[dict]] = defaultdict(list)
    for c in usable:
        by_doc[str(c.get(doc_key, "__all__"))].append(c)

    # build the work list up front so the type mix is exact, not stochastic
    jobs: list[tuple[str, list[dict]]] = []
    for qtype, frac in mix.items():
        for _ in range(round(n * frac)):
            if qtype == "multi_hop":
                # two chunks from the same document, or the hop isn't a hop
                candidates = [d for d in by_doc.values() if len(d) >= 2]
                if not candidates:
                    log.warning("No document has 2+ chunks, skipping multi_hop")
                    continue
                doc = rng.choice(candidates)
                jobs.append((qtype, rng.sample(doc, 2)))
            else:
                jobs.append((qtype, [rng.choice(usable)]))

    log.info(f"Generating {len(jobs)} questions...")

    def work(job):
        qtype, source = job
        sample = _generate(client, qtype, source, model)
        return _critique(client, sample, source, model) if sample else None

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(work, j) for j in jobs]
        for fut in as_completed(futures):
            r = fut.result()
            if r:
                results.append(r)

    kept = [r for r in results if r["passed"]]
    rejected = [r for r in results if not r["passed"]]

    log.info(f"Kept {len(kept)}, rejected {len(rejected)} ({len(rejected)/max(len(results),1):.0%})")
    counts = defaultdict(int)
    for k in kept:
        counts[k["type"]] += 1
    log.info("Kept by type: " + ", ".join(f"{t}={c}" for t, c in sorted(counts.items())))

    return kept, rejected


def main():
    parser = argparse.ArgumentParser(description="Synthetic RAG eval-set generation")
    parser.add_argument("chunks", help="chunks JSON file")
    parser.add_argument("--n", type=int, default=50, help="target number of questions")
    parser.add_argument("--out", default="golden.jsonl")
    parser.add_argument("--rejected-out", help="also write the rejected questions here")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--doc-key", default="source")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-unanswerable", action="store_true",
                        help="drop the abstention questions")
    args = parser.parse_args()

    with open(args.chunks) as f:
        chunks = json.load(f)
    log.info(f"Loaded {len(chunks)} chunks from {args.chunks}")

    mix = dict(DEFAULT_MIX)
    if args.no_unanswerable:
        mix.pop("unanswerable")
        total = sum(mix.values())
        mix = {k: v / total for k, v in mix.items()}

    kept, rejected = generate_eval_set(
        chunks,
        n=args.n,
        model=args.model,
        mix=mix,
        doc_key=args.doc_key,
        max_workers=args.workers,
        seed=args.seed,
    )

    with open(args.out, "w") as f:
        for s in kept:
            f.write(json.dumps(s) + "\n")
    log.info(f"Saved {len(kept)} questions to {args.out}")

    if args.rejected_out and rejected:
        with open(args.rejected_out, "w") as f:
            for s in rejected:
                f.write(json.dumps(s) + "\n")
        log.info(f"Saved {len(rejected)} rejected questions to {args.rejected_out}")

    for s in kept[:3]:
        print(f"\n[{s['type']}] {s['query']}")
        print(f"  answer:   {s['answer'][:100]}")
        print(f"  relevant: {s['relevant_ids']}")


if __name__ == "__main__":
    main()
