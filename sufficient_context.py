"""
Sufficient-context gate: decide whether the RAG system should answer at all.

Most RAG hallucinations aren't the generator ignoring good context — they're the
generator answering confidently from context that never contained the answer. Joren
et al. (Google, ICLR 2025 — https://arxiv.org/abs/2411.06037) split the problem with
one label per (query, context) pair:

  sufficient   : the context holds everything needed to give a definitive answer.
                 Judged WITHOUT a ground-truth answer, so it runs in production, not
                 just in eval. (The answer it supports may still be wrong — this is
                 about the context, not correctness.)

Their findings, which are the reason this script exists:
  - frontier models (Gemini, GPT, Claude) answer well when context is sufficient, but
    when it's insufficient they still answer instead of abstaining
  - retrieval quality alone doesn't fix it — a meaningful share of eval-set queries
    have insufficient context even with good retrievers
  - model self-confidence alone is a weak abstention signal. Combining it with the
    sufficiency label in a tiny logistic model ("selective generation") raised the
    accuracy of the answers the model chose to give by 2-10%

Three pieces:
  rate_sufficiency()  : LLM autorater, sufficient yes/no with a reason
  self_confidence()   : the generator's verbalized P(correct) for its own answer
  SelectiveGate       : logistic regression on [confidence, sufficient, interaction],
                        fit on a labeled eval set, thresholded for a target coverage.
                        Prints the coverage/accuracy curve against confidence alone,
                        so you can see whether the sufficiency signal earns its call.

It also prints the 2x2 diagnosis from the paper — sufficient x correct — which tells
you where to spend effort: insufficient+answered is a retrieval/abstention problem,
sufficient+wrong is a generator problem. Pairs with answer_eval.py (context_recall
splits blame the same way, but needs a ground truth answer).

Requires: anthropic (pip install anthropic)
Set ANTHROPIC_API_KEY in env.

Input JSONL, one object per line:
  {
    "query": "who founded the company?",
    "contexts": ["chunk text", ...],
    "answer": "the system's answer",
    "correct": true          <- optional; needed only to fit the gate
  }

Usage:
    python sufficient_context.py --demo                        # offline, no API calls
    python sufficient_context.py --generate-sample sample.jsonl
    python sufficient_context.py evals.jsonl --out rated.jsonl --fit gate.json --coverage 0.8
    python sufficient_context.py new.jsonl --gate gate.json    # apply a fitted gate

    from sufficient_context import rate_sufficiency, SelectiveGate
    gate = SelectiveGate.load("gate.json")
    if not gate.should_answer(confidence, sufficient): return "I don't know."
"""

import os
import json
import math
import random
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor
from typing import Any

log = logging.getLogger("sufficient_context")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# the autorater is a reading-comprehension call per query — Haiku is plenty, and in
# production it runs on every request. Use a stronger model to audit a sample of it.
DEFAULT_RATER = "claude-haiku-4-5"


def _get_client():
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY not set in environment")
    return anthropic.Anthropic()


def _ask_json(client, prompt: str, model: str, max_tokens: int = 800) -> Any:
    resp = client.messages.create(
        model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}]
    )
    raw = "".join(b.text for b in resp.content if b.type == "text").strip()

    # models sometimes wrap JSON in code fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw.strip())


def _format_contexts(contexts: list[str]) -> str:
    return "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))


def rate_sufficiency(client, query: str, contexts: list[str], model: str = DEFAULT_RATER) -> dict[str, Any]:
    """
    Autorater, following the paper's definition. Reason first, label second — the
    label is noticeably more accurate when the model has to list what the question
    needs before deciding whether the context has it.
    """
    prompt = (
        "You are judging whether a CONTEXT is sufficient to answer a QUESTION.\n\n"
        "Sufficient means: using only the context (plus basic common sense, no outside "
        "facts), a reader could give a definitive answer. Partial information, related "
        "topics, or information that requires guessing a missing link is NOT sufficient. "
        "If the question has a false premise or is ambiguous and the context resolves it, "
        "that is sufficient. You do not need to know the correct answer, and the context "
        "being wrong does not matter — only whether it contains the needed information.\n\n"
        "Steps: 1) list the pieces of information the question requires, 2) check each "
        "against the context, 3) decide.\n\n"
        f"QUESTION: {query}\n\nCONTEXT:\n{_format_contexts(contexts)}\n\n"
        'Output ONLY JSON: {"required": ["..."], "missing": ["..."], "sufficient": true|false}'
    )
    try:
        r = _ask_json(client, prompt, model)
        return {
            "sufficient": bool(r.get("sufficient")),
            "missing": r.get("missing", []),
        }
    except Exception as e:
        # unparseable rating -> treat as insufficient; failing closed means abstaining
        log.warning(f"Sufficiency rating failed ({e}), defaulting to insufficient")
        return {"sufficient": False, "missing": ["<rater error>"]}


def self_confidence(client, query: str, contexts: list[str], answer: str, model: str = DEFAULT_RATER) -> float:
    """
    Verbalized P(correct). Cheap and API-only; if you serve your own model, swap this
    for token-probability P(True) (Kadavath et al. 2022), which is better calibrated.
    """
    prompt = (
        "Here is a question, the context it was answered from, and a proposed answer.\n"
        "Estimate the probability (0.0-1.0) that the proposed answer is correct. Be "
        "calibrated: 0.9 should mean it is wrong about 1 time in 10.\n\n"
        f"QUESTION: {query}\n\nCONTEXT:\n{_format_contexts(contexts)}\n\n"
        f"PROPOSED ANSWER: {answer}\n\n"
        'Output ONLY JSON: {"p_correct": 0.0}'
    )
    try:
        p = float(_ask_json(client, prompt, model, max_tokens=100).get("p_correct", 0.0))
        return min(max(p, 0.0), 1.0)
    except Exception as e:
        log.warning(f"Confidence rating failed ({e}), defaulting to 0")
        return 0.0


def rate_dataset(samples: list[dict[str, Any]], model: str = DEFAULT_RATER, workers: int = 8) -> list[dict[str, Any]]:
    """adds _sufficient, _missing, _confidence to each sample"""
    client = _get_client()

    def work(s):
        suff = rate_sufficiency(client, s["query"], s.get("contexts", []), model)
        conf = self_confidence(client, s["query"], s.get("contexts", []), s.get("answer", ""), model)
        return {**s, "_sufficient": suff["sufficient"], "_missing": suff["missing"], "_confidence": conf}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rated = list(pool.map(work, samples))
    n_suff = sum(r["_sufficient"] for r in rated)
    log.info(f"Rated {len(rated)} samples: {n_suff} sufficient, {len(rated) - n_suff} insufficient")
    return rated


def _features(confidence: float, sufficient: bool) -> list[float]:
    s = 1.0 if sufficient else 0.0
    # interaction term: confidence means more when the context actually has the answer
    return [1.0, confidence, s, confidence * s]


def _sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-max(min(z, 30), -30)))


class SelectiveGate:
    """
    Tiny logistic regression predicting P(answer is correct) from confidence and the
    sufficiency label. Four weights, fit with plain gradient descent — at eval-set sizes
    (tens to hundreds of rows) anything fancier is overfitting with extra steps.
    """

    def __init__(self, weights: list[float] | None = None, threshold: float = 0.5):
        self.w = weights or [0.0, 1.0, 1.0, 0.0]
        self.threshold = threshold

    def score(self, confidence: float, sufficient: bool) -> float:
        x = _features(confidence, sufficient)
        return _sigmoid(sum(wi * xi for wi, xi in zip(self.w, x)))

    def should_answer(self, confidence: float, sufficient: bool) -> bool:
        return self.score(confidence, sufficient) >= self.threshold

    def fit(self, rows: list[tuple[float, bool, bool]], lr: float = 0.5, epochs: int = 2000, l2: float = 1e-3):
        """rows: (confidence, sufficient, correct)"""
        X = [_features(c, s) for c, s, _ in rows]
        y = [1.0 if ok else 0.0 for _, _, ok in rows]
        w = [0.0] * 4
        n = len(rows)
        for _ in range(epochs):
            grad = [l2 * wi for wi in w]
            for xi, yi in zip(X, y):
                err = _sigmoid(sum(a * b for a, b in zip(w, xi))) - yi
                for j in range(4):
                    grad[j] += err * xi[j] / n
            w = [wi - lr * g for wi, g in zip(w, grad)]
        self.w = w
        return self

    def set_coverage(self, rows: list[tuple[float, bool, bool]], coverage: float):
        """pick the threshold that answers `coverage` share of rows (abstains on the rest)"""
        scores = sorted((self.score(c, s) for c, s, _ in rows), reverse=True)
        k = max(1, min(len(scores), round(coverage * len(scores))))
        self.threshold = scores[k - 1]
        return self

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"weights": self.w, "threshold": self.threshold}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SelectiveGate":
        with open(path) as f:
            d = json.load(f)
        return cls(d["weights"], d["threshold"])


def coverage_curve(scores: list[float], correct: list[bool], points=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5)) -> dict[float, float]:
    """selective accuracy at each coverage: answer only the top-scored share, how many are right?"""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    out = {}
    for cov in points:
        k = max(1, round(cov * len(order)))
        out[cov] = sum(correct[i] for i in order[:k]) / k
    return out


def print_diagnosis(rows: list[tuple[float, bool, bool]], gate: SelectiveGate):
    n = len(rows)
    quad = {(s, ok): 0 for s in (True, False) for ok in (True, False)}
    for _, s, ok in rows:
        quad[(s, ok)] += 1

    print("\n" + "=" * 60)
    print(f"SUFFICIENT CONTEXT DIAGNOSIS  (n={n})")
    print("=" * 60)
    print(f"{'':<16}{'correct':>12}{'wrong':>12}")
    for s, name in ((True, "sufficient"), (False, "insufficient")):
        print(f"{name:<16}{quad[(s, True)]:>12}{quad[(s, False)]:>12}")
    print(
        f"\n  insufficient + wrong : {quad[(False, False)]/n:.0%}  <- should have abstained (retrieval gap / no abstention)\n"
        f"  sufficient + wrong   : {quad[(True, False)]/n:.0%}  <- generator failed to use good context\n"
        f"  insufficient + right : {quad[(False, True)]/n:.0%}  <- parametric knowledge or lucky guess"
    )

    correct = [ok for _, _, ok in rows]
    base = coverage_curve([c for c, _, _ in rows], correct)
    gated = coverage_curve([gate.score(c, s) for c, s, _ in rows], correct)
    print("\nselective accuracy (share correct among answered) by coverage:")
    print(f"  {'coverage':>9} {'confidence only':>16} {'+ sufficiency':>14}")
    for cov in base:
        print(f"  {cov:>9.0%} {base[cov]:>16.3f} {gated[cov]:>14.3f}")
    print(f"\ngate weights [bias, conf, suff, conf*suff]: {[round(w, 2) for w in gate.w]}")
    print(f"threshold: {gate.threshold:.3f}")
    print("=" * 60)


def _demo():
    """
    Offline: synthetic ratings shaped like the paper's finding — the model is about as
    confident on insufficient context as on sufficient, but much more often wrong there.
    Confidence alone can't separate those cases; the sufficiency label can.
    """
    rng = random.Random(0)
    rows = []
    for _ in range(400):
        suff = rng.random() < 0.6
        conf = min(1.0, max(0.0, rng.gauss(0.8, 0.12)))
        p_right = (0.35 + 0.6 * conf) if suff else (0.05 + 0.3 * conf)
        rows.append((conf, suff, rng.random() < p_right))

    gate = SelectiveGate().fit(rows).set_coverage(rows, 0.7)
    print_diagnosis(rows, gate)

    curve_base = coverage_curve([c for c, _, _ in rows], [ok for _, _, ok in rows])
    curve_gate = coverage_curve([gate.score(c, s) for c, s, _ in rows], [ok for _, _, ok in rows])
    assert curve_gate[0.6] > curve_base[0.6], "sufficiency signal should beat confidence alone here"
    assert gate.score(0.9, True) > gate.score(0.9, False)


SAMPLE = [
    {
        "query": "In what year was the company founded?",
        "contexts": ["Acme Corp was founded in 1998 in Austin by two former Dell engineers."],
        "answer": "1998",
        "correct": True,
    },
    {
        "query": "Who is the current CEO of Acme Corp?",
        "contexts": [
            "Acme Corp was founded in 1998 in Austin by two former Dell engineers.",
            "In 2015 the company moved its headquarters to Denver.",
        ],
        "answer": "The current CEO is one of the two founders.",
        "correct": False,
    },
    {
        "query": "What k value does RRF typically use?",
        "contexts": ["RRF scores documents by summing 1/(k + rank) across lists; k=60 is standard."],
        "answer": "k = 60",
        "correct": True,
    },
]


def main():
    parser = argparse.ArgumentParser(description="Sufficient-context autorater + selective generation gate")
    parser.add_argument("input", nargs="?", help="JSONL of query/contexts/answer[/correct]")
    parser.add_argument("--out", default="rated.jsonl", help="write rated samples here")
    parser.add_argument("--fit", metavar="GATE_JSON", help="fit a gate on labeled rows and save it")
    parser.add_argument("--gate", metavar="GATE_JSON", help="apply a fitted gate (adds _answer)")
    parser.add_argument("--coverage", type=float, default=0.8, help="target share of queries to answer")
    parser.add_argument("--model", default=DEFAULT_RATER)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--generate-sample", metavar="PATH")
    parser.add_argument("--demo", action="store_true", help="offline demo with synthetic ratings")
    args = parser.parse_args()

    if args.demo:
        _demo()
        return

    if args.generate_sample:
        with open(args.generate_sample, "w") as f:
            for s in SAMPLE:
                f.write(json.dumps(s) + "\n")
        print(f"Wrote {len(SAMPLE)} samples to {args.generate_sample}")
        return

    if not args.input:
        parser.print_help()
        return

    with open(args.input) as f:
        samples = [json.loads(line) for line in f if line.strip()]
    rated = rate_dataset(samples, model=args.model, workers=args.workers)

    if args.gate:
        gate = SelectiveGate.load(args.gate)
        for r in rated:
            r["_answer"] = gate.should_answer(r["_confidence"], r["_sufficient"])
        log.info(f"Gate answers {sum(r['_answer'] for r in rated)}/{len(rated)}")

    labeled = [(r["_confidence"], r["_sufficient"], bool(r["correct"])) for r in rated if "correct" in r]
    if labeled:
        gate = SelectiveGate.load(args.gate) if args.gate else SelectiveGate()
        if args.fit:
            gate.fit(labeled).set_coverage(labeled, args.coverage)
            gate.save(args.fit)
            log.info(f"Saved gate to {args.fit}")
        print_diagnosis(labeled, gate)
    elif args.fit:
        log.warning("--fit needs rows with a 'correct' field; nothing to fit")

    with open(args.out, "w") as f:
        for r in rated:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(f"Wrote {len(rated)} rated samples to {args.out}")


if __name__ == "__main__":
    main()
