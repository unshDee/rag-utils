"""
Structure-aware markdown chunker: split on headings, keep tables and code fences whole,
and carry the heading trail on every chunk.

Fixed-size splitters treat markdown as a flat string, and that breaks in the same three
places every time:
  - tables: the window cuts after row 6, so the next chunk is rows 7-14 with no header.
    "| enterprise | 5000 | 500 | 50 |" retrieves fine and tells the LLM nothing, it can't
    know which number is requests/min.
  - code fences: half a code block in one chunk, the other half (no opening fence) in the
    next, where it reads as prose.
  - sections: the last paragraph of "Authentication" and the first of "Rate limits" end
    up in one chunk, so the embedding is an average of two topics.

So parse the structure first and only then pack:
  - a heading always starts a new chunk. chunks never cross a section boundary.
  - tables and fenced code are atomic. if one is bigger than the budget it gets split by
    rows / lines, and every piece gets the header row (or the fence) repeated.
  - oversize paragraphs split on sentences, lists on items.
  - every chunk carries `headings`, the trail from the doc title down, and by default the
    trail is prepended to `text` so "It defaults to 60" still embeds near its section.
    cheap version of what contextual_retrieval.py does, no LLM.

Same idea as LangChain's MarkdownHeaderTextSplitter plus the table handling it doesn't do.
Works on pdf_to_markdown.py output. The `headings` key matches docx_to_chunks.py, so
`small_to_big.py --parent-key headings` works on these chunks as is.

Not handled: setext headings (=== / --- underlines), HTML tables, indented code blocks.
Tiny sections stay tiny chunks, run chunk_quality_scorer.py after if that bothers you.

Requires: stdlib only

Usage:
    python markdown_chunker.py --demo
    python markdown_chunker.py paper.md --out chunks.json
    python markdown_chunker.py docs/*.md --max-tokens 300 --no-breadcrumb

    from markdown_chunker import chunk_markdown
    chunks = chunk_markdown(md_text, source="guide.md", max_tokens=400)
    chunks[0]["headings"], chunks[0]["chunk_type"]   # ["Guide", "Install"], "table"
"""

import re
import json
import logging
import argparse
from itertools import groupby
from pathlib import Path
from typing import Any, NamedTuple

log = logging.getLogger("markdown_chunker")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# upper bound per chunk, breadcrumb included. 400 sits under the 512 limit of most
# embedding models with room for the ~4 chars/token estimate being off
DEFAULT_MAX_TOKENS = 400

# what goes between heading levels in the prepended trail. plain text on purpose, "#"
# markers in the trail would look like real headings if you ever re-chunk the output
BREADCRUMB_SEP = " > "

FENCE_RE = re.compile(r"^\s{0,3}(```+|~~~+)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
# the |---|:---:| line under a table header. this is what makes a table a table, a
# paragraph that happens to contain a pipe doesn't have one
TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$")
LIST_ITEM_RE = re.compile(r"^\s*([-*+]|\d+[.)])\s")
# naive splitter, same one as context_pruner.py
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def estimate_tokens(text: str) -> int:
    # same ~4 chars per token heuristic as context_window_packer.py
    return max(1, len(text) // 4)


class Block(NamedTuple):
    kind: str  # text | table | code
    text: str
    headings: tuple[str, ...]
    section: int  # bumps at every heading, so two sections with the same title stay apart


def parse_blocks(md: str) -> list[Block]:
    lines = md.splitlines()
    # yaml front matter, otherwise "title: foo" becomes the first chunk
    if lines and lines[0].strip() == "---":
        end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
        if end:
            lines = lines[end + 1:]

    blocks: list[Block] = []
    stack: list[tuple[int, str]] = []  # (level, title)
    para: list[str] = []
    section = 0

    def add(kind: str, text: str) -> None:
        blocks.append(Block(kind, text, tuple(t for _, t in stack), section))

    def flush_para() -> None:
        if para:
            add("text", "\n".join(para))
            para.clear()

    i = 0
    while i < len(lines):
        line = lines[i]

        # fences first. a "# install deps" comment inside a bash block is not a heading,
        # and checking headings first is how you get a section called "install deps"
        m = FENCE_RE.match(line)
        if m:
            flush_para()
            marker = m.group(1)
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith(marker):
                j += 1
            add("code", "\n".join(lines[i:j + 1]))  # unclosed fence just runs to EOF
            i = j + 1
            continue

        m = HEADING_RE.match(line)
        if m:
            flush_para()
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2)))
            section += 1
            i += 1
            continue

        if "|" in line and i + 1 < len(lines) and "|" in lines[i + 1] and TABLE_SEP_RE.match(lines[i + 1]):
            flush_para()
            j = i + 2
            while j < len(lines) and lines[j].strip() and "|" in lines[j]:
                j += 1
            add("table", "\n".join(lines[i:j]))
            i = j
            continue

        if line.strip():
            para.append(line)
        else:
            flush_para()
        i += 1

    flush_para()
    return blocks


def _pack(units: list[str], budget: int, joiner: str) -> list[str]:
    """greedy fill up to `budget` chars. a single unit over budget goes out on its own."""
    out, cur, size = [], [], 0
    for u in units:
        if cur and size + len(joiner) + len(u) > budget:
            out.append(joiner.join(cur))
            cur, size = [], 0
        size += len(u) + (len(joiner) if cur else 0)
        cur.append(u)
    if cur:
        out.append(joiner.join(cur))
    return out


def split_block(block: Block, budget: int) -> list[str]:
    """split one block into pieces of at most `budget` chars, each one valid on its own"""
    if len(block.text) <= budget:
        return [block.text]
    lines = block.text.split("\n")

    if block.kind == "table":
        # this is the bit that actually matters: header + separator on every piece
        head = "\n".join(lines[:2])
        return [f"{head}\n{rows}" for rows in _pack(lines[2:], budget - len(head) - 1, "\n")]

    if block.kind == "code":
        opener = lines[0]
        marker = FENCE_RE.match(opener).group(1)
        closed = len(lines) > 1 and lines[-1].strip().startswith(marker)
        body = lines[1:-1] if closed else lines[1:]
        room = budget - len(opener) - len(marker) - 2
        # keeps the info string (```python) so every piece still says what language it is
        return [f"{opener}\n{part}\n{marker}" for part in _pack(body, room, "\n")]

    if all(LIST_ITEM_RE.match(ln) or ln.startswith((" ", "\t")) for ln in lines):
        return _pack(lines, budget, "\n")
    return _pack(SENT_RE.split(block.text), budget, " ")


def chunk_markdown(
    md: str,
    source: str = "doc",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    breadcrumb: bool = True,
) -> list[dict[str, Any]]:
    stem = Path(source).stem
    chunks: list[dict[str, Any]] = []
    oversize = 0

    for _, group in groupby(parse_blocks(md), key=lambda b: b.section):
        blocks = list(group)
        headings = blocks[0].headings
        prefix = BREADCRUMB_SEP.join(headings) if breadcrumb else ""
        # the trail counts against the budget, but a silly long trail shouldn't leave
        # nothing for the body
        budget = max(max_tokens * 4 - (len(prefix) + 2 if prefix else 0), max_tokens * 2)

        pieces = [(b.kind, p) for b in blocks for p in split_block(b, budget)]
        cur: list[tuple[str, str]] = []
        size = 0

        def flush() -> None:
            nonlocal cur, size, oversize
            if not cur:
                return
            raw = "\n\n".join(p for _, p in cur)
            kinds = {k for k, _ in cur}
            text = f"{prefix}\n\n{raw}" if prefix else raw
            n = len(chunks)
            chunks.append({
                "chunk_id": f"{stem}#{n}",
                "chunk_index": n,
                "source": source,
                "text": text,
                "raw_text": raw,
                "headings": list(headings),
                "chunk_type": kinds.pop() if len(kinds) == 1 else "mixed",
                "n_tokens_est": estimate_tokens(text),
            })
            oversize += estimate_tokens(text) > max_tokens
            cur, size = [], 0

        for kind, piece in pieces:
            if cur and size + 2 + len(piece) > budget:
                flush()
            size += len(piece) + (2 if cur else 0)
            cur.append((kind, piece))
        flush()

    if oversize:
        # one sentence / table row / code line longer than the whole budget. nothing
        # sensible to cut on, so it's left alone
        log.warning(f"[{source}] {oversize} chunks over {max_tokens} tokens (unsplittable units)")
    return chunks


# --- demo ---

_PLANS = ["free", "hobby", "starter", "team", "growth", "business", "scale", "pro",
          "partner", "edu", "startup", "regulated", "enterprise", "dedicated"]


def _demo_doc() -> str:
    rows = "\n".join(
        f"| {p} | {60 * (i + 1) if p != 'enterprise' else 5000} | {10 * (i + 1)} | {i + 1} |"
        for i, p in enumerate(_PLANS)
    )
    return f"""---
title: Acme API guide
---
# Acme API guide

This guide covers authentication, rate limits and error handling for the Acme REST API.

## Authentication

Every request needs an API key in the Authorization header. Keys are scoped to a single
workspace and inherit the role of the user who created them. A key created by an admin
can manage billing, so don't put one of those in a CI job.

### Rotating keys

Rotate from the CLI. The old key stays valid for 24 hours so deploys don't break.

```bash
# rotate the key, old one stays valid for 24h
acme keys rotate --workspace prod
acme keys list --workspace prod
```

## Rate limits

Limits are counted per workspace, not per key, so making more keys doesn't help.

| plan | requests/min | burst | max payload (MB) |
|------|-------------:|------:|-----------------:|
{rows}

### What happens when you hit the limit

The API answers with status 429 and a Retry-After header. The header value is in seconds
and it is the time until the current window resets. Clients should sleep for that long
plus a little jitter. Retrying straight away counts against the next window too. Burst
capacity refills at one tenth of the per minute rate every six seconds. Requests rejected
with 429 are not billed. Three windows in a row over the limit flags the workspace for
review. Nothing gets suspended automatically. Support can raise a limit for a planned
load test if you ask a day ahead.

## Errors

- 400 means the payload failed validation, the body lists the fields
- 401 means the key is missing or revoked
- 429 means you hit a rate limit, see above
"""


def _naive(md: str, max_tokens: int) -> list[str]:
    size = max_tokens * 4
    return [md[i:i + size] for i in range(0, len(md), size)]


def _fence_broken(text: str) -> bool:
    return sum(1 for ln in text.splitlines() if FENCE_RE.match(ln)) % 2 == 1


def demo() -> None:
    md, max_tokens = _demo_doc(), 120
    header = "| plan | requests/min | burst | max payload (MB) |"
    rows = [ln for ln in md.splitlines() if ln.startswith("| ") and ln != header]

    naive = _naive(md, max_tokens)
    chunks = chunk_markdown(md, source="acme_api.md", max_tokens=max_tokens)

    def rows_with_header(texts: list[str]) -> int:
        # a row only means something if it's whole and its header is in the same chunk
        return sum(any(r in t and header in t for t in texts) for r in rows)

    ours = [c["raw_text"] for c in chunks]
    stats = [
        ("chunks", len(naive), len(chunks)),
        (f"table rows with their header (of {len(rows)})", rows_with_header(naive), rows_with_header(ours)),
        ("max tokens in a chunk", max(estimate_tokens(t) for t in naive), max(c["n_tokens_est"] for c in chunks)),
    ]
    print(f"\n  doc: {len(md)} chars, budget {max_tokens} tokens\n")
    print(f"  {'':<42} {'fixed-size':>10} {'markdown_chunker':>17}")
    for name, a, b in stats:
        print(f"  {name:<42} {a:>10} {b:>17}")

    row = next(r for r in rows if "enterprise" in r)
    print("\n  the chunk you'd retrieve for 'enterprise plan rate limit':")
    print("\n  fixed-size:")
    print("    " + next(t for t in naive if "| enterprise" in t).strip().replace("\n", "\n    "))
    hit = next(c for c in chunks if row in c["raw_text"])
    print(f"\n  markdown_chunker ({hit['chunk_id']}, type={hit['chunk_type']}):")
    print("    " + hit["text"].replace("\n", "\n    "))

    print("\n  all chunks:")
    for c in chunks:
        print(f"    {c['chunk_id']:<14} {c['chunk_type']:<6} {c['n_tokens_est']:>4} tok  "
              f"{BREADCRUMB_SEP.join(c['headings'])}")

    # self-checks
    assert rows_with_header(ours) == len(rows) and rows_with_header(naive) < len(rows)
    assert all(sum(r in t for t in ours) == 1 for r in rows), "a table row got lost or duplicated"
    assert sum(header in t for t in ours) > 1, "table should have been split at this budget"
    assert not any(map(_fence_broken, ours))
    assert all(c["n_tokens_est"] <= max_tokens for c in chunks)
    bash = next(c for c in chunks if "acme keys rotate" in c["raw_text"])
    assert bash["headings"] == ["Acme API guide", "Authentication", "Rotating keys"]
    assert not any("rotate the key" in h for c in chunks for h in c["headings"]), "code comment became a heading"
    assert not any("title: Acme" in c["text"] for c in chunks), "front matter leaked"
    assert hit["text"].startswith("Acme API guide > Rate limits\n\n")
    assert chunks[-1]["headings"] == ["Acme API guide", "Errors"]  # popped back up from the h3
    assert chunk_markdown("", "x.md") == []
    # a code block over budget: every piece keeps its fence and the language tag
    code = "```python\n" + "\n".join(f"x{i} = compute_something({i})" for i in range(40)) + "\n```"
    parts = chunk_markdown(code, "c.md", max_tokens=60)
    assert len(parts) > 1 and all(p["n_tokens_est"] <= 60 for p in parts)
    assert all(p["text"].startswith("```python\n") and p["text"].endswith("\n```") for p in parts)
    assert sum(p["text"].count("compute_something") for p in parts) == 40
    print("\n  self-checks passed")


def main():
    parser = argparse.ArgumentParser(description="Heading- and table-aware markdown chunker")
    parser.add_argument("files", nargs="*", help="markdown files to chunk")
    parser.add_argument("--out", default="markdown_chunks.json")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--no-breadcrumb", action="store_true",
                        help="don't prepend the heading trail to text (it's still in `headings`)")
    parser.add_argument("--demo", action="store_true", help="offline demo on a synthetic doc")
    args = parser.parse_args()

    if args.demo:
        demo()
        return
    if not args.files:
        parser.error("need at least one markdown file (or --demo)")

    all_chunks: list[dict[str, Any]] = []
    for path in args.files:
        text = Path(path).read_text(encoding="utf-8")
        chunks = chunk_markdown(text, source=path, max_tokens=args.max_tokens,
                                breadcrumb=not args.no_breadcrumb)
        log.info(f"[{path}] {len(text):,} chars -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    # chunk_ids come from the file stem, so docs/a/intro.md and docs/b/intro.md collide
    if len({c["chunk_id"] for c in all_chunks}) < len(all_chunks):
        log.warning("duplicate chunk_ids (two files share a name), rename or chunk them separately")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    sizes = sorted(c["n_tokens_est"] for c in all_chunks)
    if sizes:
        log.info(f"Saved {len(all_chunks)} chunks to {args.out}, token estimate: "
                 f"min {sizes[0]} / median {sizes[len(sizes) // 2]} / max {sizes[-1]}")


if __name__ == "__main__":
    main()
