"""
PDF → clean Markdown conversion with table detection.

Uses pdfplumber for text extraction + layout analysis. Handles:
  - multi-column layouts (heuristic column split)
  - tables → Markdown table syntax
  - headers/footers removal (page number lines, repeated text)
  - hyperlink extraction
  - basic heading detection (larger font size → # heading)

Standalone — requires: pdfplumber

Usage:
    python pdf_to_markdown.py paper.pdf
    python pdf_to_markdown.py paper.pdf --out paper.md --no-headers-footers
    python pdf_to_markdown.py *.pdf --out-dir markdown/

Limitation: doesn't handle scanned PDFs (no OCR). For those, use
docling or pymupdf4llm which have OCR integration.
"""

import re
import sys
import json
import logging
import argparse
from pathlib import Path
from collections import Counter
from typing import Any

log = logging.getLogger("pdf_to_md")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


def _import_pdfplumber():
    try:
        import pdfplumber
        return pdfplumber
    except ImportError:
        print("pdfplumber not installed. Run: pip install pdfplumber", file=sys.stderr)
        sys.exit(1)


def _table_to_markdown(table: list[list[str | None]]) -> str:
    """Convert pdfplumber table (list of rows) to markdown."""
    if not table:
        return ""

    rows = [[cell or "" for cell in row] for row in table]

    # try to detect header row (first row, or first row is bold — we can't tell easily,
    # so just always use first row as header for now)
    header = rows[0]
    body = rows[1:]

    def _escape(s: str) -> str:
        return s.replace("|", "\\|").replace("\n", " ").strip()

    header_md = "| " + " | ".join(_escape(h) for h in header) + " |"
    sep = "| " + " | ".join("---" for _ in header) + " |"
    body_md = "\n".join(
        "| " + " | ".join(_escape(c) for c in row) + " |"
        for row in body
        if any(c.strip() for c in row)
    )

    return header_md + "\n" + sep + "\n" + body_md


def _detect_columns(words: list[dict], page_width: float) -> int:
    """
    Rough heuristic: if there's a big gap in the horizontal word distribution,
    the page is multi-column. Returns 1 or 2.
    """
    if not words or page_width == 0:
        return 1

    mid = page_width / 2
    left_words = [w for w in words if w["x1"] < mid * 0.85]
    right_words = [w for w in words if w["x0"] > mid * 1.15]

    # if both sides have substantial text, it's two columns
    if len(left_words) > 20 and len(right_words) > 20:
        return 2
    return 1


def _is_header_footer(text: str, page_num: int, common_lines: set[str]) -> bool:
    """heuristic: page numbers, repeated lines across pages, short lone lines"""
    stripped = text.strip()
    if stripped in common_lines:
        return True
    if re.fullmatch(r"\d+", stripped):
        return True
    if re.fullmatch(r"[-–—]\s*\d+\s*[-–—]", stripped):
        return True
    return False


def _guess_heading_level(text: str, font_size: float, body_font_size: float) -> int | None:
    """Return heading level (1-3) or None if not a heading."""
    stripped = text.strip()
    if not stripped:
        return None
    # very long text is probably not a heading
    if len(stripped.split()) > 12:
        return None

    ratio = font_size / body_font_size if body_font_size > 0 else 1.0
    if ratio >= 1.5:
        return 1
    if ratio >= 1.2:
        return 2
    if ratio >= 1.1:
        return 3
    return None


def extract_page_text(page, body_font_size: float, skip_lines: set[str]) -> list[str]:
    """Extract text from a page as a list of markdown-ish lines."""
    lines = []

    words = page.extract_words(extra_attrs=["size"])
    n_cols = _detect_columns(words, float(page.width))

    if n_cols == 2:
        mid = float(page.width) / 2
        left_words = [w for w in words if w["x1"] < mid]
        right_words = [w for w in words if w["x0"] >= mid]
        # process left column then right column
        word_groups = [left_words, right_words]
    else:
        word_groups = [words]

    for word_group in word_groups:
        if not word_group:
            continue

        # group words into lines by top coordinate (within 2pt tolerance)
        by_line: dict[int, list[dict]] = {}
        for w in sorted(word_group, key=lambda x: (x["top"], x["x0"])):
            line_key = int(w["top"] / 2) * 2  # bucket by 2pt
            by_line.setdefault(line_key, []).append(w)

        for top_key in sorted(by_line):
            line_words = sorted(by_line[top_key], key=lambda w: w["x0"])
            line_text = " ".join(w["text"] for w in line_words).strip()

            if not line_text:
                continue

            if _is_header_footer(line_text, page.page_number, skip_lines):
                continue

            # estimate font size for this line
            avg_size = sum(w.get("size", 10) for w in line_words) / len(line_words)
            heading_level = _guess_heading_level(line_text, avg_size, body_font_size)

            if heading_level:
                lines.append(f"\n{'#' * heading_level} {line_text}\n")
            else:
                lines.append(line_text)

    return lines


def _find_common_lines(pdf, sample_pages: int = 10) -> set[str]:
    """Find text that appears on many pages (headers/footers)."""
    line_counter: Counter = Counter()
    pages = list(pdf.pages[:sample_pages])

    for page in pages:
        text = page.extract_text() or ""
        for line in text.splitlines():
            stripped = line.strip()
            if stripped and len(stripped) < 120:
                line_counter[stripped] += 1

    threshold = max(2, len(pages) // 2)
    return {line for line, count in line_counter.items() if count >= threshold}


def _estimate_body_font_size(pdf) -> float:
    """Look at font sizes across the first few pages, modal size = body text"""
    sizes: Counter = Counter()
    for page in pdf.pages[:5]:
        words = page.extract_words(extra_attrs=["size"])
        for w in words:
            s = round(float(w.get("size", 10)), 1)
            sizes[s] += 1
    if not sizes:
        return 10.0
    return sizes.most_common(1)[0][0]


def pdf_to_markdown(
    pdf_path: str | Path,
    remove_headers_footers: bool = True,
    extract_tables: bool = True,
) -> str:
    pdfplumber = _import_pdfplumber()
    p = Path(pdf_path)

    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {p}")

    with pdfplumber.open(str(p)) as pdf:
        log.info(f"Processing {p.name} ({len(pdf.pages)} pages)")

        skip_lines: set[str] = set()
        if remove_headers_footers:
            skip_lines = _find_common_lines(pdf)
            log.debug(f"Found {len(skip_lines)} common header/footer lines to skip")

        body_font_size = _estimate_body_font_size(pdf)
        log.debug(f"Estimated body font size: {body_font_size}pt")

        parts: list[str] = []

        for page in pdf.pages:
            page_parts: list[str] = []

            # extract tables first, replace them with placeholders in text extraction
            table_bbox_texts: list[tuple[tuple, str]] = []
            if extract_tables:
                tables = page.find_tables()
                for table_obj in tables:
                    table_data = table_obj.extract()
                    if table_data:
                        md_table = _table_to_markdown(table_data)
                        if md_table:
                            table_bbox_texts.append((table_obj.bbox, f"\n{md_table}\n"))

                # crop out table regions and process remaining text
                if table_bbox_texts:
                    # add table markdown
                    page_parts.extend(md for _, md in table_bbox_texts)

                    # get remaining text (outside table bboxes) — rough approach
                    # pdfplumber doesn't have easy bbox exclusion, so we just also
                    # get all text and it'll overlap a bit. acceptable for now.
                    # TODO: properly crop tables out before text extraction

            text_lines = extract_page_text(page, body_font_size, skip_lines)
            page_parts.extend(text_lines)

            if page_parts:
                parts.append("\n".join(page_parts))

    markdown = "\n\n".join(parts)

    # cleanup: collapse 3+ blank lines to 2
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    return markdown


def main():
    parser = argparse.ArgumentParser(description="Convert PDF to Markdown")
    parser.add_argument("pdfs", nargs="+", help="PDF file(s) to convert")
    parser.add_argument("--out", help="output .md file (single PDF only)")
    parser.add_argument("--out-dir", help="output directory (multiple PDFs)")
    parser.add_argument("--no-headers-footers", action="store_true",
                        help="keep header/footer text")
    parser.add_argument("--no-tables", action="store_true")
    args = parser.parse_args()

    if len(args.pdfs) > 1 and args.out:
        print("--out can only be used with a single PDF", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in args.pdfs:
        try:
            md = pdf_to_markdown(
                pdf_path,
                remove_headers_footers=not args.no_headers_footers,
                extract_tables=not args.no_tables,
            )
        except Exception as e:
            log.error(f"Failed to convert {pdf_path}: {e}")
            continue

        if args.out:
            out_path = args.out
        elif out_dir:
            out_path = out_dir / (Path(pdf_path).stem + ".md")
        else:
            out_path = Path(pdf_path).with_suffix(".md")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md)
        log.info(f"Saved {out_path} ({len(md)} chars)")


if __name__ == "__main__":
    main()
