"""
chunker.py
==========
Stage 1: Parse ATP 2-01.3 PDF and extract paragraph-level chunks.

Uses pdftotext (no -layout flag) for clean text extraction.
Paragraph IDs follow pattern X-Y. (e.g., "3-24. ") or
appendix patterns A-1. etc.

Output schema per line of chunks.jsonl:
{
    chunk_id, para_id, text, chapter_num, chapter_title,
    section, page_range, word_count,
    has_figure_reference, cross_references
}

Usage:
    python chunker.py --pdf ~/caimll_finetuning/ATP_2-01.3.pdf
    python chunker.py --pdf ~/caimll_finetuning/ATP_2-01.3.pdf --out data/chunks.jsonl
"""

import re
import json
import argparse
import subprocess
from pathlib import Path
from collections import Counter
from typing import Optional

# ── Chapter / Appendix Metadata ───────────────────────────────

ATP_CHAPTERS = {
    1: {"title": "IPB Fundamentals",                              "ipb_step": 0},
    2: {"title": "IPB Support to Planning",                       "ipb_step": 0},
    3: {"title": "Step 1\u2014Define the Operational Environment", "ipb_step": 1},
    4: {"title": "Step 2\u2014Describe Environmental Effects",     "ipb_step": 2},
    5: {"title": "Step 3\u2014Evaluate the Threat",               "ipb_step": 3},
    6: {"title": "Step 4\u2014Determine Threat COAs",             "ipb_step": 4},
    7: {"title": "Unified Action and Unique Environments",         "ipb_step": None},
    8: {"title": "Multi-Domain Considerations",                    "ipb_step": None},
}

ATP_APPENDICES = {
    "A": {"title": "Intelligence Products",            "ipb_step": None},
    "B": {"title": "Intelligence Synchronization",     "ipb_step": None},
    "C": {"title": "Threat Evaluation",                "ipb_step": 3},
    "D": {"title": "Course of Action Analysis",        "ipb_step": 4},
}

# Paragraph ID pattern — handles numeric chapters (1-1.) and
# appendices (A-1., B-12., etc.).  The handoff specifies this exact regex.
PARA_ID_RE   = re.compile(r'^(\d+)-(\d+)\.\s+',   re.MULTILINE)
APPEND_ID_RE = re.compile(r'^([A-D])-(\d+)\.\s+', re.MULTILINE)

# Cross-reference detection within text
CROSS_REF_RE = re.compile(r'\bparagraph[s]?\s+(\d+-\d+)', re.IGNORECASE)
FIGURE_RE    = re.compile(r'\bfigure\s+\d+-\d+',           re.IGNORECASE)

NOISE_PATTERNS = [
    re.compile(r'ATP 2-01\.3\s+\d{1,2}\s+\w+\s+\d{4}'),
    re.compile(r'\d{1,2}\s+\w+\s+\d{4}\s+ATP 2-01\.3'),
    re.compile(r'^Figure \d+[-\u2013]\d+\..*',    re.MULTILINE),
    re.compile(r'^Table \d+[-\u2013]\d+\..*',     re.MULTILINE),
    re.compile(r'\(See figure \d+[-\u2013]\d+\.\)'),
    re.compile(r'\(See table \d+[-\u2013]\d+\.\)'),
    re.compile(r'Intentionally left blank', re.IGNORECASE),
]

MAX_CHUNK_WORDS = 500
MIN_CHUNK_WORDS = 20


# ── Extraction ────────────────────────────────────────────────

def extract_text(pdf_path: str) -> str:
    """Run pdftotext (no -layout flag) and return raw text."""
    print(f"[chunker] Extracting text via pdftotext: {pdf_path}")
    result = subprocess.run(
        ["pdftotext", pdf_path, "-"],
        capture_output=True, text=True, timeout=180
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"pdftotext failed: {result.stderr}")
    words = len(result.stdout.split())
    lines = result.stdout.count('\n')
    print(f"[chunker] Extracted {words:,} words across {lines:,} lines")
    return result.stdout


def strip_noise(text: str) -> str:
    for pat in NOISE_PATTERNS:
        text = pat.sub('', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ── Chapter / section detection ───────────────────────────────

def _chapter_from_prefix(prefix: str) -> Optional[int]:
    """Return int chapter number if prefix is a digit 1-8, else None."""
    try:
        ch = int(prefix)
        return ch if 1 <= ch <= 8 else None
    except ValueError:
        return None


def _appendix_from_prefix(prefix: str) -> Optional[str]:
    return prefix if prefix in ATP_APPENDICES else None


def _detect_section(lines_before: list[str]) -> str:
    """Walk backwards to find the most recent all-caps heading."""
    for line in reversed(lines_before):
        stripped = line.strip()
        if (len(stripped) > 5
                and stripped.isupper()
                and not re.match(r'^ATP\s+2-01', stripped)
                and not stripped.isdigit()):
            return stripped
    return "General"


def _extract_cross_refs(text: str) -> list[str]:
    return CROSS_REF_RE.findall(text)


# ── Core chunking ─────────────────────────────────────────────

def chunk_paragraphs(text: str) -> list[dict]:
    """
    Split text into paragraph chunks keyed by paragraph ID.
    Handles both numeric chapters and appendix prefixes.
    Returns list of chunk dicts.
    """
    lines        = text.split('\n')
    chunks       = []
    chunk_id_ctr = 0

    cur_para_id  = None
    cur_chapter  = None    # int or "A"/"B"/... string
    cur_section  = "General"
    cur_lines    = []

    def flush():
        nonlocal chunk_id_ctr, cur_lines, cur_para_id, cur_chapter
        if not cur_lines or cur_para_id is None or cur_chapter is None:
            return
        body     = '\n'.join(cur_lines).strip()
        wc       = len(body.split())
        if wc < MIN_CHUNK_WORDS:
            return

        # Resolve chapter metadata
        if isinstance(cur_chapter, int):
            ch_info  = ATP_CHAPTERS.get(cur_chapter, {})
            ch_title = ch_info.get("title", f"Chapter {cur_chapter}")
            ch_num   = cur_chapter
        else:
            ch_info  = ATP_APPENDICES.get(cur_chapter, {})
            ch_title = ch_info.get("title", f"Appendix {cur_chapter}")
            ch_num   = cur_chapter   # str like "A"

        chunks.append({
            "chunk_id":            chunk_id_ctr,
            "para_id":             cur_para_id,
            "text":                body,
            "chapter_num":         ch_num,
            "chapter_title":       ch_title,
            "section":             cur_section,
            "page_range":          None,     # populated post-hoc if needed
            "word_count":          wc,
            "has_figure_reference": bool(FIGURE_RE.search(body)),
            "cross_references":    _extract_cross_refs(body),
        })
        chunk_id_ctr += 1

    for i, line in enumerate(lines):
        # Try numeric paragraph match first
        m_num = PARA_ID_RE.match(line)
        m_app = APPEND_ID_RE.match(line) if not m_num else None

        if m_num or m_app:
            flush()
            cur_lines = []
            if m_num:
                prefix       = m_num.group(1)
                number       = m_num.group(2)
                cur_para_id  = f"{prefix}-{number}"
                cur_chapter  = _chapter_from_prefix(prefix)
            else:
                prefix       = m_app.group(1)
                number       = m_app.group(2)
                cur_para_id  = f"{prefix}-{number}"
                cur_chapter  = _appendix_from_prefix(prefix)

            cur_section = _detect_section(lines[max(0, i - 25): i])

        # Update section from all-caps headings encountered mid-stream
        stripped = line.strip()
        if (len(stripped) > 5
                and stripped.isupper()
                and not re.match(r'^ATP\s+2-01', stripped)
                and not stripped.isdigit()):
            cur_section = stripped

        if cur_para_id is not None:
            cur_lines.append(line)
            # Split oversized chunks at paragraph boundaries
            if len(' '.join(cur_lines).split()) > MAX_CHUNK_WORDS:
                flush()
                cur_lines = []

    flush()
    return chunks


# ── Reporting ─────────────────────────────────────────────────

def print_report(chunks: list[dict]) -> None:
    ch_counts   = Counter(str(c["chapter_num"]) for c in chunks)
    total_words = sum(c["word_count"] for c in chunks)
    avg_words   = total_words // max(len(chunks), 1)
    fig_refs    = sum(1 for c in chunks if c["has_figure_reference"])

    print(f"\n[chunker] ── Chunk Report {'─'*40}")
    print(f"  Total chunks          : {len(chunks)}")
    print(f"  Total words           : {total_words:,}")
    print(f"  Avg words/chunk       : {avg_words}")
    print(f"  Chunks with fig refs  : {fig_refs}")
    print(f"\n  By chapter/appendix:")
    for ch in sorted(ch_counts, key=lambda x: (len(x), x)):
        print(f"    {ch}: {ch_counts[ch]} chunks")
    print()


# ── Entry point ───────────────────────────────────────────────

def run(pdf_path: str, out_path: str) -> list[dict]:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    raw    = extract_text(pdf_path)
    clean  = strip_noise(raw)
    chunks = chunk_paragraphs(clean)
    print_report(chunks)

    with open(out_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + '\n')
    print(f"[chunker] Saved {len(chunks)} chunks \u2192 {out_path}")
    return chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 1: Parse ATP 2-01.3 PDF into paragraph chunks"
    )
    parser.add_argument(
        "--pdf",
        default=str(Path.home() / "caimll_finetuning" / "ATP_2-01.3.pdf"),
        help="Path to ATP 2-01.3 PDF",
    )
    parser.add_argument(
        "--out",
        default="data/chunks.jsonl",
        help="Output JSONL path (default: data/chunks.jsonl)",
    )
    args = parser.parse_args()
    run(args.pdf, args.out)
