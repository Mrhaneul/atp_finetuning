"""
generator_appendix.py
=====================
Appendix-only variant of generator.py for ATP 2-01.3 Appendices A, B, C, D.

Key differences from generator.py:
- thinking_trace para-ID citation check is SKIPPED.  Appendices use section
  references like "A-1" rather than numbered paragraph IDs like "para 3-24".
- System prompt updated: rule 4 (cite para IDs) removed.
- Generation templates updated: "cite para IDs" hint removed from thinking_trace.
- Defaults: --chapters A,B,C  --machine-id mac  --target 300  --out data/seeds_mac.jsonl

Usage (Mac Studio):
    python generator_appendix.py \\
        --backend ollama --machine-id mac --seed 42 \\
        --chapters A,B,C --target 300 --out data/seeds_mac.jsonl
"""

import argparse
from typing import Optional

import generator as _gen

# ── Appendix system prompt (rule 4 removed) ───────────────────

GENERATION_SYSTEM = """\
You are an expert military intelligence educator generating high-quality training \
data for an ATP 2-01.3 (Intelligence Preparation of the Battlefield, March 2019) \
language model fine-tune.

ABSOLUTE RULES:
1. Ground every answer ONLY in the provided source text. No outside knowledge.
2. The answer MUST end with [Reference: ATP 2-01.3, para X-Y] — NEVER at the start.
3. NEVER begin the answer with "According to ATP 2-01.3" — put citations at the END.
4. thinking_trace must be ≤ 400 words. Be concise.
5. The question must be ≥ 20 words and specific enough that it cannot be answered \
   without knowing the doctrine.
6. Output valid JSON ONLY — no prose before or after.
"""

# ── Appendix templates ("cite para IDs" hint replaced) ────────

GENERATION_TEMPLATE = """\
SOURCE TEXT (Appendix {chapter_num}: {chapter_title}, Section {para_id}, Echelon: {echelon}):
\"\"\"
{text}
\"\"\"

TASK: {instruction}

Respond with this exact JSON structure:
{{
  "question": "<your question (≥20 words)>",
  "thinking_trace": "<internal reasoning about the appendix content, ≤400 words>",
  "answer": "<substantive answer (≥50 words), ending with [Reference: ATP 2-01.3, para {para_id}]>",
  "difficulty": "<basic|intermediate|advanced>",
  "citation_paragraphs": ["{para_id}"]
}}
"""

CONTRASTIVE_TEMPLATE = """\
SOURCE A (Appendix {chapter_num_a}: {chapter_title_a}, Section {para_id_a}, Echelon: {echelon_a}):
\"\"\"
{text_a}
\"\"\"

SOURCE B (Appendix {chapter_num_b}: {chapter_title_b}, Section {para_id_b}, Echelon: {echelon_b}):
\"\"\"
{text_b}
\"\"\"

TASK: {instruction}

Respond with this exact JSON structure:
{{
  "question": "<contrastive question comparing SOURCE A and SOURCE B (≥20 words)>",
  "thinking_trace": "<internal reasoning about both appendix sections, ≤400 words>",
  "answer": "<answer comparing both contexts (≥50 words), ending with [Reference: ATP 2-01.3, para {para_id_a} and {para_id_b}]>",
  "difficulty": "<basic|intermediate|advanced>",
  "citation_paragraphs": ["{para_id_a}", "{para_id_b}"]
}}
"""

# ── Appendix validation (para-ID check in thinking trace omitted) ─

def validate_qa(pair: dict, chunk_a: dict, chunk_b: Optional[dict] = None) -> tuple[bool, str]:
    """
    Identical to generator.validate_qa except the PARA_REF_RE thinking-trace
    check is skipped.  Appendix sections are identified by labels like A-1 or
    B-3, not by the "para N-N" format used in numbered chapters.
    """
    q   = pair.get("question",       "").strip()
    tt  = pair.get("thinking_trace", "").strip()
    ans = pair.get("answer",         "").strip()

    if not q:
        return False, "empty question"
    if not ans:
        return False, "empty answer"
    if len(q.split()) < _gen.MIN_QUESTION_WORDS:
        return False, f"question too short ({len(q.split())} words)"
    if len(ans.split()) < _gen.MIN_ANSWER_WORDS:
        return False, f"answer too short ({len(ans.split())} words)"
    if len(tt.split()) > _gen.MAX_THINKING_WORDS:
        return False, f"thinking trace too long ({len(tt.split())} words > {_gen.MAX_THINKING_WORDS})"
    if _gen.BAD_OPENING.match(ans):
        return False, "answer starts with forbidden 'According to ATP 2-01.3'"
    if not _gen.CITATION_RE.search(ans):
        return False, "missing [Reference: ATP 2-01.3, ...] at end of answer"
    if "SOURCE TEXT" in q.upper() or "SOURCE A" in q.upper():
        return False, "prompt leaked into question"
    return True, "ok"


# ── Patch generator so _process_task and _build_prompt pick up overrides ──

_gen.validate_qa          = validate_qa
_gen.GENERATION_SYSTEM    = GENERATION_SYSTEM
_gen.GENERATION_TEMPLATE  = GENERATION_TEMPLATE
_gen.CONTRASTIVE_TEMPLATE = CONTRASTIVE_TEMPLATE


# ── CLI ────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Appendix-only QA generation for ATP 2-01.3 (no para-ID citation filter)"
    )
    parser.add_argument("--enriched",   default="data/enriched.jsonl")
    parser.add_argument("--out",        default="data/seeds_mac.jsonl")
    parser.add_argument("--model",      default=_gen.OLLAMA_MODEL,
                        help="Ollama model name (ignored when --backend vllm)")
    parser.add_argument("--vllm-model", default=_gen.VLLM_MODEL,
                        help="HuggingFace model name for vLLM server")
    parser.add_argument("--target",     type=int, default=300,
                        help="Target number of QA pairs (default: 300)")
    parser.add_argument("--backend",    default="ollama", choices=["ollama", "vllm"])
    parser.add_argument("--api-url",    default=None,
                        help="Override base URL for Ollama or vLLM")
    parser.add_argument("--machine-id", default="mac",
                        help="Short identifier prepended to qa_ids (default: mac)")
    parser.add_argument("--seed",       type=int, default=_gen.RANDOM_SEED)
    parser.add_argument("--chapters",   default="A,B,C",
                        help="Appendix letters to process (default: A,B,C)")
    parser.add_argument("--workers",    type=int, default=1)
    args = parser.parse_args()

    _gen.OLLAMA_MODEL = args.model
    _gen.VLLM_MODEL   = args.vllm_model
    _gen.BACKEND      = args.backend
    _gen.MACHINE_ID   = args.machine_id

    if args.api_url:
        if _gen.BACKEND == "ollama":
            _gen.OLLAMA_URL = args.api_url.rstrip("/") + "/api/generate"
        else:
            _gen.VLLM_URL = args.api_url.rstrip("/")

    chapters_filter = _gen.parse_chapter_filter(args.chapters)
    _gen.run(args.enriched, args.out, args.target, seed=args.seed,
             chapters_filter=chapters_filter, workers=args.workers)
