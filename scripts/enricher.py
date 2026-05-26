"""
enricher.py
===========
Stage 2: Grounded metadata classification for each ATP 2-01.3 chunk.

Calls Ollama at temperature=0.1 (low — classification task) to classify
each chunk along doctrinal metadata axes. Results are appended to an
enriched.jsonl file.  Safe to interrupt and resume.

Input  : data/chunks.jsonl
Output : data/enriched.jsonl

Schema added per chunk:
    metadata: {
        ipb_step        : int | None   (0=fundamentals, 1-4=IPB steps, None=other)
        content_type    : str          (doctrine|procedure|example|reference|product)
        echelon         : str          (national|theater_army|corps|division|bct|battalion|general)
        domain          : str          (land|air|sea|space|cyber|multi|general)
        threat_type     : str          (conventional|unconventional|hybrid|general)
        ipb_product     : str | None   (modified_combined_obstacle_overlay|event_template|
                                        situation_template|threat_model|none)
        environment     : str          (competition|crisis|armed_conflict|lsco|general)
        doctrinal_weight: float        (1.0-3.0, from chapter weight table)
    }

CRITICAL: Temperature MUST be 0.1 for classification (not 0.7).
CRITICAL: ipb_step MUST match chapter number (e.g., chapter 3 => ipb_step=1).
CRITICAL: NEVER add beyond-manual content in metadata.

Usage:
    python enricher.py
    python enricher.py --chunks data/chunks.jsonl --out data/enriched.jsonl
    python enricher.py --chunks data/chunks.jsonl --out data/enriched.jsonl --max 500
"""

import re
import json
import argparse
import requests
from pathlib import Path
from typing import Optional

# ── Config ────────────────────────────────────────────────────

OLLAMA_MODEL = "gemma4:31b"
OLLAMA_URL   = "http://localhost:11434/api/generate"
CLASSIFY_TEMP = 0.1     # MUST stay at 0.1 — classification task

# Chapter → ipb_step ground truth (authoritative; overrides LLM output)
CHAPTER_STEP_MAP: dict[int, Optional[int]] = {
    1: 0,    # Fundamentals
    2: 0,    # IPB Support to Planning
    3: 1,    # Step 1 — Define OE
    4: 2,    # Step 2 — Describe Environmental Effects
    5: 3,    # Step 3 — Evaluate Threat
    6: 4,    # Step 4 — Determine Threat COAs
    7: None, # Unified Action / Unique Environments
    8: None, # Multi-Domain
}

APPENDIX_STEP_MAP: dict[str, Optional[int]] = {
    "A": None,
    "B": None,
    "C": 3,   # Appendix C aligns with Step 3
    "D": 4,   # Appendix D aligns with Step 4
}

# Chapter doctrinal weights (from handoff strategy doc)
CHAPTER_WEIGHTS: dict = {
    1: 2.5, 2: 2.0, 3: 2.0, 4: 2.0, 5: 2.5, 6: 2.5,
    7: 3.0, 8: 2.0,
    "A": 1.5, "B": 1.5, "C": 2.0, "D": 1.5,
}

VALID_CONTENT_TYPES  = {"doctrine", "procedure", "example", "reference", "product"}
VALID_ECHELONS       = {"national", "theater_army", "corps", "division", "bct", "battalion", "general"}
VALID_DOMAINS        = {"land", "air", "sea", "space", "cyber", "multi", "general"}
VALID_THREAT_TYPES   = {"conventional", "unconventional", "hybrid", "general"}
VALID_ENVIRONMENTS   = {"competition", "crisis", "armed_conflict", "lsco", "general"}
VALID_IPB_PRODUCTS   = {
    "modified_combined_obstacle_overlay", "event_template",
    "situation_template", "threat_model", "none",
}


# ── Ollama helpers ────────────────────────────────────────────

def coerce_str(val) -> str:
    if isinstance(val, str):   return val
    if isinstance(val, dict):  return json.dumps(val)
    if isinstance(val, list):  return " ".join(str(v) for v in val)
    return str(val)


def generate(prompt: str, system: str = "", temperature: float = CLASSIFY_TEMP) -> str:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model":  OLLAMA_MODEL,
                "prompt": prompt,
                "system": system,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_k":       64,
                    "top_p":       0.95,
                    "num_predict": 512,
                },
            },
            timeout=120,
        )
        result = response.json()
        return coerce_str(result.get("response", ""))
    except Exception as e:
        print(f"  [enricher] Ollama error: {e}")
        return ""


def extract_json(text: str) -> Optional[dict]:
    """Extract first valid JSON object from model response."""
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    m = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    m = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None


# ── Classification prompt ─────────────────────────────────────

CLASSIFY_SYSTEM = (
    "You are an Army doctrine analyst classifying ATP 2-01.3 text passages. "
    "Your job is ONLY to classify — do NOT add information beyond what the "
    "source text contains. Respond with valid JSON only, no prose."
)

CLASSIFY_TEMPLATE = """\
Classify this ATP 2-01.3 paragraph. Choose ONLY from the allowed values.

SOURCE TEXT:
\"\"\"
{text}
\"\"\"

Respond with JSON:
{{
  "content_type"  : "<doctrine|procedure|example|reference|product>",
  "echelon"       : "<national|theater_army|corps|division|bct|battalion|general>",
  "domain"        : "<land|air|sea|space|cyber|multi|general>",
  "threat_type"   : "<conventional|unconventional|hybrid|general>",
  "ipb_product"   : "<modified_combined_obstacle_overlay|event_template|situation_template|threat_model|none>",
  "environment"   : "<competition|crisis|armed_conflict|lsco|general>"
}}

Rules:
- Use "general" when the text does not specifically address that axis.
- "ipb_product" is "none" unless the text explicitly describes creating that product.
- Do NOT invent information not present in the source text.
"""


# ── Validation ────────────────────────────────────────────────

def _canonical_ipb_step(chunk: dict) -> Optional[int]:
    """Return authoritative ipb_step based on chapter number (overrides LLM)."""
    ch = chunk.get("chapter_num")
    if isinstance(ch, int):
        return CHAPTER_STEP_MAP.get(ch, None)
    if isinstance(ch, str):
        return APPENDIX_STEP_MAP.get(ch, None)
    return None


def _coerce_field(value: str, valid_set: set, default: str) -> str:
    if isinstance(value, str) and value.strip().lower() in valid_set:
        return value.strip().lower()
    return default


def validate_and_fix(meta: dict, chunk: dict) -> dict:
    """Clamp all fields to valid sets; override ipb_step from chapter map."""
    return {
        "ipb_step":         _canonical_ipb_step(chunk),
        "content_type":     _coerce_field(meta.get("content_type",  ""), VALID_CONTENT_TYPES,  "doctrine"),
        "echelon":          _coerce_field(meta.get("echelon",       ""), VALID_ECHELONS,        "general"),
        "domain":           _coerce_field(meta.get("domain",        ""), VALID_DOMAINS,         "general"),
        "threat_type":      _coerce_field(meta.get("threat_type",   ""), VALID_THREAT_TYPES,    "general"),
        "ipb_product":      _coerce_field(meta.get("ipb_product",   ""), VALID_IPB_PRODUCTS,    "none"),
        "environment":      _coerce_field(meta.get("environment",   ""), VALID_ENVIRONMENTS,    "general"),
        "doctrinal_weight": CHAPTER_WEIGHTS.get(chunk.get("chapter_num"), 1.5),
    }


# ── Fallback heuristic ────────────────────────────────────────

def heuristic_metadata(chunk: dict) -> dict:
    """
    Quick keyword-based classification used when Ollama fails.
    Ensures every chunk gets valid metadata even without LLM.
    """
    text  = chunk.get("text", "").lower()
    ch    = chunk.get("chapter_num")

    echelon = "general"
    for kw, ec in [
        ("theater army", "theater_army"), ("corps", "corps"),
        ("division", "division"),
        ("brigade combat team", "bct"), ("bct", "bct"),
        ("battalion", "battalion"),
    ]:
        if kw in text:
            echelon = ec
            break

    domain = "general"
    for kw, dm in [
        ("cyber", "cyber"), ("space", "space"), ("air", "air"),
        ("maritime", "sea"), ("sea", "sea"), ("multi-domain", "multi"),
    ]:
        if kw in text:
            domain = dm
            break

    env = "general"
    for kw, en in [
        ("large-scale combat", "lsco"), ("lsco", "lsco"),
        ("armed conflict", "armed_conflict"), ("crisis", "crisis"),
        ("competition", "competition"),
    ]:
        if kw in text:
            env = en
            break

    return {
        "ipb_step":         _canonical_ipb_step(chunk),
        "content_type":     "doctrine",
        "echelon":          echelon,
        "domain":           domain,
        "threat_type":      "general",
        "ipb_product":      "none",
        "environment":      env,
        "doctrinal_weight": CHAPTER_WEIGHTS.get(ch, 1.5),
    }


# ── Main loop ─────────────────────────────────────────────────

def run(chunks_path: str, out_path: str, max_chunks: Optional[int] = None) -> int:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Load chunks
    chunks = []
    with open(chunks_path, encoding='utf-8') as f:
        for line in f:
            try:
                chunks.append(json.loads(line))
            except Exception:
                pass
    print(f"[enricher] Loaded {len(chunks)} chunks")

    if max_chunks:
        chunks = chunks[:max_chunks]
        print(f"[enricher] Capped at {max_chunks} chunks (--max)")

    # Resume: track already-enriched chunk IDs
    done_ids: set = set()
    if Path(out_path).exists():
        with open(out_path, encoding='utf-8') as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_ids.add(r.get("chunk_id"))
                except Exception:
                    pass
        print(f"[enricher] Resuming — {len(done_ids)} already enriched")

    llm_ok = 0
    heuristic = 0

    with open(out_path, 'a', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            cid = chunk.get("chunk_id")
            if cid in done_ids:
                continue

            label = f"[{i+1}/{len(chunks)}] {chunk.get('para_id')} ch{chunk.get('chapter_num')}"
            print(f"  {label}", end=" ... ", flush=True)

            prompt = CLASSIFY_TEMPLATE.format(text=chunk["text"][:1000])
            raw    = generate(prompt, CLASSIFY_SYSTEM, CLASSIFY_TEMP)
            parsed = extract_json(raw) if raw else None

            if parsed:
                meta = validate_and_fix(parsed, chunk)
                llm_ok += 1
                print("OK")
            else:
                meta = heuristic_metadata(chunk)
                heuristic += 1
                print("HEURISTIC")

            enriched = {**chunk, "metadata": meta}
            f.write(json.dumps(enriched) + '\n')
            f.flush()
            done_ids.add(cid)

    total = llm_ok + heuristic
    print(f"\n[enricher] Done — {total} enriched "
          f"({llm_ok} LLM, {heuristic} heuristic)")
    print(f"[enricher] Output \u2192 {out_path}")
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 2: Enrich ATP 2-01.3 chunks with doctrinal metadata"
    )
    parser.add_argument("--chunks", default="data/chunks.jsonl")
    parser.add_argument("--out",    default="data/enriched.jsonl")
    parser.add_argument("--max",    type=int, default=None,
                        help="Cap number of chunks (for testing)")
    args = parser.parse_args()
    run(args.chunks, args.out, args.max)
