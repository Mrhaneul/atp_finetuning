"""
generator.py
============
Stage 3: Synthetic QA generation with thinking traces for ATP 2-01.3.

Calls gemma4:31b via Ollama (temp=0.7) to generate structured QA pairs
with native Gemma 4 thinking traces.  Seeds file is written in append
mode so the run is safely resumable.

Input  : data/enriched.jsonl
Output : data/seeds.jsonl  (or --out for per-machine shards)

Output schema per line:
{
    qa_id           : str,
    source_chunks   : [para_id, ...],
    question_type   : str,
    question        : str,
    thinking_trace  : str,          # ≤ 400 words; must cite para IDs
    answer          : str,          # citation at END in [Reference: ATP 2-01.3, ...]
    metadata        : {
        ipb_step, content_type, echelon, domain,
        environment, difficulty, citation_paragraphs
    }
}

CRITICAL RULES (from handoff):
  1. Temperature MUST be 0.7 for generation (not classification).
  2. NEVER start answer with "According to ATP 2-01.3..." — citation goes at END.
  3. NEVER use "w" mode for seeds file — always append ("a").
  4. Thinking trace must cite specific para IDs (e.g., "para 3-24").
  5. Thinking trace must be ≤ 400 words; filter if > 500.
  6. Answer MUST end with [Reference: ATP 2-01.3, para X-Y].

Distributed usage (5-machine setup — chapter-based, zero overlap):
  # DGX Spark 1 — vLLM, chapters 7 and 8
  python generator.py --backend vllm --api-url http://localhost:8000 \\
      --machine-id spark1 --chapters 7,8 --target 2000 --out data/seeds_spark1.jsonl

  # DGX Spark 2 — vLLM, chapters 5 and 6
  python generator.py --backend vllm --api-url http://localhost:8000 \\
      --machine-id spark2 --chapters 5,6 --target 1500 --out data/seeds_spark2.jsonl

  # 4090 PC 1 — Ollama, chapters 1 and 4
  python generator.py --machine-id pc1 --chapters 1,4 --target 1000 \\
      --out data/seeds_pc1.jsonl

  # 4090 PC 2 — Ollama, chapters 2, 3, and appendix D
  python generator.py --machine-id pc2 --chapters 2,3,D --target 700 \\
      --out data/seeds_pc2.jsonl

  # Mac Studio — Ollama, appendices A, B, C
  python generator.py --machine-id mac --chapters A,B,C --target 300 \\
      --out data/seeds_mac.jsonl

  # After all machines finish — merge on any one machine:
  python merge_seeds.py

Single-machine usage (original):
    python generator.py
    python generator.py --enriched data/enriched.jsonl --out data/seeds.jsonl
    python generator.py --model gemma4:31b --target 5000
"""

import re
import json
import random
import argparse
import itertools
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# ── Config ────────────────────────────────────────────────────

OLLAMA_MODEL   = "gemma4:31b"
OLLAMA_URL     = "http://localhost:11434/api/generate"

# vLLM config (OpenAI-compatible endpoint)
VLLM_MODEL     = "google/gemma-4-31b-it"
VLLM_URL       = "http://localhost:8000"
VLLM_MAX_TOKENS = 1024

# Active backend: "ollama" or "vllm"
BACKEND        = "ollama"

# Machine identifier — prepended to qa_ids to avoid collisions in merged file.
# Set via --machine-id; default "m0" for single-machine runs.
MACHINE_ID     = "m0"

GEN_TEMP       = 0.7     # generation temperature
RANDOM_SEED    = 42
TARGET_PAIRS   = 5000

# Chapter doctrinal weights
CHAPTER_WEIGHTS: dict = {
    1: 2.5, 2: 2.0, 3: 2.0, 4: 2.0, 5: 2.5, 6: 2.5,
    7: 3.0, 8: 2.0,
    "A": 1.5, "B": 1.5, "C": 2.0, "D": 1.5,
}

# 9 question types from handoff; weights target known weaknesses
QUESTION_TYPES: dict[str, float] = {
    "factual":           1.0,
    "procedural":        1.5,
    "comparative":       1.5,
    "echelon_specific":  2.0,
    "applied_reasoning": 1.5,
    "product_generation":1.5,
    "cross_step":        1.5,
    "multi_domain":      1.5,
    "contrastive":       2.0,
}

QT_NAMES  = list(QUESTION_TYPES.keys())
QT_PROBS  = [QUESTION_TYPES[t] for t in QT_NAMES]

MAX_THINKING_WORDS = 500
MIN_QUESTION_WORDS = 20
MIN_ANSWER_WORDS   = 50

# Para ID reference pattern (must appear in thinking trace)
PARA_REF_RE   = re.compile(r'\bpara(?:graph)?\s+\d+-\d+', re.IGNORECASE)
# Citation block at end of answer
CITATION_RE   = re.compile(r'\[Reference:\s*ATP\s+2-01\.3', re.IGNORECASE)
# Forbidden opening
BAD_OPENING   = re.compile(r'^According to ATP', re.IGNORECASE)


# ── Ollama helpers ────────────────────────────────────────────

def coerce_str(val) -> str:
    if isinstance(val, str):   return val
    if isinstance(val, dict):  return json.dumps(val)
    if isinstance(val, list):  return " ".join(str(v) for v in val)
    return str(val)


def generate_ollama(prompt: str, system: str = "", temperature: float = GEN_TEMP) -> str:
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
                    "num_predict": 4096,
                },
            },
            timeout=300,
        )
        result = response.json()
        return coerce_str(result.get("response", ""))
    except Exception as e:
        print(f"  [generator] Ollama error: {e}")
        return ""


def generate_vllm(prompt: str, system: str = "", temperature: float = GEN_TEMP) -> str:
    """OpenAI-compatible chat/completions call for vLLM server."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model":       VLLM_MODEL,
        "messages":    messages,
        "max_tokens":  VLLM_MAX_TOKENS,
        "temperature": temperature,
        "top_p":       0.95,
    }
    try:
        response = requests.post(
            f"{VLLM_URL}/v1/chat/completions",
            json=payload,
            timeout=300,
        )
        try:
            result = response.json()
        except Exception:
            body = response.text.strip().replace("\n", " ")
            print(
                f"  [generator] vLLM non-JSON response "
                f"(HTTP {response.status_code}, prompt_chars={len(prompt)}, "
                f"max_tokens={VLLM_MAX_TOKENS}): {body[:800]}"
            )
            return ""
        if response.status_code != 200:
            print(
                f"  [generator] vLLM HTTP {response.status_code} "
                f"(prompt_chars={len(prompt)}, max_tokens={VLLM_MAX_TOKENS}): {result}"
            )
            return ""
        choices = result.get("choices") if isinstance(result, dict) else None
        if not isinstance(choices, list) or not choices:
            print(
                f"  [generator] vLLM unexpected response "
                f"(prompt_chars={len(prompt)}, max_tokens={VLLM_MAX_TOKENS}): {result}"
            )
            return ""
        message = choices[0].get("message", {})
        if not isinstance(message, dict) or "content" not in message:
            print(
                f"  [generator] vLLM malformed choice "
                f"(prompt_chars={len(prompt)}, max_tokens={VLLM_MAX_TOKENS}): {choices[0]}"
            )
            return ""
        return coerce_str(message["content"])
    except Exception as e:
        print(f"  [generator] vLLM error: {e}")
        return ""


def generate(prompt: str, system: str = "", temperature: float = GEN_TEMP) -> str:
    """Dispatch to the active backend (ollama or vllm)."""
    if BACKEND == "vllm":
        return generate_vllm(prompt, system, temperature)
    return generate_ollama(prompt, system, temperature)


def strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> or similar model-internal blocks."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'Thinking\.\.\..*?done thinking\.', '', text, flags=re.DOTALL)
    return text.strip()


def extract_json(text: str) -> Optional[dict]:
    text = strip_thinking_tags(text)
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
    # Grab largest {...} block
    matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}', text, re.DOTALL)
    for blk in sorted(matches, key=len, reverse=True):
        try:
            return json.loads(blk)
        except Exception:
            pass
    return None


# ── Question-type prompts ─────────────────────────────────────

TYPE_INSTRUCTIONS: dict[str, str] = {
    "factual": (
        "Generate a FACTUAL question testing recall of a specific definition, "
        "concept, or doctrine stated verbatim in the ATP 2-01.3 source text."
    ),
    "procedural": (
        "Generate a PROCEDURAL question asking HOW something is done — what steps, "
        "sequence, or actions ATP 2-01.3 prescribes for this topic."
    ),
    "comparative": (
        "Generate a COMPARATIVE question asking how two related concepts, steps, "
        "or products described in the source text DIFFER from each other."
    ),
    "echelon_specific": (
        "Generate an ECHELON-SPECIFIC question focused on the unique responsibilities, "
        "capabilities, or products at the echelon mentioned in the source text. "
        "The correct answer should be DIFFERENT at a different echelon."
    ),
    "applied_reasoning": (
        "Generate an APPLIED REASONING question presenting a realistic tactical/operational "
        "scenario and asking what ATP 2-01.3 prescribes the intelligence officer should do."
    ),
    "product_generation": (
        "Generate a PRODUCT GENERATION question asking what IPB product is developed, "
        "what it contains, and how it is produced — grounded in the source text."
    ),
    "cross_step": (
        "Generate a CROSS-STEP question asking how outputs of one IPB step feed "
        "into another step. Both steps must be referenced in the source text."
    ),
    "multi_domain": (
        "Generate a MULTI-DOMAIN question asking how IPB addresses threats or "
        "considerations in a non-land domain mentioned in the source text."
    ),
    "contrastive": (
        "Using BOTH source texts, generate a CONTRASTIVE question asking how "
        "a concept, role, or responsibility DIFFERS between the two contexts provided."
    ),
}

GENERATION_SYSTEM = """\
You are an expert military intelligence educator generating high-quality training \
data for an ATP 2-01.3 (Intelligence Preparation of the Battlefield, March 2019) \
language model fine-tune.

ABSOLUTE RULES:
1. Ground every answer ONLY in the provided source text. No outside knowledge.
2. The answer MUST end with [Reference: ATP 2-01.3, para X-Y] — NEVER at the start.
3. NEVER begin the answer with "According to ATP 2-01.3" — put citations at the END.
4. The thinking_trace MUST cite specific paragraph IDs like "para 3-24".
5. thinking_trace must be ≤ 400 words. Be concise.
6. The question must be ≥ 20 words and specific enough that it cannot be answered \
   without knowing the doctrine.
7. Output valid JSON ONLY — no prose before or after.
"""

GENERATION_TEMPLATE = """\
SOURCE TEXT (Chapter {chapter_num}: {chapter_title}, Para {para_id}, Echelon: {echelon}):
\"\"\"
{text}
\"\"\"

TASK: {instruction}

Respond with this exact JSON structure:
{{
  "question": "<your question (≥20 words)>",
  "thinking_trace": "<internal reasoning: cite para IDs, ≤400 words>",
  "answer": "<substantive answer (≥50 words), ending with [Reference: ATP 2-01.3, para {para_id}]>",
  "difficulty": "<basic|intermediate|advanced>",
  "citation_paragraphs": ["{para_id}"]
}}
"""

CONTRASTIVE_TEMPLATE = """\
SOURCE A (Chapter {chapter_num_a}: {chapter_title_a}, Para {para_id_a}, Echelon: {echelon_a}):
\"\"\"
{text_a}
\"\"\"

SOURCE B (Chapter {chapter_num_b}: {chapter_title_b}, Para {para_id_b}, Echelon: {echelon_b}):
\"\"\"
{text_b}
\"\"\"

TASK: {instruction}

Respond with this exact JSON structure:
{{
  "question": "<contrastive question comparing SOURCE A and SOURCE B (≥20 words)>",
  "thinking_trace": "<internal reasoning citing para IDs from both sources, ≤400 words>",
  "answer": "<answer comparing both contexts (≥50 words), ending with [Reference: ATP 2-01.3, para {para_id_a} and {para_id_b}]>",
  "difficulty": "<basic|intermediate|advanced>",
  "citation_paragraphs": ["{para_id_a}", "{para_id_b}"]
}}
"""


# ── Weighted task scheduling ──────────────────────────────────

def build_task_list(chunks: list[dict], target: int, seed: int = RANDOM_SEED) -> list[tuple]:
    """
    Build (chunk_a, question_type, chunk_b_or_None) task triples.

    Chapter weights oversample high-value sections.  The pool is cycled so
    target can exceed the raw pool size (required when enriched.jsonl has
    fewer chunks than the generation target).

    Each machine should pass a different `seed` so that random.choices picks
    a different task ordering — this is the primary distribution mechanism
    in the 5-machine setup.
    """
    rng = random.Random(seed)

    # Index by chapter and echelon for contrastive pairing
    by_chapter: dict = {}
    for c in chunks:
        ch = c.get("chapter_num")
        by_chapter.setdefault(ch, []).append(c)

    # Expand pool by weight
    pool: list[dict] = []
    for c in chunks:
        ch     = c.get("chapter_num")
        weight = CHAPTER_WEIGHTS.get(ch, 1.5)
        reps   = max(1, round(weight))
        pool.extend([c] * reps)
    rng.shuffle(pool)

    tasks = []
    # itertools.cycle lets us loop through pool as many times as needed
    # so target > len(pool) works correctly.
    for chunk in itertools.cycle(pool):
        if len(tasks) >= target:
            break

        qt = rng.choices(QT_NAMES, weights=QT_PROBS, k=1)[0]

        chunk_b = None
        if qt == "contrastive":
            # Find chunk from same chapter with different echelon
            same_ch    = by_chapter.get(chunk.get("chapter_num"), [])
            candidates = [
                c for c in same_ch
                if c.get("metadata", {}).get("echelon") != chunk.get("metadata", {}).get("echelon")
                and c.get("para_id") != chunk.get("para_id")
            ]
            if candidates:
                chunk_b = rng.choice(candidates)
            else:
                qt = "echelon_specific"   # fallback

        tasks.append((chunk, qt, chunk_b))

    return tasks


# ── QA validation ─────────────────────────────────────────────

def validate_qa(pair: dict, chunk_a: dict, chunk_b: Optional[dict] = None) -> tuple[bool, str]:
    """Return (passes, reason)."""
    q    = pair.get("question",      "").strip()
    tt   = pair.get("thinking_trace","").strip()
    ans  = pair.get("answer",        "").strip()

    if not q:
        return False, "empty question"
    if not ans:
        return False, "empty answer"
    if len(q.split()) < MIN_QUESTION_WORDS:
        return False, f"question too short ({len(q.split())} words)"
    if len(ans.split()) < MIN_ANSWER_WORDS:
        return False, f"answer too short ({len(ans.split())} words)"
    if len(tt.split()) > MAX_THINKING_WORDS:
        return False, f"thinking trace too long ({len(tt.split())} words > {MAX_THINKING_WORDS})"
    if BAD_OPENING.match(ans):
        return False, "answer starts with forbidden 'According to ATP 2-01.3'"
    if not CITATION_RE.search(ans):
        return False, "missing [Reference: ATP 2-01.3, ...] at end of answer"
    if not PARA_REF_RE.search(tt):
        return False, "thinking trace does not cite a paragraph ID"
    # Check for prompt leakage
    if "SOURCE TEXT" in q.upper() or "SOURCE A" in q.upper():
        return False, "prompt leaked into question"
    return True, "ok"


# ── Main generation loop ──────────────────────────────────────

def parse_chapter_filter(chapters_str: str) -> Optional[set]:
    """
    Parse comma-separated chapter IDs into a set for filtering.
    Integers and appendix letters are both supported.
    E.g. "7,8" → {7, 8};  "A,B,C" → {"A","B","C"};  "2,3,D" → {2, 3, "D"}
    Returns None if the string is empty (no filter — use all chapters).
    """
    if not chapters_str:
        return None
    result = set()
    for part in chapters_str.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            result.add(int(part))
        except ValueError:
            result.add(part.upper())
    return result if result else None


ATP_CHAPTERS_TITLES = {
    1: "IPB Fundamentals", 2: "IPB Support to Planning",
    3: "Step 1\u2014Define OE", 4: "Step 2\u2014Describe Environmental Effects",
    5: "Step 3\u2014Evaluate Threat", 6: "Step 4\u2014Determine Threat COAs",
    7: "Unified Action/Unique Environments", 8: "Multi-Domain Considerations",
}


def _build_prompt(chunk_a: dict, qt: str, chunk_b: Optional[dict]) -> tuple[str, list[str]]:
    """Build the generation prompt for a single task. Returns (prompt, source_chunks)."""
    pid_a    = chunk_a.get("para_id", "")
    ch_a     = chunk_a.get("chapter_num")
    meta_a   = chunk_a.get("metadata", {})
    ch_title_a = chunk_a.get("chapter_title") or ATP_CHAPTERS_TITLES.get(ch_a, f"Chapter {ch_a}")

    if qt == "contrastive" and chunk_b:
        pid_b      = chunk_b.get("para_id", "")
        ch_b       = chunk_b.get("chapter_num")
        ch_title_b = chunk_b.get("chapter_title") or ATP_CHAPTERS_TITLES.get(ch_b, f"Chapter {ch_b}")
        prompt = CONTRASTIVE_TEMPLATE.format(
            chapter_num_a  = ch_a,
            chapter_title_a= ch_title_a,
            para_id_a      = pid_a,
            echelon_a      = meta_a.get("echelon", "general"),
            text_a         = chunk_a.get("text", "")[:600],
            chapter_num_b  = ch_b,
            chapter_title_b= ch_title_b,
            para_id_b      = pid_b,
            echelon_b      = chunk_b.get("metadata", {}).get("echelon", "general"),
            text_b         = chunk_b.get("text", "")[:600],
            instruction    = TYPE_INSTRUCTIONS[qt],
        )
        return prompt, [pid_a, pid_b]
    else:
        prompt = GENERATION_TEMPLATE.format(
            chapter_num  = ch_a,
            chapter_title= ch_title_a,
            para_id      = pid_a,
            echelon      = meta_a.get("echelon", "general"),
            text         = chunk_a.get("text", "")[:800],
            instruction  = TYPE_INSTRUCTIONS[qt],
        )
        return prompt, [pid_a]


def _process_task(task_args: tuple) -> Optional[tuple]:
    """
    Execute one generation task in a thread.
    Returns (resume_key, seed_record, label) on success, None on failure.
    Pure function — reads only module-level globals (BACKEND, MACHINE_ID, etc.).
    """
    i, total, chunk_a, qt, chunk_b, resume_key, done_count = task_args

    pid_a  = chunk_a.get("para_id", "")
    label  = f"[{i+1}/{total}] {pid_a} {qt}"

    prompt, source_chunks = _build_prompt(chunk_a, qt, chunk_b)
    raw  = generate(prompt, GENERATION_SYSTEM, GEN_TEMP)
    pair = extract_json(raw) if raw else None

    if pair is None:
        print(f"  {label} ... FAILED (no JSON)", flush=True)
        return None

    passes, reason = validate_qa(pair, chunk_a, chunk_b)
    if not passes:
        print(f"  {label} ... FILTERED ({reason})", flush=True)
        return None

    meta_a = chunk_a.get("metadata", {})
    seed_record = {
        "qa_id":          None,          # assigned under lock in run()
        "source_chunks":  source_chunks,
        "question_type":  qt,
        "question":       pair["question"].strip(),
        "thinking_trace": pair.get("thinking_trace", "").strip(),
        "answer":         pair["answer"].strip(),
        "metadata": {
            "ipb_step":           meta_a.get("ipb_step"),
            "content_type":       meta_a.get("content_type", "doctrine"),
            "echelon":            meta_a.get("echelon", "general"),
            "domain":             meta_a.get("domain", "general"),
            "environment":        meta_a.get("environment", "general"),
            "difficulty":         pair.get("difficulty", "intermediate"),
            "citation_paragraphs":pair.get("citation_paragraphs", source_chunks),
        },
    }
    return resume_key, seed_record, label


def run(enriched_path: str, seeds_path: str, target: int = TARGET_PAIRS,
        seed: int = RANDOM_SEED, chapters_filter: Optional[set] = None,
        workers: int = 1) -> int:
    """
    Generate QA pairs from enriched chunks and write to seeds_path.

    workers=1  : sequential (default, original behaviour)
    workers>1  : concurrent — N tasks in flight simultaneously, vLLM batches them.
                 Recommended: workers=4 for a single vLLM instance on a DGX Spark.
                 Monitor GPU utilisation and increase until it plateaus.
    """
    Path(seeds_path).parent.mkdir(parents=True, exist_ok=True)

    # Load enriched chunks
    chunks = []
    with open(enriched_path, encoding='utf-8') as f:
        for line in f:
            try:
                chunks.append(json.loads(line))
            except Exception:
                pass
    print(f"[generator] Loaded {len(chunks)} enriched chunks")

    # Chapter filter — keeps each machine's work non-overlapping
    if chapters_filter:
        chunks = [c for c in chunks if c.get("chapter_num") in chapters_filter]
        print(f"[generator] Chapter filter {sorted(chapters_filter, key=str)} → {len(chunks)} chunks")

    print(f"[generator] Backend: {BACKEND}  Machine: {MACHINE_ID}  "
          f"Seed: {seed}  Workers: {workers}")

    # Resume: load already-completed (source_chunks, question_type) keys
    done_keys: set[str] = set()
    if Path(seeds_path).exists():
        with open(seeds_path, encoding='utf-8') as f:
            for line in f:
                try:
                    s = json.loads(line)
                    pids = s.get("source_chunks", [])
                    done_keys.add(str(pids) + s.get("question_type", ""))
                except Exception:
                    pass
        print(f"[generator] Resuming — {len(done_keys)} pairs already saved")

    # Build 30 % more tasks than target to absorb the ~20 % filter rate.
    # Any surplus is discarded once target is reached.
    task_budget = round(target * 1.3)
    tasks = build_task_list(chunks, task_budget, seed=seed)
    print(f"[generator] Scheduled {len(tasks)} tasks → target {target} pairs  "
          f"(+30 % buffer for filter rate)\n")

    # Pre-filter tasks already done (resume)
    pending = []
    for i, (chunk_a, qt, chunk_b) in enumerate(tasks):
        pid_a = chunk_a.get("para_id", "")
        pid_b = chunk_b.get("para_id", "") if chunk_b else ""
        resume_key = str([pid_a, pid_b] if pid_b else [pid_a]) + qt
        if resume_key not in done_keys:
            pending.append((i, len(tasks), chunk_a, qt, chunk_b, resume_key, len(done_keys)))

    skipped = len(tasks) - len(pending)
    print(f"[generator] {skipped} already done, {len(pending)} pending\n")

    # ── Write lock + counters (shared across threads) ────────────
    lock      = threading.Lock()
    generated = 0   # pairs written this session
    filtered  = 0   # pairs that failed validation

    # MUST use append mode — never "w"
    with open(seeds_path, 'a', encoding='utf-8') as out_f:

        def _write_if_needed(result: Optional[tuple]) -> bool:
            """Write one validated pair. Returns True if written, False if skipped/target hit."""
            nonlocal generated, filtered
            if result is None:
                with lock:
                    filtered += 1
                return False
            resume_key, record, label = result
            with lock:
                if generated >= target:
                    return False   # another thread already hit target
                record["qa_id"] = f"{MACHINE_ID}-{generated + len(done_keys) + 1:05d}"
                out_f.write(json.dumps(record) + '\n')
                out_f.flush()
                done_keys.add(resume_key)
                generated += 1
                print(f"  {label} ... OK  [{generated}/{target}]", flush=True)
            return True

        if workers == 1:
            # ── Sequential (original behaviour) ───────────────
            for task_args in pending:
                if generated >= target:
                    break
                _write_if_needed(_process_task(task_args))
        else:
            # ── Concurrent ────────────────────────────────────
            # Submit tasks in streaming batches so we don't queue thousands
            # of futures when only a fraction are needed.
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {}
                pending_iter = iter(pending)

                # Fill initial window
                for task_args in pending_iter:
                    if generated >= target:
                        break
                    futures[pool.submit(_process_task, task_args)] = True
                    if len(futures) >= workers * 4:   # keep a small queue
                        break

                while futures:
                    done_future = next(as_completed(futures))
                    del futures[done_future]

                    _write_if_needed(done_future.result())

                    # Refill the queue if we still need more pairs
                    if generated < target:
                        for task_args in pending_iter:
                            futures[pool.submit(_process_task, task_args)] = True
                            if len(futures) >= workers * 4:
                                break

    print(f"\n[generator] Done — {generated} saved | "
          f"{skipped} already done | {filtered} failed/filtered")
    print(f"[generator] Seeds → {seeds_path}")
    return generated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 3: Generate thinking-trace QA pairs from enriched ATP 2-01.3 chunks"
    )
    parser.add_argument("--enriched",    default="data/enriched.jsonl")
    parser.add_argument("--out",         default="data/seeds.jsonl")
    parser.add_argument("--model",       default=OLLAMA_MODEL,
                        help="Ollama model name (ignored when --backend vllm)")
    parser.add_argument("--vllm-model",  default=VLLM_MODEL,
                        help="HuggingFace model name for vLLM server")
    parser.add_argument("--max-tokens",  type=int, default=VLLM_MAX_TOKENS,
                        help="Maximum completion tokens requested from vLLM per call "
                             f"(default: {VLLM_MAX_TOKENS})")
    parser.add_argument("--target",      type=int, default=TARGET_PAIRS,
                        help="Target number of QA pairs (default: 5000)")
    parser.add_argument("--backend",     default="ollama", choices=["ollama", "vllm"],
                        help="Inference backend: ollama (default) or vllm")
    parser.add_argument("--api-url",     default=None,
                        help="Override base URL. Ollama default: http://localhost:11434  "
                             "vLLM default: http://localhost:8000")
    parser.add_argument("--machine-id",  default="m0",
                        help="Short identifier for this machine (e.g. spark1, pc1, mac). "
                             "Prepended to qa_ids so merged files have no collisions.")
    parser.add_argument("--seed",        type=int, default=RANDOM_SEED,
                        help="Random seed controlling task ordering within the assigned chapters.")
    parser.add_argument("--chapters",    default="",
                        help="Comma-separated chapter IDs to process on THIS machine. "
                             "Omit to use all chapters (single-machine mode). "
                             "Example: --chapters 7,8  or  --chapters A,B,C  or  --chapters 2,3,D")
    parser.add_argument("--workers",     type=int, default=1,
                        help="Number of concurrent generation requests sent to the backend. "
                             "workers=1 is sequential (default). workers=4 is a good starting "
                             "point for a single vLLM instance on a DGX Spark — vLLM batches "
                             "the concurrent requests automatically. Increase until GPU util "
                             "plateaus (monitor with nvidia-smi dmon or watch nvitop).")
    args = parser.parse_args()

    # Apply args to module globals
    OLLAMA_MODEL = args.model
    VLLM_MODEL   = args.vllm_model
    VLLM_MAX_TOKENS = args.max_tokens
    BACKEND      = args.backend
    MACHINE_ID   = args.machine_id

    if args.api_url:
        if BACKEND == "ollama":
            OLLAMA_URL = args.api_url.rstrip("/") + "/api/generate"
        else:
            VLLM_URL = args.api_url.rstrip("/")

    chapters_filter = parse_chapter_filter(args.chapters)
    run(args.enriched, args.out, args.target, seed=args.seed,
        chapters_filter=chapters_filter, workers=args.workers)
