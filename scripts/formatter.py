"""
formatter.py
============
Stage 4a: Format seeds.jsonl into Gemma 4 chat-template training examples.

Produces train.jsonl and val.jsonl for the trainer.

Gemma 4 chat format (with native thinking trace):
    <bos><|system|>
    {system_prompt}
    <|end|>
    <|user|>
    {question}
    <|end|>
    <|assistant|>
    <|channel>thought
    {thinking_trace}
    <channel|>
    {answer}
    <|end|>

CRITICAL:
  - NEVER fp16 with Gemma 4 — bf16 only (enforced by trainer, not here).
  - The thinking trace is embedded using Gemma 4 native channel tokens.
  - System prompt includes IPB step + echelon context from metadata.
  - Citation must remain at END of answer — formatter preserves order.

Input  : data/seeds.jsonl
Output : data/train.jsonl, data/val.jsonl

Usage:
    python formatter.py
    python formatter.py --seeds data/seeds.jsonl --train data/train.jsonl --val data/val.jsonl
    python formatter.py --seeds data/seeds.jsonl --val-ratio 0.10
"""

import json
import random
import argparse
from pathlib import Path
from collections import Counter
from typing import Optional

# ── Config ────────────────────────────────────────────────────

VAL_RATIO   = 0.10
RANDOM_SEED = 42

IPB_STEP_LABELS = {
    0:    "IPB Fundamentals",
    1:    "Step 1\u2014Define the Operational Environment",
    2:    "Step 2\u2014Describe Environmental Effects on Operations",
    3:    "Step 3\u2014Evaluate the Threat",
    4:    "Step 4\u2014Determine Threat Courses of Action",
    None: "General IPB",
}

# Base system prompt template — metadata enriches per-example context
SYSTEM_TEMPLATE = (
    "You are a doctrine-grounded military intelligence assistant specializing in "
    "ATP 2-01.3 (Intelligence Preparation of the Battlefield, March 2019). "
    "IPB Context: {ipb_label}. Echelon: {echelon}. Domain: {domain}. "
    "Provide thorough, doctrinally accurate answers. "
    "Place all citations at the END of your answer in "
    "[Reference: ATP 2-01.3, para X-Y] format — NEVER at the beginning."
)


# ── Formatting ────────────────────────────────────────────────

def build_system_prompt(meta: dict) -> str:
    ipb_step = meta.get("ipb_step")
    ipb_label = IPB_STEP_LABELS.get(ipb_step, "General IPB")
    echelon   = meta.get("echelon", "general").replace("_", " ").title()
    domain    = meta.get("domain",  "general").title()
    return SYSTEM_TEMPLATE.format(
        ipb_label=ipb_label,
        echelon=echelon,
        domain=domain,
    )


def format_example(seed: dict) -> Optional[dict]:
    """
    Convert a seed QA pair to Gemma 4 chat-template text.
    Returns None if the seed is malformed.
    """
    q   = seed.get("question",      "").strip()
    tt  = seed.get("thinking_trace","").strip()
    ans = seed.get("answer",        "").strip()
    meta = seed.get("metadata",     {})

    if not q or not ans:
        return None

    system = build_system_prompt(meta)

    # Gemma 4 native format with thinking channel tokens
    # If no thinking trace, omit the channel block
    if tt:
        assistant_block = (
            f"<|channel>thought\n{tt}\n<channel|>\n{ans}"
        )
    else:
        assistant_block = ans

    text = (
        f"<bos><|system|>\n{system}\n<|end|>\n"
        f"<|user|>\n{q}\n<|end|>\n"
        f"<|assistant|>\n{assistant_block}\n<|end|>"
    )

    return {
        "text":          text,
        "qa_id":         seed.get("qa_id"),
        "question_type": seed.get("question_type"),
        "chapter_num":   meta.get("chapter_num"),
        "ipb_step":      meta.get("ipb_step"),
        "echelon":       meta.get("echelon"),
    }


# ── Split & save ──────────────────────────────────────────────

def format_and_split(
    seeds_path: str,
    train_path: str,
    val_path:   str,
    val_ratio:  float = VAL_RATIO,
) -> tuple[int, int]:
    """
    Load seeds, format, shuffle, split train/val.
    Returns (n_train, n_val).
    """
    seeds = []
    with open(seeds_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                seeds.append(json.loads(line))
            except Exception:
                pass
    print(f"[formatter] Loaded {len(seeds)} seeds from {seeds_path}")

    examples = []
    skipped  = 0
    for seed in seeds:
        ex = format_example(seed)
        if ex:
            examples.append(ex)
        else:
            skipped += 1
    if skipped:
        print(f"[formatter] Skipped {skipped} malformed seeds")

    random.seed(RANDOM_SEED)
    random.shuffle(examples)

    n_val   = max(1, int(len(examples) * val_ratio))
    n_train = len(examples) - n_val
    train   = examples[:n_train]
    val     = examples[n_train:]

    Path(train_path).parent.mkdir(parents=True, exist_ok=True)

    with open(train_path, 'w', encoding='utf-8') as f:
        for ex in train:
            f.write(json.dumps(ex) + '\n')

    with open(val_path, 'w', encoding='utf-8') as f:
        for ex in val:
            f.write(json.dumps(ex) + '\n')

    # Distribution report
    qt_dist   = Counter(ex.get("question_type", "unknown") for ex in train)
    step_dist = Counter(str(ex.get("ipb_step")) for ex in train)

    print(f"\n[formatter] Split: {n_train} train | {n_val} val")
    print("[formatter] Question type distribution (train):")
    for qt, cnt in sorted(qt_dist.items(), key=lambda x: -x[1]):
        print(f"  {qt:<25}: {cnt}")
    print("[formatter] IPB step distribution (train):")
    for step, cnt in sorted(step_dist.items()):
        label = IPB_STEP_LABELS.get(int(step) if step.isdigit() else None, step)
        print(f"  Step {step} ({label[:30]}): {cnt}")
    print(f"\n[formatter] Train \u2192 {train_path}")
    print(f"[formatter] Val   \u2192 {val_path}")

    return n_train, n_val


# ── Entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 4a: Format seeds into Gemma 4 chat-template training data"
    )
    parser.add_argument("--seeds",     default="data/seeds.jsonl")
    parser.add_argument("--train",     default="data/train.jsonl")
    parser.add_argument("--val",       default="data/val.jsonl")
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO,
                        help=f"Fraction held out for validation (default: {VAL_RATIO})")
    args = parser.parse_args()
    format_and_split(args.seeds, args.train, args.val, args.val_ratio)
