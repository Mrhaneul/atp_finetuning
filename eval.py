"""
eval.py
=======
Stage 5a: Automated evaluation of the fine-tuned ATP 2-01.3 model.

Runs 24 hardcoded evaluation questions against the fine-tuned adapter
and (optionally) a base model for comparison.  Uses keyword coverage
scoring for automated assessment.

Scores range 0.0-1.0.  Questions with score < 0.5 are flagged for DPO.

Input  : eval/questions.json, outputs/<adapter>
Output : eval/results/<run_id>.json

Usage:
    python eval.py --adapter outputs/atp-gemma4-31b-v1
    python eval.py --adapter outputs/atp-gemma4-31b-v1 --compare-base
    python eval.py --adapter outputs/atp-gemma4-31b-v1 --out eval/results/v1.json
    python eval.py --questions eval/questions.json --adapter outputs/atp-gemma4-31b-v1
"""

import json
import argparse
import sys
import re
from pathlib import Path
from datetime import datetime

# ── Config ────────────────────────────────────────────────────

BASE_MODEL       = "unsloth/gemma-4-31B"
MAX_SEQ_LENGTH   = 4096
MAX_NEW_TOKENS   = 512
EVAL_TEMPERATURE = 0.1    # deterministic evaluation
DPO_THRESHOLD    = 0.5    # questions below this score go to DPO

QUESTIONS_FILE   = Path(__file__).parent / "eval" / "questions.json"

# System prompt for eval inference
EVAL_SYSTEM = (
    "You are a doctrine-grounded military intelligence assistant specializing in "
    "ATP 2-01.3 (Intelligence Preparation of the Battlefield, March 2019). "
    "Provide accurate, doctrinally grounded answers. Place citations at the END "
    "in [Reference: ATP 2-01.3, para X-Y] format."
)


# ── Load questions ────────────────────────────────────────────

def load_questions(path: str) -> list[dict]:
    with open(path, encoding='utf-8') as f:
        questions = json.load(f)
    print(f"[eval] Loaded {len(questions)} eval questions from {path}")
    return questions


# ── Scoring ───────────────────────────────────────────────────

def score_response(response: str, keywords: list[str]) -> float:
    """Keyword coverage — fraction of keywords present in response (case-insensitive)."""
    r    = response.lower()
    hits = sum(1 for kw in keywords if kw.lower() in r)
    return round(hits / len(keywords), 3) if keywords else 0.0


def check_citation(response: str) -> bool:
    """Check if response ends with proper ATP 2-01.3 citation."""
    return bool(re.search(r'\[Reference:\s*ATP\s+2-01\.3', response, re.IGNORECASE))


def check_no_bad_opening(response: str) -> bool:
    """Check that response doesn't start with 'According to ATP 2-01.3'."""
    return not bool(re.match(r'^\s*According to ATP', response, re.IGNORECASE))


# ── Inference ─────────────────────────────────────────────────

def load_model(model_path: str, is_adapter: bool = True):
    """Load model/tokenizer.  Returns (model, tokenizer)."""
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("[eval] unsloth not found. Activate the caimll_finetuning environment.")
        sys.exit(1)

    print(f"[eval] Loading: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = model_path,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype          = None,
        load_in_4bit   = True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def query(model, tokenizer, question: str) -> str:
    """Run inference and return decoded response."""
    import torch

    prompt = (
        f"<bos><|system|>\n{EVAL_SYSTEM}\n<|end|>\n"
        f"<|user|>\n{question}\n<|end|>\n"
        f"<|assistant|>\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = MAX_NEW_TOKENS,
            temperature    = EVAL_TEMPERATURE,
            do_sample      = EVAL_TEMPERATURE > 0,
            pad_token_id   = tokenizer.eos_token_id,
        )
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    # Strip thinking trace from eval response for scoring
    response = re.sub(r'<\|channel>thought.*?<channel\|>', '', response, flags=re.DOTALL).strip()
    return response


def evaluate_model(model_path: str, questions: list[dict], label: str) -> list[dict]:
    """Load model, run all eval questions, return scored results."""
    model, tokenizer = load_model(model_path)

    results = []
    for eq in questions:
        response = query(model, tokenizer, eq["question"])
        kw_score = score_response(response, eq["keywords"])
        has_cite = check_citation(response)
        ok_open  = check_no_bad_opening(response)

        # Penalize missing citation or bad opening
        adjusted = kw_score
        if not has_cite:
            adjusted = max(0.0, adjusted - 0.1)
        if not ok_open:
            adjusted = max(0.0, adjusted - 0.1)

        results.append({
            "id":           eq["id"],
            "type":         eq["type"],
            "difficulty":   eq.get("difficulty", "unknown"),
            "kw_score":     kw_score,
            "score":        round(adjusted, 3),
            "has_citation": has_cite,
            "ok_opening":   ok_open,
            "response":     response[:600],
            "model":        label,
            "needs_dpo":    adjusted < DPO_THRESHOLD,
        })

        dpo_flag = " [DPO]" if adjusted < DPO_THRESHOLD else ""
        print(f"  {eq['id']:>6} [{label}] {adjusted:.3f}{dpo_flag}  "
              f"{'CITE_OK' if has_cite else 'NO_CITE'}  "
              f"{response[:60]}...")

    # Free VRAM
    import torch
    del model
    torch.cuda.empty_cache()

    return results


# ── Report ────────────────────────────────────────────────────

def print_report(ft_results: list[dict],
                 base_results: list[dict] | None,
                 out_path: str,
                 run_id: str) -> None:
    n = len(ft_results)
    sep = "=" * 82

    print(f"\n{sep}")
    print(f"  EVAL RESULTS — ATP 2-01.3 Pipeline v2  [{run_id}]")
    print(sep)

    if base_results:
        print(f"{'ID':<6} {'Type':<20} {'Diff':<12} {'Base':>6} {'FT':>6} {'Delta':>7} {'DPO?'}")
        print("-" * 82)
    else:
        print(f"{'ID':<6} {'Type':<20} {'Diff':<12} {'Score':>6} {'Cite':>5} {'DPO?'}")
        print("-" * 82)

    ft_avg   = 0.0
    base_avg = 0.0
    dpo_ids  = []
    records  = []

    for i, ft in enumerate(ft_results):
        ft_avg += ft["score"]
        base  = base_results[i] if base_results else None
        delta = ft["score"] - (base["score"] if base else 0.0)
        if base_avg is not None and base:
            base_avg += base["score"]

        if ft["needs_dpo"]:
            dpo_ids.append(ft["id"])

        rec = {
            "id":           ft["id"],
            "type":         ft["type"],
            "difficulty":   ft["difficulty"],
            "ft_score":     ft["score"],
            "has_citation": ft["has_citation"],
            "needs_dpo":    ft["needs_dpo"],
            "ft_response":  ft["response"],
        }
        if base:
            rec["base_score"]    = base["score"]
            rec["delta"]         = round(delta, 3)
            rec["base_response"] = base["response"]

        records.append(rec)

        if base:
            marker = "\u2191" if delta > 0.05 else ("\u2193" if delta < -0.05 else "=")
            dpo_flag = " DPO" if ft["needs_dpo"] else ""
            print(f"{ft['id']:<6} {ft['type'][:18]:<20} {ft['difficulty'][:10]:<12} "
                  f"{base['score']:>6.3f} {ft['score']:>6.3f} {delta:>+6.3f} {marker}"
                  f"{dpo_flag}")
        else:
            cite_str = "Y" if ft["has_citation"] else "N"
            dpo_flag = "YES" if ft["needs_dpo"] else ""
            print(f"{ft['id']:<6} {ft['type'][:18]:<20} {ft['difficulty'][:10]:<12} "
                  f"{ft['score']:>6.3f} {cite_str:>5} {dpo_flag}")

    ft_avg /= n
    if base_results:
        base_avg /= n

    print("-" * 82)
    if base_results:
        print(f"{'AVG':<6} {'':<20} {'':<12} "
              f"{base_avg:>6.3f} {ft_avg:>6.3f} {ft_avg - base_avg:>+6.3f}")
    else:
        print(f"{'AVG':<6} {'':<20} {'':<12} {ft_avg:>6.3f}")
    print(sep)

    if dpo_ids:
        print(f"\n[eval] Questions flagged for DPO ({len(dpo_ids)}): {', '.join(dpo_ids)}")
    else:
        print("\n[eval] No questions flagged for DPO — all scores >= 0.5")

    # Save JSON
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    output = {
        "run_id":       run_id,
        "timestamp":    datetime.now().isoformat(),
        "ft_avg":       round(ft_avg, 4),
        "base_avg":     round(base_avg, 4) if base_results else None,
        "delta":        round(ft_avg - base_avg, 4) if base_results else None,
        "dpo_threshold":DPO_THRESHOLD,
        "dpo_ids":      dpo_ids,
        "questions":    records,
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    print(f"\n[eval] Results \u2192 {out_path}")


# ── Entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 5a: Evaluate fine-tuned ATP 2-01.3 adapter"
    )
    parser.add_argument("--adapter",      required=True,
                        help="Path to fine-tuned LoRA adapter directory")
    parser.add_argument("--questions",    default=str(QUESTIONS_FILE))
    parser.add_argument("--compare-base", action="store_true",
                        help="Also evaluate base model for comparison")
    parser.add_argument("--base-model",   default=BASE_MODEL,
                        help=f"Base model for comparison (default: {BASE_MODEL})")
    parser.add_argument("--out",          default=None,
                        help="Output JSON path (default: eval/results/<adapter_name>.json)")
    args = parser.parse_args()

    adapter_name = Path(args.adapter).name
    run_id       = f"{adapter_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_path     = args.out or f"eval/results/{adapter_name}.json"

    questions    = load_questions(args.questions)

    print(f"\n[eval] Running fine-tuned model: {args.adapter}")
    ft_results   = evaluate_model(args.adapter, questions, label=adapter_name)

    base_results = None
    if args.compare_base:
        print(f"\n[eval] Running base model: {args.base_model}")
        base_results = evaluate_model(args.base_model, questions, label="BASE")

    print_report(ft_results, base_results, out_path, run_id)
