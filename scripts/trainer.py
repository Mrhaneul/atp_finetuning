"""
trainer.py
==========
Stage 4b: QLoRA SFT fine-tuning of Gemma 4 31B Dense on ATP 2-01.3 data.

CRITICAL — GEMMA 4 RULES (from handoff; violations will crash or corrupt training):
  1. NEVER fp16 — Gemma 4 activations overflow fp16.  Always bf16.
  2. NEVER hardcode base model path — read from adapter_config.json if resuming.
  3. NEVER exceed 3 epochs at 5K+ examples.
  4. NEVER LoRA r > 16 without evidence it helps.
  5. STOP Ollama before running trainer (VRAM conflict).

Hyperparameters (proven defaults from handoff):
  - epochs: 2, batch: 2, grad_accum: 8, lr: 2e-4
  - LoRA r=16, alpha=32, all linear layers, adamw_8bit
  - cosine scheduler, warmup_ratio=0.05
  - max_seq_length: 4096 (thinking traces need room)

Usage:
    python trainer.py
    python trainer.py --train data/train.jsonl --val data/val.jsonl --out outputs/atp-gemma4-v1
    python trainer.py --train data/train.jsonl --epochs 1   # quick test
"""

import json
import argparse
import sys
import time
from pathlib import Path

# ── Default Config ────────────────────────────────────────────

BASE_MODEL     = "unsloth/gemma-4-31B"
MAX_SEQ_LENGTH = 4096     # thinking traces need more than 2048

# Handoff proven hyperparameters — do not change without evidence
EPOCHS       = 2
BATCH_SIZE   = 2
GRAD_ACCUM   = 8          # effective batch = 16
LR           = 2e-4
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.0        # Unsloth recommends 0.0
RANDOM_SEED  = 42
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01


def parse_args():
    p = argparse.ArgumentParser(
        description="Stage 4b: QLoRA SFT of Gemma 4 31B on ATP 2-01.3"
    )
    p.add_argument("--train",      default="data/train.jsonl")
    p.add_argument("--val",        default="data/val.jsonl")
    p.add_argument("--out",        default="outputs/atp-gemma4-31b-v1",
                   help="Output directory for adapter weights")
    p.add_argument("--model",      default=BASE_MODEL)
    p.add_argument("--epochs",     type=int,   default=EPOCHS)
    p.add_argument("--batch",      type=int,   default=BATCH_SIZE)
    p.add_argument("--grad-accum", type=int,   default=GRAD_ACCUM)
    p.add_argument("--lr",         type=float, default=LR)
    p.add_argument("--lora-r",     type=int,   default=LORA_R)
    p.add_argument("--seq-len",    type=int,   default=MAX_SEQ_LENGTH)
    return p.parse_args()


def check_vram():
    """Print VRAM info and warn if Ollama may be running."""
    try:
        import torch
        props = torch.cuda.get_device_properties(0)
        total_gb  = props.total_memory  / 1e9
        avail_gb  = torch.cuda.mem_get_info()[0] / 1e9
        print(f"[trainer] GPU   : {props.name}")
        print(f"[trainer] VRAM  : {total_gb:.1f} GB total | {avail_gb:.1f} GB free")
        if avail_gb < (total_gb * 0.85):
            print("[trainer] WARNING: Less than 85% VRAM free — Ollama may still be running.")
            print("[trainer]          Run: ollama stop   (or kill the ollama process)")
            print("[trainer]          Continuing anyway — training may OOM.")
    except Exception:
        pass


def load_dataset_from_jsonl(path: str):
    """Load train.jsonl or val.jsonl into HF Dataset format."""
    from datasets import Dataset
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "text" in obj:
                    rows.append({"text": obj["text"]})
            except Exception:
                pass
    if not rows:
        raise ValueError(f"No valid examples found in {path}")
    return Dataset.from_list(rows)


def train(args) -> None:
    try:
        import torch
        from unsloth import FastLanguageModel
        from trl import SFTTrainer, SFTConfig
    except ImportError as e:
        print(f"[trainer] Import error: {e}")
        print("[trainer] Activate environment: source ~/caimll_finetuning/bin/activate")
        sys.exit(1)

    check_vram()

    # Gemma 4 REQUIRES bf16 — NEVER fp16
    if not torch.cuda.is_bf16_supported():
        print("[trainer] FATAL: bf16 not supported on this GPU.")
        print("[trainer] Gemma 4 activations overflow fp16. Cannot continue.")
        sys.exit(1)

    Path(args.out).mkdir(parents=True, exist_ok=True)

    print(f"\n[trainer] Base model      : {args.model}")
    print(f"[trainer] Train file      : {args.train}")
    print(f"[trainer] Val file        : {args.val}")
    print(f"[trainer] Output dir      : {args.out}")
    print(f"[trainer] Epochs          : {args.epochs}")
    print(f"[trainer] Batch           : {args.batch} x {args.grad_accum} = {args.batch * args.grad_accum} effective")
    print(f"[trainer] LR              : {args.lr}")
    print(f"[trainer] LoRA r/alpha    : {args.lora_r}/{args.lora_r * 2}")
    print(f"[trainer] Max seq length  : {args.seq_len}")
    print(f"[trainer] Precision       : bf16 (ALWAYS — Gemma 4 requirement)\n")

    # Load base model with 4-bit quantization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = args.model,
        max_seq_length = args.seq_len,
        dtype          = None,      # auto-detect; will use bf16
        load_in_4bit   = True,
    )

    # Apply LoRA to all linear layers
    model = FastLanguageModel.get_peft_model(
        model,
        r              = args.lora_r,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha              = args.lora_r * 2,
        lora_dropout            = LORA_DROPOUT,
        bias                    = "none",
        use_gradient_checkpointing = "unsloth",
        random_state            = RANDOM_SEED,
        use_rslora              = True,
    )
    model.print_trainable_parameters()

    # Load datasets
    print("[trainer] Loading datasets...")
    train_ds = load_dataset_from_jsonl(args.train)
    val_ds   = load_dataset_from_jsonl(args.val)
    print(f"[trainer] Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Sanity check
    print("\n" + "=" * 54)
    print("  PRE-TRAINING SANITY CHECK")
    print("=" * 54)
    sample = train_ds[0]["text"]
    has_channel = "<|channel>thought" in sample
    has_ref     = "[Reference: ATP 2-01.3" in sample
    has_end     = "<|end|>" in sample
    print(f"  Thinking channel token : {'OK' if has_channel else 'MISSING'}")
    print(f"  Reference citation     : {'OK' if has_ref else 'MISSING'}")
    print(f"  End token              : {'OK' if has_end else 'MISSING'}")
    print(f"  Sample text (first 200 chars):")
    print(f"    {sample[:200]!r}")
    print("=" * 54 + "\n")

    start = time.time()

    trainer = SFTTrainer(
        model        = model,
        tokenizer    = tokenizer,
        train_dataset= train_ds,
        eval_dataset = val_ds,
        args = SFTConfig(
            output_dir                  = args.out,
            dataset_text_field          = "text",
            max_seq_length              = args.seq_len,
            per_device_train_batch_size = args.batch,
            gradient_accumulation_steps = args.grad_accum,
            num_train_epochs            = args.epochs,
            learning_rate               = args.lr,
            lr_scheduler_type           = "cosine",
            warmup_ratio                = WARMUP_RATIO,
            optim                       = "adamw_8bit",
            weight_decay                = WEIGHT_DECAY,
            # CRITICAL: bf16=True, fp16=False — ALWAYS for Gemma 4
            bf16                        = True,
            fp16                        = False,
            logging_steps               = 10,
            eval_strategy               = "steps",
            eval_steps                  = 100,
            save_strategy               = "epoch",
            load_best_model_at_end      = False,
            seed                        = RANDOM_SEED,
            report_to                   = "none",
            packing                     = True,
            dataset_num_proc            = 2,
        ),
    )

    stats   = trainer.train()
    elapsed = time.time() - start

    model.save_pretrained(args.out, safe_serialization=True)
    tokenizer.save_pretrained(args.out)

    print("\n" + "=" * 54)
    print("  TRAINING COMPLETE")
    print("=" * 54)
    print(f"  Elapsed       : {elapsed/60:.1f} minutes")
    print(f"  Train loss    : {stats.metrics.get('train_loss', 'N/A'):.4f}")
    print(f"  Train examples: {len(train_ds)}")
    print(f"  Epochs        : {args.epochs}")
    print(f"  Output        : {args.out}")
    print("=" * 54)
    print(f"\n[trainer] Adapter saved \u2192 {args.out}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
