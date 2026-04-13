"""
run_pipeline.py
===============
Orchestrator for the ATP 2-01.3 fine-tuning pipeline.

Runs all 5 stages in order, with support for skipping completed stages.

Stages:
  1. chunk     — Parse PDF → data/chunks.jsonl
  2. enrich    — Metadata classification → data/enriched.jsonl
  3. generate  — QA generation → data/seeds.jsonl
  4. train     — Format + QLoRA SFT → outputs/<run_name>
  5. eval      — 24-question evaluation → eval/results/<run_name>.json
  5b. dpo      — DPO on weak questions (optional, requires --run-dpo)

VRAM Note: Stages 1-3 use Ollama. Stop Ollama before Stage 4 (training).
           This script DOES NOT automatically stop Ollama — do it manually:
               ollama stop
           or kill the Ollama process before running --start-from train.

Usage:
    # Full pipeline
    python run_pipeline.py --pdf ~/caimll_finetuning/ATP_2-01.3.pdf

    # Skip already-completed stages
    python run_pipeline.py --pdf ATP_2-01.3.pdf --skip-chunk --skip-enrich

    # Start from a specific stage
    python run_pipeline.py --start-from generate

    # Training only
    python run_pipeline.py --start-from train --run-name atp-gemma4-v1

    # Eval only
    python run_pipeline.py --start-from eval --adapter outputs/atp-gemma4-v1

    # Full pipeline with DPO
    python run_pipeline.py --pdf ATP_2-01.3.pdf --run-dpo
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

# ── Stage ordering ────────────────────────────────────────────

STAGES = ["chunk", "enrich", "generate", "train", "eval", "dpo"]


def parse_args():
    p = argparse.ArgumentParser(
        description="ATP 2-01.3 fine-tuning pipeline orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Global
    p.add_argument("--run-name",     default=None,
                   help="Run name for outputs (default: auto-timestamped)")
    p.add_argument("--data-dir",     default="data",
                   help="Data directory (default: data/)")
    p.add_argument("--out-dir",      default="outputs",
                   help="Output directory for adapters (default: outputs/)")

    # Stage control
    p.add_argument("--start-from",   choices=STAGES, default="chunk",
                   help="Start from this stage (skip all before it)")
    p.add_argument("--stop-after",   choices=STAGES, default=None,
                   help="Stop after this stage")
    p.add_argument("--skip-chunk",   action="store_true")
    p.add_argument("--skip-enrich",  action="store_true")
    p.add_argument("--skip-generate",action="store_true")
    p.add_argument("--skip-train",   action="store_true")
    p.add_argument("--skip-eval",    action="store_true")
    p.add_argument("--run-dpo",      action="store_true",
                   help="Run DPO after eval (default: off)")

    # Stage 1 — chunker
    p.add_argument("--pdf",          default=str(Path.home() / "caimll_finetuning" / "ATP_2-01.3.pdf"),
                   help="Path to ATP 2-01.3 PDF")

    # Stage 3 — generator
    p.add_argument("--target",       type=int, default=5000,
                   help="Target QA pairs (default: 5000)")
    p.add_argument("--model",        default="gemma4:31b",
                   help="Ollama model for generation (default: gemma4:31b)")

    # Stage 4 — trainer
    p.add_argument("--epochs",       type=int,   default=2)
    p.add_argument("--batch",        type=int,   default=2)
    p.add_argument("--grad-accum",   type=int,   default=8)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--seq-len",      type=int,   default=4096)
    p.add_argument("--base-model",   default="unsloth/gemma-4-31B")

    # Stage 5 — eval
    p.add_argument("--adapter",      default=None,
                   help="Adapter path for eval-only runs")
    p.add_argument("--compare-base", action="store_true",
                   help="Also evaluate base model during eval stage")

    # Stage 5b — DPO
    p.add_argument("--dpo-max-pairs",type=int, default=200)

    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────

def run_cmd(cmd: list[str], stage: str) -> None:
    """Run a subprocess command; exit on failure."""
    print(f"\n{'='*60}")
    print(f"  STAGE: {stage.upper()}")
    print(f"  CMD  : {' '.join(cmd)}")
    print(f"{'='*60}\n")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[pipeline] Stage '{stage}' FAILED (exit {result.returncode})")
        sys.exit(result.returncode)
    print(f"\n[pipeline] Stage '{stage}' OK ({elapsed:.0f}s / {elapsed/60:.1f}min)")


def should_run(stage: str, args, stage_order: list[str]) -> bool:
    """Determine if a stage should run given skip flags and start-from."""
    start_idx = stage_order.index(args.start_from) if args.start_from in stage_order else 0
    stage_idx = stage_order.index(stage)
    if stage_idx < start_idx:
        return False
    if args.stop_after and stage_order.index(args.stop_after) < stage_idx:
        return False
    # Check per-stage skip flags
    skip_map = {
        "chunk":    getattr(args, "skip_chunk",    False),
        "enrich":   getattr(args, "skip_enrich",   False),
        "generate": getattr(args, "skip_generate", False),
        "train":    getattr(args, "skip_train",    False),
        "eval":     getattr(args, "skip_eval",     False),
        "dpo":      not getattr(args, "run_dpo",   False),
    }
    return not skip_map.get(stage, False)


def vram_warning():
    print("\n" + "!" * 60)
    print("  VRAM WARNING: About to start TRAINING.")
    print("  Ensure Ollama is stopped before proceeding!")
    print("  Run:  ollama stop   (or kill the ollama process)")
    print("!" * 60 + "\n")
    time.sleep(3)


# ── Pipeline ──────────────────────────────────────────────────

def main():
    args     = parse_args()
    py       = sys.executable
    stage_order = ["chunk", "enrich", "generate", "train", "eval", "dpo"]

    # Auto-generate run name
    run_name = args.run_name or f"atp-gemma4-{datetime.now().strftime('%Y%m%d-%H%M')}"
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)

    chunks_path   = data_dir / "chunks.jsonl"
    enriched_path = data_dir / "enriched.jsonl"
    seeds_path    = data_dir / "seeds.jsonl"
    train_path    = data_dir / "train.jsonl"
    val_path      = data_dir / "val.jsonl"
    adapter_path  = out_dir / run_name
    eval_out      = Path("eval") / "results" / f"{run_name}.json"
    dpo_out       = out_dir / f"{run_name}-dpo"

    print(f"\n{'#'*60}")
    print(f"  ATP 2-01.3 Fine-Tuning Pipeline v2")
    print(f"  Run : {run_name}")
    print(f"  Time: {datetime.now().isoformat()}")
    print(f"{'#'*60}\n")

    pipeline_start = time.time()

    # ── Stage 1: Chunk ────────────────────────────────────────
    if should_run("chunk", args, stage_order):
        run_cmd([
            py, "chunker.py",
            "--pdf", args.pdf,
            "--out", str(chunks_path),
        ], "chunk")

    # ── Stage 2: Enrich ───────────────────────────────────────
    if should_run("enrich", args, stage_order):
        run_cmd([
            py, "enricher.py",
            "--chunks", str(chunks_path),
            "--out",    str(enriched_path),
        ], "enrich")

    # ── Stage 3: Generate ─────────────────────────────────────
    if should_run("generate", args, stage_order):
        run_cmd([
            py, "generator.py",
            "--enriched", str(enriched_path),
            "--out",      str(seeds_path),
            "--model",    args.model,
            "--target",   str(args.target),
        ], "generate")

    # ── Stage 4: Train (Format + SFT) ────────────────────────
    if should_run("train", args, stage_order):
        vram_warning()

        # 4a: Format seeds → train/val splits
        run_cmd([
            py, "formatter.py",
            "--seeds", str(seeds_path),
            "--train", str(train_path),
            "--val",   str(val_path),
        ], "format")

        # 4b: QLoRA SFT
        run_cmd([
            py, "trainer.py",
            "--train",     str(train_path),
            "--val",       str(val_path),
            "--out",       str(adapter_path),
            "--model",     args.base_model,
            "--epochs",    str(args.epochs),
            "--batch",     str(args.batch),
            "--grad-accum",str(args.grad_accum),
            "--lr",        str(args.lr),
            "--seq-len",   str(args.seq_len),
        ], "train")

    # Resolve adapter path for eval
    eval_adapter = args.adapter or str(adapter_path)

    # ── Stage 5: Eval ─────────────────────────────────────────
    if should_run("eval", args, stage_order):
        eval_cmd = [
            py, "eval.py",
            "--adapter",   eval_adapter,
            "--questions", "eval/questions.json",
            "--out",       str(eval_out),
        ]
        if args.compare_base:
            eval_cmd.append("--compare-base")
        run_cmd(eval_cmd, "eval")

    # ── Stage 5b: DPO ─────────────────────────────────────────
    if should_run("dpo", args, stage_order):
        if not eval_out.exists():
            print(f"[pipeline] DPO requires eval results at {eval_out} — run eval first.")
        else:
            run_cmd([
                py, "dpo.py",
                "--sft",       eval_adapter,
                "--eval",      str(eval_out),
                "--seeds",     str(seeds_path),
                "--out",       str(dpo_out),
                "--max-pairs", str(args.dpo_max_pairs),
            ], "dpo")

    elapsed = time.time() - pipeline_start
    print(f"\n{'#'*60}")
    print(f"  PIPELINE COMPLETE: {run_name}")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Adapter   : {eval_adapter}")
    if args.run_dpo:
        print(f"  DPO adapter: {dpo_out}")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
