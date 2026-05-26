"""
monitor.py
==========
Live progress monitoring for the ATP 2-01.3 pipeline.

Watches pipeline data files and prints a dashboard showing:
  - Chunk count
  - Enriched count
  - Seeds count (with question type breakdown)
  - Training progress (loss from trainer log)
  - Eval scores (if eval results exist)

Run in a separate terminal while the pipeline is executing:
    python monitor.py
    python monitor.py --watch   # continuous refresh every 30s
    python monitor.py --data-dir data --out-dir outputs

Note: This script reads existing files only — it does NOT affect the pipeline.
"""

import json
import time
import argparse
import os
from pathlib import Path
from datetime import datetime
from collections import Counter

# ── Config ────────────────────────────────────────────────────

REFRESH_SECONDS = 30
DPO_THRESHOLD   = 0.5


# ── File readers ──────────────────────────────────────────────

def count_jsonl(path: Path) -> int:
    """Count lines in a JSONL file (each line = one record)."""
    if not path.exists():
        return 0
    count = 0
    try:
        with open(path, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    count += 1
    except Exception:
        pass
    return count


def read_jsonl_sample(path: Path, n: int = 3) -> list[dict]:
    if not path.exists():
        return []
    records = []
    try:
        with open(path, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
                if len(records) >= n:
                    break
    except Exception:
        pass
    return records


def read_all_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    try:
        with open(path, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
    except Exception:
        pass
    return records


def read_latest_eval(eval_dir: Path) -> dict | None:
    """Return the most recently modified eval JSON result."""
    if not eval_dir.exists():
        return None
    results = sorted(eval_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not results:
        return None
    try:
        with open(results[0], encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def find_latest_trainer_log(out_dir: Path) -> Path | None:
    """Find trainer_state.json from the most recent adapter directory."""
    candidates = []
    for d in out_dir.iterdir() if out_dir.exists() else []:
        ts = d / "trainer_state.json"
        if ts.exists():
            candidates.append(ts)
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def read_trainer_state(log_path: Path | None) -> dict:
    if log_path is None or not log_path.exists():
        return {}
    try:
        with open(log_path, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


# ── Dashboard rendering ───────────────────────────────────────

def render_dashboard(data_dir: Path, out_dir: Path, eval_dir: Path) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*64}")
    print(f"  ATP 2-01.3 Pipeline Monitor  [{now}]")
    print(f"{'='*64}")

    # ── Stage 1: Chunks ──────────────────────────────────────
    chunks_path = data_dir / "chunks.jsonl"
    n_chunks    = count_jsonl(chunks_path)
    ch_status   = f"{n_chunks:,} chunks" if n_chunks > 0 else "NOT STARTED"
    if n_chunks > 0:
        chunks     = read_all_jsonl(chunks_path)
        ch_counts  = Counter(str(c.get("chapter_num", "?")) for c in chunks)
        ch_summary = "  ".join(f"ch{k}:{v}" for k, v in sorted(ch_counts.items(), key=lambda x: (len(x[0]), x[0])))
        ch_status += f" | {ch_summary}"
    print(f"\n  [1] Chunks    : {ch_status}")
    print(f"       File     : {chunks_path}")

    # ── Stage 2: Enriched ────────────────────────────────────
    enriched_path = data_dir / "enriched.jsonl"
    n_enriched    = count_jsonl(enriched_path)
    if n_enriched > 0:
        pct          = int(100 * n_enriched / max(n_chunks, 1))
        enrich_status = f"{n_enriched:,} enriched ({pct}% of chunks)"
    else:
        enrich_status = "NOT STARTED"
    print(f"  [2] Enriched  : {enrich_status}")
    print(f"       File     : {enriched_path}")

    # ── Stage 3: Seeds ───────────────────────────────────────
    seeds_path = data_dir / "seeds.jsonl"
    n_seeds    = count_jsonl(seeds_path)
    if n_seeds > 0:
        seeds    = read_all_jsonl(seeds_path)
        qt_dist  = Counter(s.get("question_type", "?") for s in seeds)
        qt_parts = "  ".join(f"{k[:3]}:{v}" for k, v in qt_dist.most_common())
        seeds_status = f"{n_seeds:,} pairs | {qt_parts}"

        # Citation check on sample
        with_cite = sum(1 for s in seeds if "[Reference: ATP 2-01.3" in s.get("answer", ""))
        seeds_status += f" | citations: {with_cite}/{n_seeds}"
    else:
        seeds_status = "NOT STARTED"
    print(f"  [3] Seeds     : {seeds_status}")
    print(f"       File     : {seeds_path}")

    # ── Stage 4: Training ────────────────────────────────────
    trainer_log  = find_latest_trainer_log(out_dir)
    trainer_state = read_trainer_state(trainer_log)
    if trainer_state:
        history = trainer_state.get("log_history", [])
        # Get latest loss
        loss_entries = [e for e in history if "loss" in e]
        eval_entries = [e for e in history if "eval_loss" in e]
        if loss_entries:
            last = loss_entries[-1]
            train_status = (
                f"step {last.get('step', '?')} / "
                f"epoch {last.get('epoch', '?'):.2f} | "
                f"loss={last.get('loss', '?'):.4f}"
            )
            if eval_entries:
                ev = eval_entries[-1]
                train_status += f" | eval_loss={ev.get('eval_loss', '?'):.4f}"
        else:
            train_status = "initializing..."
        adapter_name = trainer_log.parent.name if trainer_log else "unknown"
        print(f"  [4] Training  : {train_status}")
        print(f"       Adapter  : {adapter_name}")
    else:
        # Check if any adapter dirs exist
        adapters = [d for d in (out_dir.iterdir() if out_dir.exists() else [])
                    if d.is_dir() and (d / "adapter_config.json").exists()]
        if adapters:
            latest = sorted(adapters, key=lambda d: d.stat().st_mtime, reverse=True)[0]
            print(f"  [4] Training  : COMPLETE ({latest.name})")
        else:
            print(f"  [4] Training  : NOT STARTED")

    # ── Stage 5: Eval ────────────────────────────────────────
    eval_result = read_latest_eval(eval_dir)
    if eval_result:
        ft_avg   = eval_result.get("ft_avg", 0)
        base_avg = eval_result.get("base_avg")
        dpo_ids  = eval_result.get("dpo_ids", [])
        run_id   = eval_result.get("run_id", "?")

        eval_line = f"avg={ft_avg:.3f}"
        if base_avg is not None:
            delta = ft_avg - base_avg
            eval_line += f" (base={base_avg:.3f}, Δ={delta:+.3f})"
        if dpo_ids:
            eval_line += f" | DPO needed: {', '.join(dpo_ids)}"

        print(f"  [5] Eval      : {eval_line}")
        print(f"       Run ID   : {run_id}")

        # Per-question scores
        qs = eval_result.get("questions", [])
        if qs:
            low  = sorted([q for q in qs if q.get("ft_score", 1) < DPO_THRESHOLD],
                          key=lambda q: q.get("ft_score", 0))
            high = sorted([q for q in qs if q.get("ft_score", 0) >= DPO_THRESHOLD],
                          key=lambda q: -q.get("ft_score", 0))
            if low:
                print(f"       Weak ({len(low)}): " +
                      "  ".join(f"{q['id']}={q.get('ft_score', 0):.2f}" for q in low))
            if high[:3]:
                print(f"       Strong    : " +
                      "  ".join(f"{q['id']}={q.get('ft_score', 0):.2f}" for q in high[:3]))
    else:
        print(f"  [5] Eval      : NOT STARTED")

    # ── File sizes ────────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  Data directory: {data_dir}")
    for fname in ["chunks.jsonl", "enriched.jsonl", "seeds.jsonl", "train.jsonl", "val.jsonl"]:
        fpath = data_dir / fname
        if fpath.exists():
            size_kb = fpath.stat().st_size / 1024
            print(f"    {fname:<20} {size_kb:>8.1f} KB")

    print(f"{'='*64}\n")


# ── Entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ATP 2-01.3 pipeline progress monitor"
    )
    parser.add_argument("--data-dir",  default="data",
                        help="Data directory (default: data/)")
    parser.add_argument("--out-dir",   default="outputs",
                        help="Adapter output directory (default: outputs/)")
    parser.add_argument("--eval-dir",  default="eval/results",
                        help="Eval results directory (default: eval/results/)")
    parser.add_argument("--watch",     action="store_true",
                        help=f"Refresh every {REFRESH_SECONDS}s (default: run once)")
    parser.add_argument("--interval",  type=int, default=REFRESH_SECONDS,
                        help=f"Refresh interval in seconds (default: {REFRESH_SECONDS})")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    eval_dir = Path(args.eval_dir)

    try:
        render_dashboard(data_dir, out_dir, eval_dir)
        if args.watch:
            while True:
                time.sleep(args.interval)
                # Clear screen on supported terminals
                os.system("clear" if os.name != "nt" else "cls")
                render_dashboard(data_dir, out_dir, eval_dir)
    except KeyboardInterrupt:
        print("\n[monitor] Stopped.")
