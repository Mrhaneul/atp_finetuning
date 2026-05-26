#!/bin/bash
# ATP 2-01.3 overnight fine-tuning run
# Usage: ./run_overnight.sh [path/to/ATP_2-01.3.pdf]
# Logs to: overnight.log

PDF="${1:-ATP_2-01.3.pdf}"
LOG="overnight.log"
ADAPTER="outputs/atp-gemma4-e4b-mlx-v1-300"
CONDA_ENV="caimll_finetuning"

PYTHON="$(conda run -n $CONDA_ENV which python)"

echo "========================================" | tee "$LOG"
echo " ATP 2-01.3 Overnight Run"               | tee -a "$LOG"
echo " Started: $(date)"                        | tee -a "$LOG"
echo " PDF: $PDF"                               | tee -a "$LOG"
echo " Adapter out: $ADAPTER"                   | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

fail() { echo "[FAILED] $1" | tee -a "$LOG"; exit 1; }

# ── Stage 1: Chunk ────────────────────────────────────────────
if [ -f "data/chunks.jsonl" ]; then
    echo "[skip] data/chunks.jsonl exists" | tee -a "$LOG"
else
    echo "" | tee -a "$LOG"
    echo "── Stage 1: Chunk ──" | tee -a "$LOG"
    $PYTHON scripts/chunker.py --pdf "$PDF" --out data/chunks.jsonl 2>&1 | tee -a "$LOG"
    [ ${PIPESTATUS[0]} -eq 0 ] || fail "chunker.py"
fi

# ── Stage 2: Enrich ───────────────────────────────────────────
if [ -f "data/enriched.jsonl" ]; then
    echo "[skip] data/enriched.jsonl exists" | tee -a "$LOG"
else
    echo "" | tee -a "$LOG"
    echo "── Stage 2: Enrich ──" | tee -a "$LOG"
    $PYTHON scripts/enricher.py --chunks data/chunks.jsonl --out data/enriched.jsonl 2>&1 | tee -a "$LOG"
    [ ${PIPESTATUS[0]} -eq 0 ] || fail "enricher.py"
fi

# ── Stage 3: Generate 300 QA pairs ───────────────────────────
echo "" | tee -a "$LOG"
echo "── Stage 3: Generate QA pairs (target: 300) ──" | tee -a "$LOG"
$PYTHON scripts/generator.py \
    --enriched data/enriched.jsonl \
    --out data/seeds.jsonl \
    --model gemma4:latest \
    --target 300 \
    2>&1 | tee -a "$LOG"
[ ${PIPESTATUS[0]} -eq 0 ] || fail "generator.py"

# ── Stage 4: Format ───────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "── Stage 4: Format ──" | tee -a "$LOG"
$PYTHON scripts/formatter.py \
    --seeds data/seeds.jsonl \
    --train data/train.jsonl \
    --val   data/valid.jsonl \
    2>&1 | tee -a "$LOG"
[ ${PIPESTATUS[0]} -eq 0 ] || fail "formatter.py"

# ── Stage 5: Train (MLX LoRA) ─────────────────────────────────
echo "" | tee -a "$LOG"
echo "── Stage 5: Train (MLX LoRA) ──" | tee -a "$LOG"
echo "    Stop Ollama before this step to free unified memory."
echo "    Stopping Ollama..." | tee -a "$LOG"
ollama stop 2>/dev/null || true

$PYTHON -m mlx_lm lora \
    --model        mlx-community/gemma-4-E4B-it-4bit \
    --train \
    --data         data \
    --iters        1500 \
    --batch-size   4 \
    --num-layers   8 \
    --learning-rate 5e-6 \
    --adapter-path "$ADAPTER" \
    --steps-per-eval 100 \
    --save-every   300 \
    2>&1 | tee -a "$LOG"
[ ${PIPESTATUS[0]} -eq 0 ] || fail "mlx_lm lora training"

# ── Stage 6: Evaluate (ROUGE-L) ───────────────────────────────
echo "" | tee -a "$LOG"
echo "── Stage 6: Evaluate ──" | tee -a "$LOG"
$PYTHON scripts/evaluator.py \
    --adapter "$ADAPTER" \
    --valid   data/valid.jsonl \
    --out     eval/results.jsonl \
    2>&1 | tee -a "$LOG"
[ ${PIPESTATUS[0]} -eq 0 ] || fail "evaluator.py"

# ── Stage 7: Plot ─────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "── Stage 7: Plot ──" | tee -a "$LOG"
$PYTHON scripts/plot_eval.py \
    --valid   data/valid.jsonl \
    --results eval/results.jsonl \
    --out     eval/rouge_chart.png \
    2>&1 | tee -a "$LOG"
[ ${PIPESTATUS[0]} -eq 0 ] || fail "plot_eval.py"

echo "" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo " DONE: $(date)"                           | tee -a "$LOG"
echo " Adapter  : $ADAPTER"                     | tee -a "$LOG"
echo " Results  : eval/results.jsonl"           | tee -a "$LOG"
echo " Chart    : eval/rouge_chart.png"         | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
