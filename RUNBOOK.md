# ATP 2-01.3 Fine-tuning Runbook

End-to-end guide for running the ATP IPB fine-tuning pipeline on any machine.
Covers data prep, model selection, training, and export.

---

## 1. Environment Setup

### Conda (primary)
```bash
conda activate caimll_finetuning
```

### Install dependencies (first time or new machine)
```bash
conda activate caimll_finetuning
pip install -r requirements.txt
```

---

## 2. Model Selection

Pick a model based on the machine's available memory. All models below are
Unsloth 4-bit quantized and work with this pipeline's `trainer.py`.

| Model | Size | Min VRAM | `--model` value | Notes |
|---|---|---|---|---|
| Gemma 4 31B | 31B dense | ~64 GB | `unsloth/gemma-4-31B` | DGX Spark default. bf16 only — NEVER fp16. |
| Gemma 4 26B-A4B | 26B MoE (4B active) | ~20 GB | `unsloth/Gemma-4-26B-A4B-it` | MoE — lighter on memory than 31B dense. |
| Gemma 4 E4B | 4B edge | ~8 GB | `unsloth/gemma-4-E4B-it-unsloth-bnb-4bit` | Good for laptops / low-VRAM machines. |
| Gemma 4 E2B | 2B edge | ~4 GB | `unsloth/gemma-4-E2B-it-unsloth-bnb-4bit` | Fastest. Any machine. |
| Llama 3.2 3B | 3B | ~4 GB | `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` | Used in v1 adapter. Proven on DGX Spark. |
| Llama 3.2 1B | 1B | ~2 GB | `unsloth/Llama-3.2-1B-Instruct-bnb-4bit` | Minimum viable. Testing only. |
| Qwen3 8B | 8B | ~10 GB | `unsloth/Qwen3-8B-unsloth-bnb-4bit` | Strong reasoning alternative. |
| Qwen3 4B | 4B | ~6 GB | `unsloth/Qwen3-4B-unsloth-bnb-4bit` | Compact, modern reasoning model. |

> **CRITICAL for all Gemma 4 models**: Always `bf16=True`, `fp16=False`.
> Gemma 4 activations overflow fp16 and will silently corrupt training.

---

## 3. Data Pipeline

Run these stages in order. If data already exists at a stage, skip it.

### Stage 1 — Chunk the PDF
```bash
python chunker.py
# Output: data/chunks.jsonl
```

### Stage 2 — Enrich chunks with metadata
```bash
python enricher.py
# Output: data/enriched.jsonl
```

### Stage 3 — Generate QA seeds (via vLLM or OpenAI)
```bash
# Full run (all chapters):
python generator.py --backend vllm --api-url http://localhost:8000

# Specific chapters (parallelise across tmux sessions):
python generator.py --backend vllm --api-url http://localhost:8000 \
  --machine-id spark2a --chapters 5 --target 800 --out data/seeds_spark2a.jsonl

python generator.py --backend vllm --api-url http://localhost:8000 \
  --machine-id spark2b --chapters 6 --target 700 --out data/seeds_spark2b.jsonl
```

### Stage 4 — Combine and format seeds
```bash
# If seeds are in per-chapter files under data/:
cat data/combined_ch_*.jsonl data/combined_appendix_*.jsonl > data/seeds.jsonl

# If you also have extra spark seeds:
cat data/seeds_spark2a.jsonl data/seeds_spark2b.jsonl >> data/seeds.jsonl

# Format into train/val splits (Gemma 4 chat template):
python formatter.py --seeds data/seeds.jsonl \
                    --train data/train.jsonl \
                    --val   data/val.jsonl
```

Formatter output summary to expect:
- ~4,000–4,500 train examples
- ~400–500 val examples
- Distribution printed by question type and IPB step

---

## 4. Pre-Training Checklist

Run these every time before training.

```bash
# 1. Kill vLLM if it is running (frees the GPU)
#    Find the PID:
ps aux | grep vllm
kill <PID>                   # or: pkill -f "vllm serve"

# 2. Stop Ollama
sudo systemctl stop ollama

# 3. Confirm GPU is clear (should show 0% utilisation, no compute processes)
nvidia-smi

# 4. Activate env
conda activate caimll_finetuning
```

---

## 5. Training

### DGX Spark GB10 (128 GB unified memory) — Gemma 4 31B
```bash
cd /home/student/caimll_finetuning/atp_pipeline_v2

conda activate caimll_finetuning && \
python trainer.py \
  --train data/train.jsonl \
  --val   data/val.jsonl \
  --batch 4 \
  --grad-accum 4 \
  --out   outputs/atp-gemma4-31b-v1
```

### Medium machine (~20–40 GB VRAM) — Gemma 4 26B MoE
```bash
python trainer.py \
  --model unsloth/Gemma-4-26B-A4B-it \
  --train data/train.jsonl \
  --val   data/val.jsonl \
  --batch 2 \
  --grad-accum 8 \
  --out   outputs/atp-gemma4-26b-v1
```

### Low-VRAM machine (~8–16 GB) — Gemma 4 E4B or Qwen3 8B
```bash
# Gemma 4 E4B
python trainer.py \
  --model unsloth/gemma-4-E4B-it-unsloth-bnb-4bit \
  --train data/train.jsonl \
  --val   data/val.jsonl \
  --batch 2 \
  --grad-accum 8 \
  --out   outputs/atp-gemma4-e4b-v1

# Qwen3 8B
python trainer.py \
  --model unsloth/Qwen3-8B-unsloth-bnb-4bit \
  --train data/train.jsonl \
  --val   data/val.jsonl \
  --batch 2 \
  --grad-accum 8 \
  --out   outputs/atp-qwen3-8b-v1
```

### Minimal machine (~4 GB) — Gemma 4 E2B or Llama 3.2 3B
```bash
# Gemma 4 E2B
python trainer.py \
  --model unsloth/gemma-4-E2B-it-unsloth-bnb-4bit \
  --train data/train.jsonl \
  --val   data/val.jsonl \
  --batch 2 \
  --grad-accum 8 \
  --out   outputs/atp-gemma4-e2b-v1

# Llama 3.2 3B (proven, used in v1)
python trainer.py \
  --model unsloth/Llama-3.2-3B-Instruct-bnb-4bit \
  --train data/train.jsonl \
  --val   data/val.jsonl \
  --batch 2 \
  --grad-accum 8 \
  --out   outputs/atp-llama3-3b-v1
```

### Run in tmux (always recommended — prevents disconnect killing training)
```bash
tmux new-session -d -s train
tmux send-keys -t train "conda activate caimll_finetuning && python trainer.py \
  --train data/train.jsonl --val data/val.jsonl \
  --batch 4 --grad-accum 4 --out outputs/atp-gemma4-31b-v1" Enter

# Attach to watch:
tmux attach -t train

# Detach (leave running):
# Ctrl+B, then D
```

### Rough time estimates

| Model | Machine | Est. time (4K examples, 2 epochs) |
|---|---|---|
| Gemma 4 31B | DGX Spark GB10 | 4–8 hours |
| Gemma 4 26B MoE | A100 80GB | 3–5 hours |
| Gemma 4 E4B | RTX 4090 | 1–2 hours |
| Qwen3 8B | RTX 3090/4090 | 1–2 hours |
| Llama 3.2 3B | RTX 3060+ | 30–60 min |
| Gemma 4 E2B | Any GPU 4GB+ | 20–40 min |

---

## 6. Monitoring Training

```bash
# Watch the tmux session live
tmux attach -t train

# Or tail the output if redirected to a file:
tail -f training.log

# Check GPU utilisation in a separate terminal:
watch -n 5 nvidia-smi
```

First eval at step 100. Multiply per-step time by total steps to estimate finish:
- Total steps = `ceil(train_examples / batch_size) * epochs`
- Optimizer updates = `total_steps / grad_accum`

---

## 7. Export to GGUF (for Ollama / llama.cpp)

```bash
python burn_gguf.py --adapter outputs/atp-gemma4-31b-v1
# Output: outputs/atp-gemma4-31b-v1/*.gguf
```

---

## 8. Hardware Notes

### DGX Spark GB10
- **Single GPU** — no multi-GPU parallelism
- **128 GB unified memory** (shared CPU+GPU via NVLink-C2C, LPDDR5X)
- `nvidia-smi` shows memory as `[N/A]` — this is normal for unified memory
- `--batch 4 --grad-accum 4` is the sweet spot (effective batch 16)
- `packing=True` is set in trainer.py — packs examples to reduce padding waste
- Kill vLLM before training — it holds nearly all 128 GB

### Multi-GPU machines (A100 x2, H100 x4, etc.)
- Use `torchrun` or `accelerate` for DDP — not handled by trainer.py yet
- Increase `--batch` proportionally to GPU count
- Use `CUDA_VISIBLE_DEVICES=0,1` to select specific GPUs

---

## 9. Key Rules (from handoff — do not change without reason)

1. **NEVER fp16 with Gemma 4** — always `bf16=True, fp16=False`
2. **NEVER exceed 3 epochs** at 5K+ examples
3. **NEVER LoRA r > 16** without evidence it helps
4. **Stop vLLM before training** — VRAM conflict
5. Proven hyperparameters: `lr=2e-4`, `r=16`, `alpha=32`, `epochs=2`, `warmup_ratio=0.05`
