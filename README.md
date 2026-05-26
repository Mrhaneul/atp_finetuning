# ATP 2-01.3 — Fine-Tuned LLM Doctrine Assistant

Fine-tune a language model on ATP 2-01.3 (Intelligence Preparation of the Battlefield, March 2019)
to serve as an on-device doctrine assistant for military intelligence questions.

Two notebooks are provided — pick the one that matches your hardware:

| Notebook | Hardware | Model | Framework | Launch |
|----------|----------|-------|-----------|--------|
| `ATP_Finetune_MLX.ipynb` | Mac Apple Silicon (M1–M4) | Gemma 4 E4B | MLX + LoRA | |
| `ATP_Finetune_Colab.ipynb` | Google Colab (free T4 GPU) | Gemma 2-2B | Unsloth + QLoRA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mrhaneul/atp_finetuning/blob/main/ATP_Finetune_Colab.ipynb) |

---

## Pipeline

```
PDF → Chunks → Enrich → QA Pairs → Format → Train → Evaluate → Export GGUF
  1       2        3         4         5        6         7           8
```

| Stage | Script | Output |
|-------|--------|--------|
| 1 — Chunk | `scripts/chunker.py` | `data/chunks.jsonl` |
| 2 — Enrich | `scripts/enricher.py` | `data/enriched.jsonl` |
| 3 — Generate | `scripts/generator.py` | `data/seeds.jsonl` |
| 4 — Format | `scripts/formatter.py` | `data/train.jsonl` + `data/val.jsonl` |
| 5 — Train | `scripts/trainer.py` | `outputs/<run_name>/` |
| 6 — Evaluate | `scripts/eval.py` | `eval/results/<run_name>.json` |
| 7 — DPO (optional) | `scripts/dpo.py` | `outputs/<run_name>-dpo/` |
| 8 — Export | `scripts/burn_gguf.py` | `burns/model.Q4_K_M.gguf` |

---

## Quick Start (MLX — Apple Silicon)

**Prerequisites:** Mac with M-series chip, [Ollama](https://ollama.com) running with `gemma4:latest` pulled.

```bash
# 1. Create conda environment
conda create -n caimll_finetuning python=3.11 -y
conda activate caimll_finetuning

# 2. Install dependencies
pip install -r requirements.txt
pip install mlx-lm

# 3. Place your PDF in the project root
cp /path/to/ATP_2-01.3.pdf .

# 4. Run the full pipeline
python scripts/run_pipeline.py --pdf ATP_2-01.3.pdf

# 5. Deploy with Ollama
cd burns/atp-gemma4-e4b-mlx-v1
ollama create atp-doctrine -f Modelfile
ollama run atp-doctrine
```

**Run individual stages:**
```bash
python scripts/run_pipeline.py --pdf ATP_2-01.3.pdf --start-from generate
python scripts/run_pipeline.py --start-from train --run-name atp-gemma4-v1
python scripts/run_pipeline.py --start-from eval --adapter outputs/atp-gemma4-v1
python scripts/run_pipeline.py --pdf ATP_2-01.3.pdf --run-dpo
```

---

## Quick Start (Colab — T4 GPU)

1. Open `ATP_Finetune_Colab.ipynb` in Google Colab
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Accept the [Gemma 2 license](https://huggingface.co/google/gemma-2-2b-it) on HuggingFace
4. Run cells top to bottom — upload `ATP_2-01.3.pdf` when prompted in Step 2

---

## Evaluation Metrics

The MLX pipeline uses **keyword coverage scoring** (0.0–1.0) against 24 hardcoded doctrine questions.
Questions scoring below 0.5 are flagged as DPO candidates.

| Metric | Score |
|--------|-------|
| Base model avg score | TBD |
| Fine-tuned avg score | TBD |
| Improvement | TBD |
| DPO candidates | TBD |

---

## Key Design Choices

- **LLM-generated QA pairs** — Ollama (Gemma 4 31B) generates questions with reasoning traces, not rule-based keyword matching
- **Reasoning traces** — Every MLX training example includes a `<|channel>thought` trace showing how to derive the answer from doctrine
- **Citation enforcement** — All answers end with `[Reference: ATP 2-01.3, para X-Y]`
- **DPO refinement** — Questions scoring below 0.5 automatically feed into a Direct Preference Optimization pass
- **Distributed generation** — `generator.py` supports chapter-based sharding across multiple machines (DGX Spark, 4090 PCs, Mac Studio)
- **No cloud required** — MLX pipeline runs entirely on-device on Apple Silicon

---

## Project Structure

```
├── scripts/
│   ├── chunker.py          # Stage 1: PDF → doctrine chunks
│   ├── enricher.py         # Stage 2: metadata classification
│   ├── generator.py        # Stage 3: QA generation (Ollama / vLLM)
│   ├── generator_appendix.py  # Appendix-specific QA generation
│   ├── merge_seeds.py      # Merge seeds from distributed generation
│   ├── formatter.py        # Stage 4: chat template formatting
│   ├── trainer.py          # Stage 5: QLoRA SFT training
│   ├── eval.py             # Stage 6: keyword coverage evaluation
│   ├── dpo.py              # Stage 7: DPO on weak questions
│   ├── burn_gguf.py        # Stage 8: GGUF export for Ollama
│   ├── monitor.py          # Live training monitor
│   └── run_pipeline.py     # Pipeline orchestrator
├── Streamlit_code/
│   ├── app.py              # Streamlit chat interface
│   └── response_cleaning.py
├── eval/
│   └── questions.json      # 24 evaluation questions
├── data/                   # Generated (not committed)
├── ATP_Finetune_MLX.ipynb  # Apple Silicon notebook
├── ATP_Finetune_Colab.ipynb  # Google Colab notebook
├── requirements.txt
└── RUNBOOK.md              # Detailed operational notes
```

---

## Requirements

```
requests>=2.31.0
pypdf>=6.0.0
rouge-score>=0.1.2
matplotlib
numpy
```

MLX-specific (install separately):
```bash
pip install mlx-lm
```

Colab-specific:
```bash
pip install unsloth pdfplumber datasets trl
```
