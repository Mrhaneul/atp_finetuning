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
| 4 — Format | `scripts/formatter.py` | `data/train.jsonl` + `data/valid.jsonl` |
| 5 — Train | `mlx_lm lora` | `outputs/<run_name>/` |
| 6 — Evaluate | `scripts/evaluator.py` | `eval/results.jsonl` |
| 7 — Plot | `scripts/plot_eval.py` | `eval/rouge_chart.png` |
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

## Evaluation Results (MLX run)

| Metric | Score |
|--------|-------|
| Base model ROUGE-L | 0.1104 |
| Fine-tuned ROUGE-L | 0.1364 |
| Improvement | **+23.6%** |
| Examples improved | **20 / 28** |

![ROUGE-L Chart](eval/rouge_chart.png)

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
│   ├── generator.py        # Stage 3: QA generation (Ollama)
│   ├── formatter.py        # Stage 4: chat template formatting
│   ├── evaluator.py        # Stage 6: ROUGE-L evaluation
│   ├── plot_eval.py        # Stage 7: evaluation chart
│   ├── burn_gguf.py        # Stage 8: GGUF export for Ollama
│   └── run_pipeline.py     # Pipeline orchestrator
├── eval/
│   └── rouge_chart.png     # Evaluation results chart
├── data/                   # Generated (not committed)
├── ATP_Finetune_MLX.ipynb  # Apple Silicon notebook
├── ATP_Finetune_Colab.ipynb  # Google Colab notebook
├── run_overnight.sh        # One-command overnight run
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
