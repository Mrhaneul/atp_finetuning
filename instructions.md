# ATP 2-01.3 Distributed Generation — End-to-End Instructions

## Current state

| Stage | File | Status |
|---|---|---|
| Stage 1 | `chunks.jsonl` | ✅ 690 chunks |
| Stage 2 | `enriched.jsonl` | ✅ 690 chunks enriched |
| Stage 3 | `seeds.jsonl` | ⬜ not started — this is what we're doing |

## Machines

| Access | Hostname | IP | OS | Role |
|---|---|---|---|---|
| here | spark-ce3e | 100.118.193.70 | Linux | Spark 1 → chapters 7, 8 |
| SSH | spark-87dc | 100.76.51.83 | Linux | Spark 2 → chapters 5, 6 |
| SSH | cbu11760 | 100.119.29.29 | macOS | Mac → appendices A, B, C |
| git only | csds-pc-1 | 100.75.150.116 | Windows | PC1 → chapters 1, 4 |
| git only | csds-pc-2 | 100.113.51.103 | Windows | PC2 → chapters 2, 3, D |

## Why 1 vLLM instance per Spark

BF16 Gemma 4 31B needs ~62 GB. The GB10 has 128 GB unified memory.
Two instances would leave only 4 GB for KV cache — vLLM would crash.
One instance with `--gpu-memory-utilization 0.90` gives 53 GB of KV cache — plenty.

The throughput trick: run **two generator.py processes simultaneously** against the
same vLLM instance, each on different chapters. vLLM batches their concurrent
requests automatically. Full BF16 precision, no quantization.

## Work split

| Output file | Machine | Port | Chapters | Target |
|---|---|---|---|---|
| seeds_spark1a.jsonl | spark-ce3e | 8000 | 7 | 1600 |
| seeds_spark1b.jsonl | spark-ce3e | 8000 | 8 | 400 |
| seeds_spark2a.jsonl | spark-87dc | 8000 | 5 | 800 |
| seeds_spark2b.jsonl | spark-87dc | 8000 | 6 | 700 |
| seeds_mac.jsonl | cbu11760 | Ollama | A, B, C | 300 |
| seeds_pc1.jsonl | csds-pc-1 | Ollama | 1, 4 | 1000 |
| seeds_pc2.jsonl | csds-pc-2 | Ollama | 2, 3, D | 700 |
| **Total** | | | | **5500 → ~4,400 after filter** |

---

## Part 1 — Push pipeline to git (on this Spark, do first)

Create a **private** GitHub repo, then:

```bash
cd /home/student/caimll_finetuning/atp_pipeline_v2

cat > .gitignore << 'EOF'
__pycache__/
*.pyc
*.pdf
pipeline_run.log
data/chunks.jsonl
EOF

git init
git add generator.py merge_seeds.py requirements.txt .gitignore data/enriched.jsonl
git commit -m "atp pipeline v2 - distributed generation setup"
git remote add origin https://github.com/YOUR_ORG/YOUR_REPO.git
git push -u origin main
```

---

## Part 2 — Spark 1: spark-ce3e (this machine)

```bash
cd /home/student/caimll_finetuning/atp_pipeline_v2

# Stop Ollama so it doesn't hold VRAM when vLLM loads
sudo systemctl stop ollama

# One-time: install vLLM and log into HuggingFace
pip install vllm
huggingface-cli login
# Accept Gemma 4 license at hf.co/google/gemma-4-31b-it first, then paste your HF token

# Download model weights (~62 GB)
python -c "from huggingface_hub import snapshot_download; snapshot_download('google/gemma-4-31b-it')"

# Start ONE vLLM instance
tmux new-session -d -s vllm
tmux send-keys -t vllm "vllm serve google/gemma-4-31b-it \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --port 8000" Enter

# Watch until "Application startup complete" (~2 min), then Ctrl-B D to detach
tmux attach -t vllm

# Run TWO generators simultaneously against the same vLLM instance
tmux new-session -d -s gen_ch7
tmux send-keys -t gen_ch7 "cd /home/student/caimll_finetuning/atp_pipeline_v2 && \
  python generator.py \
    --backend vllm --api-url http://localhost:8000 \
    --machine-id spark1a --chapters 7 --target 1600 \
    --out data/seeds_spark1a.jsonl 2>&1 | tee gen_spark1a.log" Enter

tmux new-session -d -s gen_ch8
tmux send-keys -t gen_ch8 "cd /home/student/caimll_finetuning/atp_pipeline_v2 && \
  python generator.py \
    --backend vllm --api-url http://localhost:8000 \
    --machine-id spark1b --chapters 8 --target 400 \
    --out data/seeds_spark1b.jsonl 2>&1 | tee gen_spark1b.log" Enter
```

---

## Part 3 — Spark 2: spark-87dc (100.76.51.83)

```bash
ssh student@100.76.51.83

# One-time setup
pip install vllm huggingface_hub requests
huggingface-cli login
python -c "from huggingface_hub import snapshot_download; snapshot_download('google/gemma-4-31b-it')"

git clone https://github.com/YOUR_ORG/YOUR_REPO.git ~/atp_pipeline_v2

# Start ONE vLLM instance
tmux new-session -d -s vllm
tmux send-keys -t vllm "vllm serve google/gemma-4-31b-it \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --port 8000" Enter

# Wait for "Application startup complete", then Ctrl-B D to detach
tmux attach -t vllm

# Run TWO generators simultaneously
tmux new-session -d -s gen_ch5
tmux send-keys -t gen_ch5 "cd ~/atp_pipeline_v2 && \
  python generator.py \
    --backend vllm --api-url http://localhost:8000 \
    --machine-id spark2a --chapters 5 --target 800 \
    --out data/seeds_spark2a.jsonl 2>&1 | tee gen_spark2a.log" Enter

tmux new-session -d -s gen_ch6
tmux send-keys -t gen_ch6 "cd ~/atp_pipeline_v2 && \
  python generator.py \
    --backend vllm --api-url http://localhost:8000 \
    --machine-id spark2b --chapters 6 --target 700 \
    --out data/seeds_spark2b.jsonl 2>&1 | tee gen_spark2b.log" Enter

exit
```

---

## Part 4 — Mac: cbu11760 (100.119.29.29)

```bash
ssh YOUR_MAC_USERNAME@100.119.29.29
# Don't know the username? Run `whoami` on the Mac itself

# One-time setup
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma4:31b
pip3 install requests

git clone https://github.com/YOUR_ORG/YOUR_REPO.git ~/atp_pipeline_v2

# Run generator (nohup so it survives SSH disconnect)
cd ~/atp_pipeline_v2
nohup python3 generator.py \
  --machine-id mac \
  --chapters A,B,C \
  --target 300 \
  --out data/seeds_mac.jsonl \
  > gen_mac.log 2>&1 &

echo "Mac generation running, PID: $!"
exit
```

---

## Part 5 — PC1: csds-pc-1 (someone runs this locally or via RDP)

```powershell
winget install Ollama.Ollama
ollama pull gemma4:31b
pip install requests

git clone https://github.com/YOUR_ORG/YOUR_REPO.git atp_pipeline_v2
cd atp_pipeline_v2

python generator.py `
  --machine-id pc1 `
  --chapters 1,4 `
  --target 1000 `
  --out data/seeds_pc1.jsonl

# When done — push results back
git add data/seeds_pc1.jsonl
git commit -m "pc1 done"
git push
```

> If the PC has WSL2, run identical Linux commands inside WSL instead.

---

## Part 6 — PC2: csds-pc-2 (someone runs this locally or via RDP)

```powershell
winget install Ollama.Ollama
ollama pull gemma4:31b
pip install requests

git clone https://github.com/YOUR_ORG/YOUR_REPO.git atp_pipeline_v2
cd atp_pipeline_v2

python generator.py `
  --machine-id pc2 `
  --chapters 2,3,D `
  --target 700 `
  --out data/seeds_pc2.jsonl

git add data/seeds_pc2.jsonl
git commit -m "pc2 done"
git push
```

---

## Part 7 — Monitor (from this Spark anytime)

```bash
# This Spark
echo "=== spark1a ===" && wc -l /home/student/caimll_finetuning/atp_pipeline_v2/data/seeds_spark1a.jsonl 2>/dev/null
echo "=== spark1b ===" && wc -l /home/student/caimll_finetuning/atp_pipeline_v2/data/seeds_spark1b.jsonl 2>/dev/null

# Spark 2
ssh student@100.76.51.83 "wc -l ~/atp_pipeline_v2/data/seeds_spark2*.jsonl 2>/dev/null"

# Mac
ssh YOUR_MAC_USERNAME@100.119.29.29 "wc -l ~/atp_pipeline_v2/data/seeds_mac.jsonl 2>/dev/null"

# PCs — check what's been pushed to git
git fetch && git log --oneline | grep -E "pc[12] done"
```

Attach to any live tmux session to watch output:
```bash
tmux attach -t gen_ch7    # on this Spark
# Ctrl-B then D to detach without stopping it
```

Any run that crashes is safe to restart with the exact same command — the generator
resumes from where it left off automatically.

---

## Part 8 — Collect and merge (once all machines finish)

```bash
cd /home/student/caimll_finetuning/atp_pipeline_v2

# Pull PC results from git
git pull

# SCP from Spark 2 and Mac
scp student@100.76.51.83:"~/atp_pipeline_v2/data/seeds_spark2*.jsonl" data/
scp YOUR_MAC_USERNAME@100.119.29.29:"~/atp_pipeline_v2/data/seeds_mac.jsonl" data/

# Confirm all 7 files are present
ls -lh data/seeds_*.jsonl

# Preview counts without writing
python merge_seeds.py --dry-run

# Write final merged file → data/seeds.jsonl
python merge_seeds.py

# Verify
wc -l data/seeds.jsonl
```

`data/seeds.jsonl` is the input for Stage 4 (formatter + trainer).
