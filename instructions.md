End-to-End Distributed Generation
Where things stand
Stage 1 ✅ chunks.jsonl — 690 chunks
Stage 2 ✅ enriched.jsonl — 690 chunks enriched
Stage 3 ⬜ seeds.jsonl — not started, this is what we're doing
Your machines:

Machine	IP	OS	Role
You are here	spark-ce3e	100.118.193.70	Linux	Spark 1 → chapters 7, 8
SSH	spark-87dc	100.76.51.83	Linux	Spark 2 → chapters 5, 6
SSH	cbu11760	100.119.29.29	macOS	Mac → appendices A, B, C
git only	csds-pc-1	100.75.150.116	Windows	PC1 → chapters 1, 4
git only	csds-pc-2	100.113.51.103	Windows	PC2 → chapters 2, 3, D
Part 1 — Push pipeline to git (do this first, on this Spark)
Create a private repo on GitHub first, then:


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
Part 2 — Set up Spark 2 (SSH from this machine)

# Confirm username — if it's not student, adjust
ssh student@100.76.51.83
Once inside spark-87dc:


# Install vLLM
pip install vllm huggingface_hub requests

# Log into HuggingFace (need to accept Gemma 4 license first at hf.co/google/gemma-4-31b-it)
huggingface-cli login
# paste your HF token

# Download model (~62 GB, do this now so it's ready)
python -c "from huggingface_hub import snapshot_download; snapshot_download('google/gemma-4-31b-it')"

# Clone pipeline
git clone https://github.com/YOUR_ORG/YOUR_REPO.git ~/atp_pipeline_v2

# Start vLLM server in its own tmux pane
tmux new-session -d -s vllm -x 220 -y 50
tmux send-keys -t vllm "vllm serve google/gemma-4-31b-it --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.90 --port 8000" Enter

# Wait until you see "Application startup complete" — takes ~2 min
# Watch the vLLM logs:
tmux attach -t vllm
# Detach with Ctrl-B then D once it says startup complete

# Start generation in a second pane
tmux new-session -d -s gen -x 220 -y 50
tmux send-keys -t gen "cd ~/atp_pipeline_v2 && python generator.py --backend vllm --api-url http://localhost:8000 --machine-id spark2 --chapters 5,6 --target 1500 --out data/seeds_spark2.jsonl 2>&1 | tee gen.log" Enter

exit
Part 3 — Set up Mac (SSH from this machine)

ssh YOUR_MAC_USERNAME@100.119.29.29
# Not sure of the username? Check the Mac login screen or ask whoever uses it
# On the Mac itself: whoami
Once inside the Mac:


# Install Ollama if not present
curl -fsSL https://ollama.com/install.sh | sh

# Pull Gemma 4 31B (~20 GB Q4 quantized — fits in 32 GB RAM)
ollama pull gemma4:31b

pip3 install requests

# Clone pipeline
git clone https://github.com/YOUR_ORG/YOUR_REPO.git ~/atp_pipeline_v2

# Start generation (use nohup so it survives SSH disconnect)
cd ~/atp_pipeline_v2
nohup python3 generator.py \
  --machine-id mac \
  --chapters A,B,C \
  --target 300 \
  --out data/seeds_mac.jsonl \
  > gen.log 2>&1 &

echo "PID: $!"  # save this in case you need to kill it later
exit
Part 4 — Set up Windows PCs (someone does this locally on each PC)
Send these instructions to whoever has access to csds-pc-1 and csds-pc-2.

On csds-pc-1 — open PowerShell:


# Install Ollama (skip if already installed)
winget install Ollama.Ollama
# Or download from ollama.com if winget isn't available

# Pull Gemma 4 — open a new PowerShell and run:
ollama pull gemma4:31b

# Install Python dependency
pip install requests

# Clone pipeline
git clone https://github.com/YOUR_ORG/YOUR_REPO.git atp_pipeline_v2
cd atp_pipeline_v2

# Run generation
python generator.py --machine-id pc1 --chapters 1,4 --target 1000 --out data/seeds_pc1.jsonl

# When done — push results back
git add data/seeds_pc1.jsonl
git commit -m "pc1 done"
git push
On csds-pc-2 — same thing, different chapters:


winget install Ollama.Ollama
ollama pull gemma4:31b
pip install requests
git clone https://github.com/YOUR_ORG/YOUR_REPO.git atp_pipeline_v2
cd atp_pipeline_v2
python generator.py --machine-id pc2 --chapters 2,3,D --target 700 --out data/seeds_pc2.jsonl
git add data/seeds_pc2.jsonl
git commit -m "pc2 done"
git push
If the PCs have WSL2 installed, the commands are the same but run them inside WSL — better compatibility and no Windows path issues.

Part 5 — Start generation on THIS Spark (spark-ce3e)
This machine has gemma4:31b already in Ollama but we'll use vLLM for better throughput. Stop Ollama first to free VRAM, then start vLLM.


cd /home/student/caimll_finetuning/atp_pipeline_v2

# Stop Ollama to free VRAM for vLLM (they'd compete otherwise)
sudo systemctl stop ollama
# or if no systemctl: pkill ollama

# Install vLLM
pip install vllm

# Log into HuggingFace (same token as Spark 2)
huggingface-cli login

# Download model (~62 GB — may already be cached if you did Spark 2 first on this same machine)
python -c "from huggingface_hub import snapshot_download; snapshot_download('google/gemma-4-31b-it')"

# Start vLLM in its own tmux pane
tmux new-session -d -s vllm -x 220 -y 50
tmux send-keys -t vllm "vllm serve google/gemma-4-31b-it --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.90 --port 8000" Enter

# Watch until "Application startup complete"
tmux attach -t vllm
# Ctrl-B then D to detach

# Start generation
tmux new-session -d -s gen -x 220 -y 50
tmux send-keys -t gen "cd /home/student/caimll_finetuning/atp_pipeline_v2 && python generator.py --backend vllm --api-url http://localhost:8000 --machine-id spark1 --chapters 7,8 --target 2000 --out data/seeds_spark1.jsonl 2>&1 | tee gen.log" Enter

# Watch it:
tmux attach -t gen
# Ctrl-B then D to detach
Part 6 — Monitor progress (from this Spark anytime)

# Check this Spark's own output
wc -l /home/student/caimll_finetuning/atp_pipeline_v2/data/seeds_spark1.jsonl

# Check Spark 2
ssh student@100.76.51.83 "wc -l ~/atp_pipeline_v2/data/seeds_spark2.jsonl 2>/dev/null && tail -1 ~/atp_pipeline_v2/gen.log"

# Check Mac
ssh YOUR_MAC_USERNAME@100.119.29.29 "wc -l ~/atp_pipeline_v2/data/seeds_mac.jsonl 2>/dev/null && tail -1 ~/atp_pipeline_v2/gen.log"

# Check PCs — see what's been pushed to git
git fetch && git log --oneline -8
If anything crashes and needs to be restarted, just rerun the exact same python generator.py ... command — it resumes from where it left off automatically.

Part 7 — Collect and merge (once all machines finish)

cd /home/student/caimll_finetuning/atp_pipeline_v2

# Pull PC results from git
git pull

# Pull from Spark 2 and Mac directly
scp student@100.76.51.83:~/atp_pipeline_v2/data/seeds_spark2.jsonl data/
scp YOUR_MAC_USERNAME@100.119.29.29:~/atp_pipeline_v2/data/seeds_mac.jsonl data/

# Confirm all 5 files are here
ls -lh data/seeds_*.jsonl

# Preview merge
python merge_seeds.py --dry-run

# Write final merged file
python merge_seeds.py

# Verify
wc -l data/seeds.jsonl
You now have data/seeds.jsonl — the input for Stage 4 (formatter + trainer).

Summary of what runs where

spark-ce3e (you)  │  vLLM  │  chapters 7,8   │  target 2000
spark-87dc        │  vLLM  │  chapters 5,6   │  target 1500
cbu11760 (Mac)    │  Ollama │  chapters A,B,C │  target  300
csds-pc-1         │  Ollama │  chapters 1,4   │  target 1000
csds-pc-2         │  Ollama │  chapters 2,3,D │  target  700
                  │         │                 │  ─────────────
                  │         │  TOTAL          │  5500 scheduled
                  │         │  ~80% pass rate │  ~4,400 final pairs