"""
burn_gguf.py
============
Merge a Gemma 4 31B LoRA adapter into the base model and export GGUF
for Ollama deployment.

CRITICAL — Gemma 4 specific:
  - NEVER fp16 — Gemma 4 activations overflow fp16. Always bf16 or f32.
  - NEVER hardcode base model path — read from adapter_config.json.
  - Load Unsloth environment before running (not system Python).

Quant options:
    q4_k_m  — default, best size/quality for Ollama (~17 GB)
    q8_0    — higher quality, larger (~32 GB)
    q5_k_m  — middle ground
    f16     — keep BF16 GGUF without additional quantization

After the burn, load into Ollama with:
    cd <output_dir>
    ollama create atp-gemma4 -f Modelfile
    ollama run atp-gemma4

Usage:
    python burn_gguf.py --adapter outputs/atp-gemma4-31b-v1
    python burn_gguf.py --adapter outputs/atp-gemma4-31b-dpo --quant q8_0
    python burn_gguf.py --adapter outputs/atp-gemma4-31b-v1 --out burns/atp-gemma4-v1
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ── Defaults ──────────────────────────────────────────────────

DEFAULT_ADAPTER = Path("outputs") / "atp-gemma4-31b-v1"
LLAMA_CPP_DIR   = Path("/home/student/.unsloth/llama.cpp")

DEFAULT_SYSTEM_PROMPT = (
    "You are a doctrine-grounded military intelligence assistant specialized in "
    "ATP 2-01.3 (Intelligence Preparation of the Battlefield, March 2019). "
    "Answer questions about IPB using ATP 2-01.3 doctrine only. "
    "Place all citations at the END of your answer in "
    "[Reference: ATP 2-01.3, para X-Y] format — NEVER at the beginning. "
    "If a question cannot be answered from ATP 2-01.3, say so clearly."
)

QUANT_MAP = {
    "f16":   None,       # keep BF16 as-is
    "q8_0":  "Q8_0",
    "q4_k_m":"Q4_K_M",
    "q5_k_m":"Q5_K_M",
}


# ── Helpers ───────────────────────────────────────────────────

def read_json(path: Path) -> dict:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def is_adapter_dir(path: Path) -> bool:
    return path.is_dir() and (path / "adapter_config.json").exists()


def is_hf_model_dir(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").exists():
        return False
    return (
        (path / "model.safetensors").exists()
        or (path / "model.safetensors.index.json").exists()
        or any(path.glob("model-*.safetensors"))
    )


def fp16_base(model_id: str) -> str:
    """Strip Unsloth 4-bit suffix to get the full-precision model id."""
    return model_id.replace("-unsloth-bnb-4bit", "").replace("-bnb-4bit", "")


def resolve_base_model(raw_base: str | None, override: str | None = None) -> Path:
    """Resolve the full-precision base model from a local cache or HF hub."""
    candidates = []
    if override:
        candidates.append(override)
    if raw_base:
        stripped = fp16_base(raw_base)
        candidates.append(stripped)
        if stripped != raw_base:
            candidates.append(raw_base)

    tried = []
    for candidate in candidates:
        tried.append(candidate)
        # Try as local path
        p = Path(candidate).expanduser()
        if p.exists():
            return p.resolve()
        # Try HF hub cache
        try:
            from huggingface_hub import snapshot_download
            snapshot = snapshot_download(repo_id=candidate, local_files_only=True)
            return Path(snapshot).resolve()
        except Exception:
            pass

    print("ERROR: Could not resolve full-precision base model.")
    if raw_base:
        print(f"  adapter_config base: {raw_base}")
    print("  Tried:")
    for t in tried:
        print(f"    - {t}")
    print("Use --base-model to provide the local HF path explicitly.")
    sys.exit(1)


def resolve_llama_cpp_tools() -> tuple[Path, Path]:
    converter_candidates = [
        LLAMA_CPP_DIR / "convert_hf_to_gguf.py",
        LLAMA_CPP_DIR / "unsloth_convert_hf_to_gguf.py",
    ]
    quantizer_candidates = [
        LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize",
        LLAMA_CPP_DIR / "llama-quantize",
    ]
    converter = next((p for p in converter_candidates if p.exists()), None)
    quantizer = next((p for p in quantizer_candidates if p.exists()), None)
    if converter is None:
        print(f"ERROR: convert_hf_to_gguf.py not found under {LLAMA_CPP_DIR}")
        sys.exit(1)
    if quantizer is None:
        print(f"ERROR: llama-quantize not found under {LLAMA_CPP_DIR}")
        sys.exit(1)
    return converter, quantizer


# ── Merge ─────────────────────────────────────────────────────

def merge_adapter_to_hf(
    adapter_path:       Path,
    output_dir:         Path,
    max_seq_length:     int,
    base_model_override:str | None,
) -> Path:
    """
    Load base model + adapter via Unsloth and save merged 16-bit HF weights.
    CRITICAL: save_pretrained_merged uses bf16 (Unsloth default for Gemma 4).
    """
    try:
        from unsloth import FastLanguageModel
    except ModuleNotFoundError:
        print("ERROR: 'unsloth' not found. Activate the caimll_finetuning environment.")
        sys.exit(1)

    # Read base model from adapter_config.json — NEVER hardcode
    config   = read_json(adapter_path / "adapter_config.json")
    raw_base = config.get("base_model_name_or_path")
    if not raw_base and not base_model_override:
        print("ERROR: adapter_config.json missing 'base_model_name_or_path'")
        sys.exit(1)

    base_model = resolve_base_model(raw_base, base_model_override)
    merged_dir = output_dir / "merged_hf"

    print(f"  Source kind : adapter")
    print(f"  Base model  : {base_model}")
    print(f"  Adapter     : {adapter_path}")
    print(f"  Merge output: {merged_dir}")

    tmp_adapter = Path(tempfile.mkdtemp(prefix="burn_gemma4_"))
    try:
        shutil.copytree(adapter_path, tmp_adapter, dirs_exist_ok=True)
        # Patch config to point at resolved local path
        patched = {**config, "base_model_name_or_path": str(base_model)}
        with open(tmp_adapter / "adapter_config.json", "w", encoding="utf-8") as f:
            json.dump(patched, f, indent=2)

        if merged_dir.exists():
            shutil.rmtree(merged_dir)
        merged_dir.mkdir(parents=True, exist_ok=True)

        print("\n[1/3] Loading base + adapter via Unsloth (load_in_4bit=False for merge)...")
        # CRITICAL: load_in_4bit=False for merge to get full-precision weights
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name     = str(tmp_adapter),
            max_seq_length = max_seq_length,
            load_in_4bit   = False,     # must be False for merge
            dtype          = None,      # auto (bf16 on Gemma 4 capable hardware)
        )

        print("[2/3] Merging and saving as merged_16bit (bf16)...")
        # Unsloth's merged_16bit output respects bf16 for Gemma 4
        model.save_pretrained_merged(
            str(merged_dir),
            tokenizer,
            save_method="merged_16bit",
        )
        del model
        return merged_dir

    finally:
        shutil.rmtree(tmp_adapter, ignore_errors=True)


# ── Convert + Quantize ────────────────────────────────────────

def convert_to_gguf(
    hf_model_dir: Path,
    output_dir:   Path,
    quant:        str,
    step_label:   str,
) -> list[Path]:
    converter, quantizer = resolve_llama_cpp_tools()
    bf16_gguf = output_dir / "model.BF16.gguf"

    print(f"\n{step_label} Converting HF weights to GGUF (bf16)...")
    subprocess.run(
        [
            sys.executable, str(converter),
            str(hf_model_dir),
            "--outfile", str(bf16_gguf),
            "--outtype", "bf16",
        ],
        check=True,
    )

    quant_type = QUANT_MAP[quant]
    if quant_type is None:
        # f16 — keep bf16 GGUF as-is
        final_gguf = bf16_gguf
    else:
        final_gguf = output_dir / f"model.{quant.upper()}.gguf"
        print(f"    Quantizing to {quant_type}...")
        subprocess.run(
            [str(quantizer), str(bf16_gguf), str(final_gguf), quant_type],
            check=True,
        )
        if bf16_gguf.exists():
            bf16_gguf.unlink()

    size_gb = final_gguf.stat().st_size / 1e9
    print(f"\n  GGUF saved: {final_gguf} ({size_gb:.2f} GB)")
    return [final_gguf]


# ── Modelfile ─────────────────────────────────────────────────

def write_modelfile(
    output_dir:    Path,
    gguf_files:    list[Path],
    model_name:    str,
    system_prompt: str,
) -> None:
    gguf = gguf_files[0]
    modelfile_path = output_dir / "Modelfile"
    modelfile_path.write_text(
        f'FROM ./{gguf.name}\n\n'
        f'SYSTEM """{system_prompt}"""\n\n'
        f"PARAMETER temperature 0.1\n"
        f"PARAMETER top_p 0.9\n"
        f"PARAMETER top_k 64\n"
        f"PARAMETER num_ctx 4096\n"
        f"PARAMETER repeat_penalty 1.1\n",
        encoding="utf-8",
    )
    print(f"\n  Modelfile: {modelfile_path}")
    print("\n  To deploy in Ollama:")
    print(f"    cd {output_dir}")
    print(f"    ollama create {model_name} -f Modelfile")
    print(f"    ollama run {model_name}")


# ── Main ──────────────────────────────────────────────────────

def burn(
    source_path:        Path,
    output_dir:         Path,
    quant:              str,
    max_seq_length:     int,
    base_model_override:str | None,
) -> tuple[list[Path], Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_adapter_dir(source_path):
        hf_model_dir    = merge_adapter_to_hf(
            source_path, output_dir, max_seq_length, base_model_override
        )
        cleanup_merged  = True
        step_label      = "[3/3]"
    elif is_hf_model_dir(source_path):
        hf_model_dir    = source_path
        cleanup_merged  = False
        step_label      = "[1/1]"
        print(f"\n  Source: merged HF model ({source_path})")
    else:
        print(f"ERROR: {source_path} is neither a LoRA adapter nor a merged HF model dir")
        sys.exit(1)

    try:
        gguf_files = convert_to_gguf(hf_model_dir, output_dir, quant, step_label)
    finally:
        if cleanup_merged:
            shutil.rmtree(hf_model_dir, ignore_errors=True)

    return gguf_files, output_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge Gemma 4 31B LoRA adapter into GGUF for Ollama"
    )
    parser.add_argument(
        "--adapter", "--source", dest="adapter",
        default=str(DEFAULT_ADAPTER) if DEFAULT_ADAPTER.exists() else None,
        help="LoRA adapter directory or merged HF model directory",
    )
    parser.add_argument(
        "--base-model", default=None,
        help="Override base model path (reads from adapter_config.json by default)",
    )
    parser.add_argument(
        "--quant", default="q4_k_m", choices=sorted(QUANT_MAP),
        help="Quantization format (default: q4_k_m)",
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=4096,
        help="Sequence length for Unsloth merge (default: 4096)",
    )
    parser.add_argument(
        "--system-prompt", default=DEFAULT_SYSTEM_PROMPT,
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory (default: ./burns/<adapter_name>)",
    )
    args = parser.parse_args()
    if not args.adapter:
        parser.error("--adapter is required (no default adapter found)")
    return args


def main():
    args = parse_args()

    source_path = Path(args.adapter).expanduser().resolve()
    if not source_path.exists():
        print(f"ERROR: {source_path} not found")
        sys.exit(1)

    model_name = source_path.name
    output_dir = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (Path.cwd() / "burns" / model_name)
    )

    print("=" * 56)
    print("  ATP 2-01.3 Gemma 4 31B GGUF Burn")
    print("=" * 56)
    print(f"  Source  : {source_path}")
    print(f"  Quant   : {args.quant}")
    print(f"  Output  : {output_dir}")

    gguf_files, output_dir = burn(
        source_path         = source_path,
        output_dir          = output_dir,
        quant               = args.quant,
        max_seq_length      = args.max_seq_length,
        base_model_override = args.base_model,
    )
    write_modelfile(output_dir, gguf_files, model_name, args.system_prompt)


if __name__ == "__main__":
    main()
