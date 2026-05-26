"""
merge_seeds.py
==============
Post-generation utility: merge per-machine seeds_*.jsonl files into a single
deduplicated data/seeds.jsonl.

Deduplication key: (source_chunks, question_type)
If two machines generated a pair for the same paragraph + question type, the
first one encountered is kept (files are processed alphabetically so spark
files come before pc/mac files).

Usage:
    # Run on any one machine after copying all seeds_*.jsonl into data/
    python merge_seeds.py

    # Custom glob or output path:
    python merge_seeds.py --pattern "data/seeds_*.jsonl" --out data/seeds.jsonl

    # Dry run — show counts without writing:
    python merge_seeds.py --dry-run
"""

import json
import glob
import argparse
from pathlib import Path


def merge(pattern: str, out_path: str, dry_run: bool = False) -> None:
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[merge] No files matched: {pattern}")
        return

    print(f"[merge] Found {len(files)} shard files:")
    for f in files:
        print(f"  {f}")

    seen: set[str] = set()        # dedup key = str(source_chunks) + question_type
    records: list[dict] = []

    total_read = 0
    per_file: dict[str, int] = {}

    for path in files:
        file_count = 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    continue
                total_read += 1
                key = str(record.get("source_chunks", [])) + record.get("question_type", "")
                if key not in seen:
                    seen.add(key)
                    records.append(record)
                    file_count += 1
        per_file[path] = file_count

    print(f"\n[merge] Read {total_read} records total")
    print(f"[merge] {total_read - len(records)} duplicates removed")
    print(f"[merge] {len(records)} unique QA pairs")

    if dry_run:
        print("[merge] DRY RUN — nothing written")
        for path, count in per_file.items():
            print(f"  {Path(path).name}: {count} unique pairs contributed")
        return

    # Renumber qa_ids sequentially in the merged file
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for i, record in enumerate(records, 1):
            record["qa_id"] = f"atp201_qa_{i:05d}"
            fh.write(json.dumps(record) + "\n")

    print(f"[merge] Wrote {len(records)} records → {out_path}")
    for path, count in per_file.items():
        print(f"  {Path(path).name}: {count} unique pairs contributed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge per-machine seeds_*.jsonl into a single deduplicated seeds.jsonl"
    )
    parser.add_argument("--pattern",  default="data/seeds_*.jsonl",
                        help="Glob pattern for shard files (default: data/seeds_*.jsonl)")
    parser.add_argument("--out",      default="data/seeds.jsonl",
                        help="Output path (default: data/seeds.jsonl)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Show counts without writing the output file")
    args = parser.parse_args()
    merge(args.pattern, args.out, args.dry_run)
