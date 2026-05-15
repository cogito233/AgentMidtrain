#!/usr/bin/env python3
"""Convert merged task files to SQA (system/query/answer) format for training.

Usage:
    python scripts/convert_to_sqa.py
    python scripts/convert_to_sqa.py --input-dir data/tasks_merged_test --output-dir data/training_sqa_v2
"""

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def convert_file(input_file: Path, output_file: Path) -> dict:
    """Convert a single JSONL task file to SQA format."""
    count = 0
    errors = 0

    with open(input_file) as inp, open(output_file, "w") as out:
        for line in inp:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)

                # Map to SQA format
                sqa = {
                    "system": obj.get("prompt", ""),
                    "query": obj.get("input", ""),
                    "answer": obj.get("output", ""),
                    "task_type": obj.get("task_type", input_file.stem),
                    "repo": obj.get("repo", ""),
                    "commit": obj.get("commit", ""),
                }

                # Skip empty entries
                if not sqa["query"] or not sqa["answer"]:
                    errors += 1
                    continue

                out.write(json.dumps(sqa, ensure_ascii=False) + "\n")
                count += 1
            except (json.JSONDecodeError, Exception) as e:
                errors += 1
                continue

    return {"file": input_file.name, "count": count, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description="Convert tasks to SQA format")
    parser.add_argument("--input-dir", type=str, default="data/tasks_merged_test",
                        help="Input directory with merged JSONL files")
    parser.add_argument("--output-dir", type=str, default="data/training_sqa_v2",
                        help="Output directory for SQA files")
    args = parser.parse_args()

    input_dir = BASE_DIR / args.input_dir if not args.input_dir.startswith("/") else Path(args.input_dir)
    output_dir = BASE_DIR / args.output_dir if not args.output_dir.startswith("/") else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {input_dir} → {output_dir}")
    print("=" * 60)

    total = 0
    for f in sorted(input_dir.glob("*.jsonl")):
        result = convert_file(f, output_dir / f.name)
        total += result["count"]
        print(f"  {result['file']}: {result['count']:,} samples ({result['errors']} errors)")

    print("=" * 60)
    print(f"Total SQA samples: {total:,}")
    print(f"Output: {output_dir}")

    # Also create a single combined file for convenience
    combined = output_dir / "all_combined.jsonl"
    with open(combined, "w") as out:
        for f in sorted(output_dir.glob("*.jsonl")):
            if f.name == "all_combined.jsonl":
                continue
            with open(f) as inp:
                for line in inp:
                    out.write(line)

    combined_count = sum(1 for _ in open(combined))
    print(f"Combined file: {combined} ({combined_count:,} lines)")


if __name__ == "__main__":
    main()
