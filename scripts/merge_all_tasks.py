#!/usr/bin/env python3
"""Merge all task directories into a single unified directory for dedup.

Scans all data/tasks/*/ directories and merges files by task_type (filename).
Handles deduplication of identical entries (same commit + same output).

Usage:
    python scripts/merge_all_tasks.py
    python scripts/merge_all_tasks.py --output-dir data/tasks_merged
    python scripts/merge_all_tasks.py --exclude-dirs django,django_v2_full,django_v2_dedup,django_django_new
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
TASKS_DIR = BASE_DIR / "data" / "tasks"


def merge_tasks(output_dir: Path, exclude_dirs: set[str] = None):
    """Merge all task dirs into one, deduplicating by content hash."""
    exclude_dirs = exclude_dirs or set()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all source directories
    source_dirs = []
    for d in sorted(TASKS_DIR.iterdir()):
        if not d.is_dir():
            continue
        if d.name in exclude_dirs:
            continue
        if d == output_dir:
            continue
        source_dirs.append(d)

    print(f"Found {len(source_dirs)} source directories")

    # Track task types and counts
    task_type_files = defaultdict(list)  # task_type -> list of source files
    for d in source_dirs:
        for f in d.glob("*.jsonl"):
            task_type_files[f.stem].append(f)

    print(f"Task types found: {list(task_type_files.keys())}")

    total_input = 0
    total_output = 0
    total_dupes = 0

    for task_type, files in sorted(task_type_files.items()):
        print(f"\n=== {task_type} ===")
        print(f"  Source files: {len(files)}")

        seen_hashes = set()
        output_file = output_dir / f"{task_type}.jsonl"
        count_in = 0
        count_out = 0
        count_dupe = 0

        with open(output_file, "w") as out:
            for f in files:
                try:
                    with open(f) as inp:
                        for line in inp:
                            line = line.strip()
                            if not line:
                                continue
                            count_in += 1

                            # Hash the full line for exact dedup
                            h = hashlib.md5(line.encode()).hexdigest()
                            if h in seen_hashes:
                                count_dupe += 1
                                continue
                            seen_hashes.add(h)

                            # Also try to deduplicate by commit+output
                            try:
                                obj = json.loads(line)
                                commit = obj.get("commit", obj.get("commit_hash", ""))
                                output_text = obj.get("output", obj.get("answer", ""))
                                content_key = hashlib.md5(
                                    f"{commit}:{output_text}".encode()
                                ).hexdigest()
                                if content_key in seen_hashes:
                                    count_dupe += 1
                                    continue
                                seen_hashes.add(content_key)
                            except (json.JSONDecodeError, TypeError):
                                pass

                            out.write(line + "\n")
                            count_out += 1
                except Exception as e:
                    print(f"  WARNING: Error reading {f}: {e}")

        total_input += count_in
        total_output += count_out
        total_dupes += count_dupe
        pct = count_dupe / count_in * 100 if count_in > 0 else 0
        print(f"  Input: {count_in:,} | Output: {count_out:,} | Dupes removed: {count_dupe:,} ({pct:.1f}%)")

    print(f"\n{'=' * 60}")
    print(f"TOTAL Input: {total_input:,}")
    print(f"TOTAL Output: {total_output:,}")
    print(f"TOTAL Dupes removed: {total_dupes:,} ({total_dupes / total_input * 100:.1f}%)")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Merge all task directories")
    parser.add_argument("--output-dir", type=str, default="data/tasks_merged",
                        help="Output directory for merged files")
    parser.add_argument("--exclude-dirs", type=str, default="django,django_v2_full,django_v2_dedup,django_django_new,tasks_merged,tasks_dedup",
                        help="Comma-separated dir names to exclude")
    args = parser.parse_args()

    output_dir = BASE_DIR / args.output_dir if not args.output_dir.startswith("/") else Path(args.output_dir)
    exclude_dirs = set(args.exclude_dirs.split(",")) if args.exclude_dirs else set()

    merge_tasks(output_dir, exclude_dirs)


if __name__ == "__main__":
    main()
