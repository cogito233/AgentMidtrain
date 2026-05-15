#!/usr/bin/env python3
"""Convert SQA format to OpenAI-style messages JSONL for SFT training.

Input format (SQA):
    {"system": "...", "query": "...", "answer": "...", "task_type": "...", "repo": "...", "commit": "..."}

Output format (messages):
    {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Usage:
    python scripts/convert_to_messages.py \
        --input-dir data/training_sqa_v6 \
        --output-dir data/training_messages_v6 \
        --task-types localization,commit_message,edit_generation,test_writing
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def convert_line(line: str) -> str | None:
    """Convert a single SQA JSON line to messages format."""
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None

    system = obj.get("system", "").strip()
    query = obj.get("query", "").strip()
    answer = obj.get("answer", "").strip()

    if not query or not answer:
        return None

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": query})
    messages.append({"role": "assistant", "content": answer})

    out = {"messages": messages}

    # Pack all non-core fields into a metadata dict
    metadata = {}
    for key in ("task_type", "repo", "commit"):
        if key in obj:
            metadata[key] = obj[key]
    # Include any extra fields not already handled
    for key, val in obj.items():
        if key not in ("system", "query", "answer", "task_type", "repo", "commit"):
            metadata[key] = val
    if metadata:
        out["metadata"] = metadata

    return json.dumps(out, ensure_ascii=False)


def convert_file(input_path: Path, output_path: Path) -> tuple[int, int]:
    """Convert a single JSONL file. Returns (success_count, error_count)."""
    success = 0
    errors = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            result = convert_line(line)
            if result:
                fout.write(result + "\n")
                success += 1
            else:
                errors += 1

    return success, errors


def main():
    parser = argparse.ArgumentParser(description="Convert SQA to OpenAI messages format")
    parser.add_argument("--input-dir", required=True, help="Input SQA directory")
    parser.add_argument("--output-dir", required=True, help="Output messages directory")
    parser.add_argument(
        "--task-types",
        default="localization,commit_message,edit_generation,test_writing",
        help="Comma-separated task types to include (default: 4 recommended types)",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Also produce a combined all_train.jsonl file",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_types = [t.strip() for t in args.task_types.split(",")]

    total_success = 0
    total_errors = 0
    combined_path = output_dir / "all_train.jsonl" if args.combined else None
    combined_fh = open(combined_path, "w", encoding="utf-8") if combined_path else None

    logger.info(f"Converting {input_dir} → {output_dir}")
    logger.info(f"Task types: {task_types}")

    for task_type in task_types:
        input_path = input_dir / f"{task_type}.jsonl"
        if not input_path.exists():
            logger.warning(f"  {task_type}.jsonl not found, skipping")
            continue

        output_path = output_dir / f"{task_type}.jsonl"
        success, errors = convert_file(input_path, output_path)
        total_success += success
        total_errors += errors
        logger.info(f"  {task_type}: {success:,} samples ({errors} errors)")

        # Append to combined file
        if combined_fh:
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    combined_fh.write(line)

    if combined_fh:
        combined_fh.close()
        logger.info(f"  Combined: {total_success:,} samples → {combined_path}")

    logger.info("=" * 60)
    logger.info(f"Total: {total_success:,} samples, {total_errors} errors")
    logger.info(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
