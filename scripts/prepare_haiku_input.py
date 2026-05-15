#!/usr/bin/env python3
"""Prepare lightweight JSONL payloads for Haiku LLM issue synthesis.

Extracts from filtered commits the minimal information Haiku needs to generate
natural language issue descriptions, without requiring git repo access at
synthesis time.

Input:  filtered commits JSONL (output of filter_commits.py / parse_commits.py)
Output: JSONL with one record per commit containing:
  - commit_hash, repo, original_message, cleaned_message
  - changed_files (list of paths)
  - diff_summary (truncated hunk content, stripped of diff headers)
  - commit_type (keyword-based classification)

Usage:
    python prepare_haiku_input.py \
        --input data/filtered_commits/django_django.jsonl \
        --output data/haiku_input/django_django.jsonl \
        --max-diff-chars 2000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Commit message cleaning (same logic as generate_tasks.py)
# ---------------------------------------------------------------------------


def clean_commit_message(msg: str) -> str:
    """Strip ticket references, hash refs, and cleanup commit messages.

    Removes patterns like:
      - "Fixed #37095 -- "
      - "Refs #35870 -- "
      - "Closes #12345 -- "
      - "Fixed #37092, Refs #35870 -- "
      - "Follow-up to 63c56cda..."
    """
    # Remove "Fixed/Refs/Closes #NNN(, Fixed/Refs/Closes #NNN)* -- " prefix
    msg = re.sub(
        r"^(?:(?:Fixed|Refs|Closes)\s+#\d+(?:,\s*)?)+\s*--\s*",
        "",
        msg,
        flags=re.IGNORECASE,
    )
    # Remove "Follow-up to <hash>..." references
    msg = re.sub(r"Follow-up to [0-9a-f]{6,40}\.?", "", msg, flags=re.IGNORECASE)
    # Remove standalone ticket references like "(#12345)"
    msg = re.sub(r"\(#\d+\)", "", msg)
    # Remove inline ticket references like "Refs #12345", "(Refs #12345)", "refs #12345"
    msg = re.sub(
        r"\(?(?:Refs|Fixed|Closes|Related to)\s+#\d+\)?", "", msg, flags=re.IGNORECASE
    )
    # Remove CVE references prefix like "Fixed CVE-2026-6907 -- "
    msg = re.sub(r"^Fixed\s+CVE-[\d-]+\s*--\s*", "", msg, flags=re.IGNORECASE)
    return msg.strip()


# ---------------------------------------------------------------------------
# Commit type classification (same logic as generate_tasks.py)
# ---------------------------------------------------------------------------


def classify_commit_type(message: str) -> str:
    """Classify commit type from message keywords."""
    msg_lower = message.lower()

    if any(
        kw in msg_lower
        for kw in ["fixed", "fix", "bug", "crash", "error", "prevent", "cve"]
    ):
        return "bug_fix"
    elif any(
        kw in msg_lower for kw in ["added", "add", "support", "implement", "new"]
    ):
        return "feature"
    elif any(
        kw in msg_lower
        for kw in ["refactor", "clean", "simplify", "reorganiz", "move"]
    ):
        return "refactor"
    elif any(kw in msg_lower for kw in ["doc", "readme", "comment", "typo"]):
        return "docs"
    elif any(kw in msg_lower for kw in ["test", "assert"]):
        return "test"
    elif any(
        kw in msg_lower for kw in ["deprecat", "remov", "adjust", "updat", "improv"]
    ):
        return "enhancement"
    else:
        return "enhancement"


# ---------------------------------------------------------------------------
# Diff processing
# ---------------------------------------------------------------------------


def extract_diff_summary(patch: str, max_chars: int) -> str:
    """Extract hunk content from a patch, stripping diff/index/file headers.

    Keeps only the actual hunk content (@@ lines and +/- lines) to give Haiku
    enough context about what changed without verbose metadata.
    """
    if not patch:
        return ""

    lines = patch.split("\n")
    output_lines: list[str] = []
    total_chars = 0

    for line in lines:
        # Skip diff --git headers
        if line.startswith("diff --git "):
            continue
        # Skip index lines (e.g., "index 6d67e18..5bf4cf1 100644")
        if line.startswith("index "):
            continue
        # Skip --- and +++ file path lines
        if line.startswith("--- ") or line.startswith("+++ "):
            continue

        # Keep everything else: @@ hunk headers, +/- content, context lines
        line_len = len(line) + 1  # +1 for newline
        if total_chars + line_len > max_chars:
            # Add as much of this line as we can fit
            remaining = max_chars - total_chars
            if remaining > 10:  # only add if there's meaningful space
                output_lines.append(line[:remaining] + "...")
            break
        output_lines.append(line)
        total_chars += line_len

    return "\n".join(output_lines)


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------


def extract_changed_files(commit: dict) -> list[str]:
    """Extract file paths from commit data.

    Handles both field name conventions:
      - changed_files: [{filepath: ..., is_test: ...}]  (parse_commits.py)
      - src_files: [{path: ...}]  (generate_tasks.py normalized)
    """
    paths: list[str] = []

    # Try changed_files first (parse_commits.py format)
    if "changed_files" in commit:
        for f in commit["changed_files"]:
            fp = f.get("filepath", f.get("path", ""))
            if fp:
                paths.append(fp)

    # Fallback: src_files (already normalized format)
    elif "src_files" in commit:
        for f in commit["src_files"]:
            fp = f.get("path", "")
            if fp:
                paths.append(fp)

    return paths


def get_patch(commit: dict) -> str:
    """Get the combined patch (source + test) from commit data."""
    source_patch = commit.get("source_patch", commit.get("src_patch", ""))
    test_patch = commit.get("test_patch", "")

    if source_patch and test_patch:
        return source_patch + "\n" + test_patch
    return source_patch or test_patch or ""


def process_commit(commit: dict, repo_name: str, max_diff_chars: int) -> dict:
    """Transform a single commit into a Haiku input record."""
    original_message = commit.get("commit_message", commit.get("message", ""))
    cleaned = clean_commit_message(original_message)

    changed_files = extract_changed_files(commit)
    patch = get_patch(commit)
    diff_summary = extract_diff_summary(patch, max_diff_chars)
    commit_type = classify_commit_type(original_message)

    # Quality-based synthesis decision:
    #   - Short/empty cleaned messages need Haiku rewrite
    #   - Medium messages benefit from rewrite
    #   - Long detailed messages are likely high quality, keep original
    cleaned_len = len(cleaned)
    if cleaned_len < 30:
        needs_synthesis = "must"
        quality = "low"
    elif cleaned_len < 100:
        needs_synthesis = "recommended"
        quality = "medium"
    else:
        needs_synthesis = "optional"
        quality = "high"

    return {
        "commit_hash": commit.get("commit_hash", ""),
        "repo": repo_name,
        "original_message": original_message,
        "cleaned_message": cleaned,
        "cleaned_message_len": cleaned_len,
        "needs_synthesis": needs_synthesis,
        "quality": quality,
        "changed_files": changed_files,
        "diff_summary": diff_summary,
        "commit_type": commit_type,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def derive_repo_name(input_path: str) -> str:
    """Derive repo name from input filename (e.g., django_django.jsonl -> django_django)."""
    basename = os.path.basename(input_path)
    name, _ = os.path.splitext(basename)
    return name


def main():
    parser = argparse.ArgumentParser(
        description="Prepare lightweight JSONL for Haiku LLM issue synthesis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to filtered commits JSONL file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for output JSONL file",
    )
    parser.add_argument(
        "--max-diff-chars",
        type=int,
        default=2000,
        help="Maximum characters for diff_summary field (default: 2000)",
    )
    parser.add_argument(
        "--repo-name",
        default=None,
        help="Repository name (default: derived from input filename)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    repo_name = args.repo_name or derive_repo_name(args.input)
    max_diff_chars = args.max_diff_chars

    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Repo:   {repo_name}")
    logger.info(f"Max diff chars: {max_diff_chars}")

    count = 0
    errors = 0

    with open(input_path, "r", encoding="utf-8") as fin, open(
        output_path, "w", encoding="utf-8"
    ) as fout:
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                commit = json.loads(line)
                record = process_commit(commit, repo_name, max_diff_chars)
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
            except (json.JSONDecodeError, KeyError) as e:
                errors += 1
                if errors <= 5:
                    logger.warning(f"Line {line_num}: {e}")

    # Print quality distribution stats
    logger.info(f"Done! Wrote {count} records, {errors} errors.")
    logger.info(f"Output: {output_path}")

    # Read back to compute quality stats
    quality_counts = {"low": 0, "medium": 0, "high": 0}
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            q = rec.get("quality", "medium")
            quality_counts[q] = quality_counts.get(q, 0) + 1

    logger.info(f"Quality distribution:")
    logger.info(f"  Low  (must synthesize, <30 chars):       {quality_counts['low']}")
    logger.info(f"  Med  (recommended, 30-100 chars):        {quality_counts['medium']}")
    logger.info(f"  High (optional, >100 chars):             {quality_counts['high']}")


if __name__ == "__main__":
    main()
