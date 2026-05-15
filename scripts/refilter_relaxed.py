#!/usr/bin/env python3
"""Re-filter existing raw commits with relaxed criteria to increase data volume.

The strict filter (--require-test --fix-keywords --python-only) yields ~3-7% of commits.
This relaxed filter drops --require-test and --fix-keywords, keeping:
  - --require-src (must change at least one source file)
  - --python-only (only .py files)
  - --exclude-docs (no comment-only changes)
  - --max-src-files 8 (slightly relaxed from 5)
  - --max-edit-lines 300 (slightly relaxed from 200)

Expected yield: 15-30% of raw commits (3-5x more than strict filter).

Usage:
    python scripts/refilter_relaxed.py                    # Process all repos
    python scripts/refilter_relaxed.py --repos django_django,sympy_sympy  # Specific repos
    python scripts/refilter_relaxed.py --generate         # Also regenerate tasks
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
FILTERED_DIR = BASE_DIR / "data" / "filtered_commits"
TASKS_DIR = BASE_DIR / "data" / "tasks"
REPOS_DIR = BASE_DIR / "repos"
SCRIPTS_DIR = BASE_DIR / "scripts"


def refilter_repo(repo_name: str) -> dict:
    """Re-filter a repo's raw commits with relaxed settings."""
    raw_file = FILTERED_DIR / f"{repo_name}_raw.jsonl"
    relaxed_file = FILTERED_DIR / f"{repo_name}_relaxed.jsonl"

    if not raw_file.exists():
        return {"repo": repo_name, "status": "no_raw"}

    # Skip if relaxed file already exists and is newer than raw
    if relaxed_file.exists() and relaxed_file.stat().st_mtime > raw_file.stat().st_mtime:
        lines = sum(1 for _ in open(relaxed_file))
        return {"repo": repo_name, "status": "exists", "count": lines}

    cmd = [
        sys.executable, str(SCRIPTS_DIR / "filter_commits.py"),
        str(raw_file),
        "-o", str(relaxed_file),
        "--require-src",
        "--python-only",
        "--exclude-docs",
        "--max-src-files", "8",
        "--max-edit-lines", "300",
        "--max-patch-length", "15000",
    ]

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(BASE_DIR))
        elapsed = time.time() - t0
        if proc.returncode != 0:
            return {"repo": repo_name, "status": "failed", "error": proc.stderr[-300:], "elapsed": elapsed}

        lines = sum(1 for _ in open(relaxed_file)) if relaxed_file.exists() else 0
        return {"repo": repo_name, "status": "ok", "count": lines, "elapsed": elapsed}
    except subprocess.TimeoutExpired:
        return {"repo": repo_name, "status": "timeout"}


def generate_relaxed(repo_name: str, workers: int = 4) -> dict:
    """Generate tasks from relaxed-filtered commits."""
    relaxed_file = FILTERED_DIR / f"{repo_name}_relaxed.jsonl"
    output_dir = TASKS_DIR / f"{repo_name}_relaxed"

    if not relaxed_file.exists():
        return {"repo": repo_name, "status": "no_relaxed"}

    repo_path = REPOS_DIR / repo_name
    if not repo_path.exists():
        return {"repo": repo_name, "status": "no_repo"}

    output_dir.mkdir(parents=True, exist_ok=True)

    # Only generate task types that don't need test files:
    # localization, edit_generation, commit_message, code_review
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "generate_tasks.py"),
        "--input", str(relaxed_file),
        "--output-dir", str(output_dir),
        "--repo-path", str(repo_path),
        "--workers", str(workers),
        "--task-types", "localization,edit_generation,commit_message,code_review,bug_detection",
    ]

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, cwd=str(BASE_DIR))
        elapsed = time.time() - t0
        if proc.returncode != 0:
            return {"repo": repo_name, "status": "failed", "error": proc.stderr[-300:], "elapsed": elapsed}

        counts = {}
        for jf in output_dir.glob("*.jsonl"):
            with open(jf) as f:
                counts[jf.stem] = sum(1 for _ in f)

        return {
            "repo": repo_name, "status": "ok",
            "counts": counts, "total": sum(counts.values()), "elapsed": elapsed,
        }
    except subprocess.TimeoutExpired:
        return {"repo": repo_name, "status": "timeout"}


def main():
    parser = argparse.ArgumentParser(description="Re-filter with relaxed criteria")
    parser.add_argument("--repos", type=str, help="Comma-separated repo names (e.g. django_django)")
    parser.add_argument("--generate", action="store_true", help="Also generate tasks")
    parser.add_argument("--workers", type=int, default=4, help="Workers for generation")
    args = parser.parse_args()

    # Get list of repos to process
    if args.repos:
        repo_names = [r.strip() for r in args.repos.split(",") if r.strip()]
    else:
        # Find all repos with raw files
        repo_names = []
        for f in sorted(FILTERED_DIR.glob("*_raw.jsonl")):
            name = f.stem.replace("_raw", "")
            repo_names.append(name)

    print(f"Processing {len(repo_names)} repos with relaxed filter...")
    print("=" * 60)

    total_filtered = 0
    total_generated = 0

    for repo_name in repo_names:
        # Step 1: Re-filter
        result = refilter_repo(repo_name)
        count = result.get("count", 0)
        total_filtered += count
        status = result["status"]
        print(f"  [{repo_name}] Filter: {status} ({count} commits)")

        # Step 2: Generate (if requested)
        if args.generate and count > 0:
            gen_result = generate_relaxed(repo_name, workers=args.workers)
            gen_total = gen_result.get("total", 0)
            total_generated += gen_total
            print(f"  [{repo_name}] Generate: {gen_result['status']} ({gen_total} samples)")

    print("=" * 60)
    print(f"Total filtered commits: {total_filtered:,}")
    if args.generate:
        print(f"Total generated samples: {total_generated:,}")


if __name__ == "__main__":
    main()
