#!/usr/bin/env python3
"""Batch regenerate bug_detection for all repos and refine with Sonnet.

Two-phase pipeline:
1. Regenerate bug_detection.jsonl for each repo using generate_tasks.py (v10)
2. Refine/validate using refine_bug_detection.py with claude-sonnet-4.6

Usage:
    # Full pipeline: regenerate + refine
    python scripts/batch_regen_bugdet.py

    # Only refine existing bug_detection (skip regeneration)
    python scripts/batch_regen_bugdet.py --refine-only

    # Only regenerate (skip refinement)
    python scripts/batch_regen_bugdet.py --regen-only

    # Specific repos
    python scripts/batch_regen_bugdet.py --repos django_django,sympy_sympy
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path("/data_fast_v3/eremite/cogito_explore/AgentMidtrain")
REPOS_DIR = BASE_DIR / "repos"
DATA_DIR = BASE_DIR / "data"
FILTERED_DIR = DATA_DIR / "filtered_commits"
TASKS_DIR = DATA_DIR / "tasks"
SCRIPTS_DIR = BASE_DIR / "scripts"
LOG_DIR = BASE_DIR / "logs"

SKIP_REPOS = {"r2e_gym", "swe_bench"}

DEFAULT_GATEWAY_URL = "http://106.54.223.20:8000"
DEFAULT_MODEL = "claude-sonnet-4.6"


def discover_repos(specific_repos: list[str] | None = None) -> list[str]:
    """Find all valid repos."""
    all_repos = []
    for entry in sorted(REPOS_DIR.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in SKIP_REPOS:
            continue
        if not (entry / ".git").exists():
            continue
        all_repos.append(entry.name)

    if specific_repos:
        requested = set(specific_repos)
        return [r for r in all_repos if r in requested]
    return all_repos


def regen_bug_detection(repo_name: str, workers: int = 4) -> dict:
    """Regenerate bug_detection.jsonl for a single repo."""
    filtered_path = FILTERED_DIR / f"{repo_name}.jsonl"
    output_dir = TASKS_DIR / repo_name

    if not filtered_path.exists():
        return {"repo": repo_name, "status": "skipped", "reason": "no filtered commits"}

    output_dir.mkdir(parents=True, exist_ok=True)

    # Back up existing bug_detection if present
    existing = output_dir / "bug_detection.jsonl"
    if existing.exists():
        backup = output_dir / "bug_detection_pre_v10.jsonl.bak"
        if not backup.exists():
            existing.rename(backup)

    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "generate_tasks.py"),
        "--input", str(filtered_path),
        "--output-dir", str(output_dir),
        "--repo-path", str(REPOS_DIR / repo_name),
        "--task-types", "bug_detection",
        "--workers", str(workers),
    ]

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600, cwd=str(BASE_DIR)
        )
        elapsed = time.time() - t0

        if proc.returncode != 0:
            return {
                "repo": repo_name,
                "status": "failed",
                "error": proc.stderr[-500:],
                "elapsed": elapsed,
            }

        # Count output
        count = 0
        if existing.exists():
            with open(existing) as f:
                count = sum(1 for _ in f)

        return {
            "repo": repo_name,
            "status": "ok",
            "count": count,
            "elapsed": elapsed,
        }

    except subprocess.TimeoutExpired:
        return {"repo": repo_name, "status": "timeout", "elapsed": 3600}
    except Exception as e:
        return {"repo": repo_name, "status": "error", "error": str(e)}


def refine_bug_detection(
    repo_name: str,
    gateway_url: str = DEFAULT_GATEWAY_URL,
    model: str = DEFAULT_MODEL,
) -> dict:
    """Run LLM refinement on bug_detection.jsonl for a single repo."""
    input_path = TASKS_DIR / repo_name / "bug_detection.jsonl"
    output_path = TASKS_DIR / repo_name / "bug_detection_refined.jsonl"

    if not input_path.exists():
        return {"repo": repo_name, "status": "skipped", "reason": "no bug_detection.jsonl"}

    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "refine_bug_detection.py"),
        "--input", str(input_path),
        "--output", str(output_path),
        "--gateway-url", gateway_url,
        "--model", model,
    ]

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=7200, cwd=str(BASE_DIR)
        )
        elapsed = time.time() - t0

        if proc.returncode != 0:
            return {
                "repo": repo_name,
                "status": "failed",
                "error": proc.stderr[-500:],
                "elapsed": elapsed,
            }

        # Count output
        count_in = 0
        count_out = 0
        with open(input_path) as f:
            count_in = sum(1 for _ in f)
        if output_path.exists():
            with open(output_path) as f:
                count_out = sum(1 for _ in f)

        return {
            "repo": repo_name,
            "status": "ok",
            "input_count": count_in,
            "output_count": count_out,
            "pass_rate": f"{count_out/count_in*100:.1f}%" if count_in > 0 else "N/A",
            "elapsed": elapsed,
        }

    except subprocess.TimeoutExpired:
        return {"repo": repo_name, "status": "timeout", "elapsed": 7200}
    except Exception as e:
        return {"repo": repo_name, "status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Batch regenerate and refine bug_detection for all repos"
    )
    parser.add_argument("--repos", type=str, default=None,
                        help="Comma-separated repo names (default: all)")
    parser.add_argument("--regen-only", action="store_true",
                        help="Only regenerate, skip refinement")
    parser.add_argument("--refine-only", action="store_true",
                        help="Only refine existing bug_detection, skip regen")
    parser.add_argument("--parallel", "-p", type=int, default=4,
                        help="Parallel repos for regen phase (default: 4)")
    parser.add_argument("--refine-parallel", type=int, default=2,
                        help="Parallel repos for refine phase (default: 2)")
    parser.add_argument("--workers", "-w", type=int, default=4,
                        help="Workers per repo for generation (default: 4)")
    parser.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    specific_repos = None
    if args.repos:
        specific_repos = [r.strip() for r in args.repos.split(",")]

    repos = discover_repos(specific_repos)
    print(f"Target repos: {len(repos)}")
    print(f"  {', '.join(repos)}")
    print()

    # Phase 1: Regenerate
    if not args.refine_only:
        print("=" * 70)
        print("PHASE 1: Regenerating bug_detection (v10)")
        print("=" * 70)

        regen_results = []
        t0 = time.time()

        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(regen_bug_detection, repo, args.workers): repo
                for repo in repos
            }
            for future in as_completed(futures):
                result = future.result()
                regen_results.append(result)
                status = result["status"]
                count = result.get("count", "?")
                elapsed = result.get("elapsed", 0)
                print(f"  {result['repo']}: {status} "
                      f"({count} samples, {elapsed:.1f}s)")

        total_regen = sum(r.get("count", 0) for r in regen_results if r["status"] == "ok")
        print(f"\nPhase 1 done: {total_regen} total samples, "
              f"{time.time()-t0:.0f}s elapsed")
        print()

    # Phase 2: Refine with LLM
    if not args.regen_only:
        print("=" * 70)
        print(f"PHASE 2: Refining with {args.model}")
        print("=" * 70)

        refine_results = []
        t0 = time.time()

        # Refine sequentially or with limited parallelism
        # (API rate limits make high parallelism counterproductive)
        with ProcessPoolExecutor(max_workers=args.refine_parallel) as executor:
            futures = {
                executor.submit(
                    refine_bug_detection, repo, args.gateway_url, args.model
                ): repo
                for repo in repos
            }
            for future in as_completed(futures):
                result = future.result()
                refine_results.append(result)
                status = result["status"]
                if status == "ok":
                    print(f"  {result['repo']}: {result['input_count']} → "
                          f"{result['output_count']} ({result['pass_rate']}), "
                          f"{result.get('elapsed', 0):.0f}s")
                else:
                    reason = result.get("reason", result.get("error", "unknown"))
                    print(f"  {result['repo']}: {status} - {reason}")

        total_in = sum(r.get("input_count", 0) for r in refine_results if r["status"] == "ok")
        total_out = sum(r.get("output_count", 0) for r in refine_results if r["status"] == "ok")
        print(f"\nPhase 2 done: {total_in} → {total_out} "
              f"({total_out/total_in*100:.1f}% pass rate)" if total_in > 0 else "")
        print(f"Total time: {time.time()-t0:.0f}s")

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    total_refined = 0
    for repo in repos:
        refined_path = TASKS_DIR / repo / "bug_detection_refined.jsonl"
        raw_path = TASKS_DIR / repo / "bug_detection.jsonl"
        raw_count = 0
        refined_count = 0
        if raw_path.exists():
            with open(raw_path) as f:
                raw_count = sum(1 for _ in f)
        if refined_path.exists():
            with open(refined_path) as f:
                refined_count = sum(1 for _ in f)
            total_refined += refined_count
        print(f"  {repo}: {raw_count} raw → {refined_count} refined")

    print(f"\n  TOTAL REFINED: {total_refined}")
    print("=" * 70)


if __name__ == "__main__":
    main()
