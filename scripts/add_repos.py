#!/usr/bin/env python3
"""Clone new repos and run the full 3-stage pipeline (parse -> filter -> generate).

Usage:
    # From a file of repo URLs (one per line: owner/repo)
    python scripts/add_repos.py --repo-list repos_to_add.txt

    # Specific repos
    python scripts/add_repos.py --repos celery/celery,encode/django-rest-framework

    # With language filter for non-Python repos
    python scripts/add_repos.py --repos golang/go --lang go --ext .go
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

BASE_DIR = Path("/data_fast_v3/eremite/cogito_explore/AgentMidtrain")
REPOS_DIR = BASE_DIR / "repos"
DATA_DIR = BASE_DIR / "data"
FILTERED_DIR = DATA_DIR / "filtered_commits"
TASKS_DIR = DATA_DIR / "tasks"
SCRIPTS_DIR = BASE_DIR / "scripts"
LOG_DIR = BASE_DIR / "logs"


def clone_repo(owner_repo: str, shallow: bool = False) -> dict:
    """Clone a repo into repos/ directory."""
    repo_name = owner_repo.replace("/", "_")
    repo_dir = REPOS_DIR / repo_name

    if repo_dir.exists() and (repo_dir / ".git").exists():
        return {"repo": repo_name, "status": "exists", "path": str(repo_dir)}

    url = f"https://github.com/{owner_repo}.git"
    cmd = ["git", "clone"]
    if shallow:
        cmd += ["--depth", "50000"]  # Enough history for good commit coverage
    cmd += [url, str(repo_dir)]

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        elapsed = time.time() - t0
        if proc.returncode != 0:
            return {
                "repo": repo_name,
                "status": "clone_failed",
                "error": proc.stderr[-300:],
                "elapsed": elapsed,
            }
        return {"repo": repo_name, "status": "cloned", "path": str(repo_dir), "elapsed": elapsed}
    except subprocess.TimeoutExpired:
        return {"repo": repo_name, "status": "timeout"}
    except Exception as e:
        return {"repo": repo_name, "status": "error", "error": str(e)}


def run_parse(repo_name: str, workers: int = 8, lang: str = "python") -> dict:
    """Run parse_commits.py on a repo."""
    repo_dir = REPOS_DIR / repo_name
    output_file = FILTERED_DIR / f"{repo_name}_raw.jsonl"

    if output_file.exists() and output_file.stat().st_size > 0:
        lines = sum(1 for _ in open(output_file))
        return {"repo": repo_name, "status": "exists", "count": lines}

    FILTERED_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(SCRIPTS_DIR / "parse_commits.py"),
        str(repo_dir),
        "--output-file", str(output_file),
        "--workers", str(workers),
    ]

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, cwd=str(BASE_DIR))
        elapsed = time.time() - t0
        if proc.returncode != 0:
            return {"repo": repo_name, "status": "failed", "error": proc.stderr[-300:], "elapsed": elapsed}

        lines = sum(1 for _ in open(output_file)) if output_file.exists() else 0
        return {"repo": repo_name, "status": "ok", "count": lines, "elapsed": elapsed}
    except subprocess.TimeoutExpired:
        return {"repo": repo_name, "status": "timeout"}


def run_filter(repo_name: str, lang: str = "python") -> dict:
    """Run filter_commits.py on raw parsed commits."""
    raw_file = FILTERED_DIR / f"{repo_name}_raw.jsonl"
    output_file = FILTERED_DIR / f"{repo_name}.jsonl"

    if output_file.exists() and output_file.stat().st_size > 0:
        lines = sum(1 for _ in open(output_file))
        return {"repo": repo_name, "status": "exists", "count": lines}

    if not raw_file.exists():
        return {"repo": repo_name, "status": "no_raw"}

    cmd = [
        sys.executable, str(SCRIPTS_DIR / "filter_commits.py"),
        str(raw_file),
        "-o", str(output_file),
        "--require-test",
        "--require-src",
        "--fix-keywords",
        "--max-src-files", "5",
    ]

    # Add language-specific filters
    if lang == "python":
        cmd.append("--python-only")

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, cwd=str(BASE_DIR))
        elapsed = time.time() - t0
        if proc.returncode != 0:
            return {"repo": repo_name, "status": "failed", "error": proc.stderr[-300:], "elapsed": elapsed}

        lines = sum(1 for _ in open(output_file)) if output_file.exists() else 0
        return {"repo": repo_name, "status": "ok", "count": lines, "elapsed": elapsed}
    except subprocess.TimeoutExpired:
        return {"repo": repo_name, "status": "timeout"}


def run_generate(repo_name: str, workers: int = 8) -> dict:
    """Run generate_tasks.py on filtered commits."""
    filtered_file = FILTERED_DIR / f"{repo_name}.jsonl"
    output_dir = TASKS_DIR / repo_name

    if not filtered_file.exists():
        return {"repo": repo_name, "status": "no_filtered"}

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(SCRIPTS_DIR / "generate_tasks.py"),
        "--input", str(filtered_file),
        "--output-dir", str(output_dir),
        "--repo-path", str(REPOS_DIR / repo_name),
        "--workers", str(workers),
    ]

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, cwd=str(BASE_DIR))
        elapsed = time.time() - t0
        if proc.returncode != 0:
            return {"repo": repo_name, "status": "failed", "error": proc.stderr[-300:], "elapsed": elapsed}

        # Count outputs
        counts = {}
        for jf in output_dir.glob("*.jsonl"):
            with open(jf) as f:
                counts[jf.stem] = sum(1 for _ in f)

        return {
            "repo": repo_name,
            "status": "ok",
            "counts": counts,
            "total": sum(counts.values()),
            "elapsed": elapsed,
        }
    except subprocess.TimeoutExpired:
        return {"repo": repo_name, "status": "timeout"}


def process_single_repo(owner_repo: str, lang: str = "python", workers: int = 4) -> dict:
    """Full pipeline for a single repo: clone -> parse -> filter -> generate."""
    repo_name = owner_repo.replace("/", "_")
    result = {"owner_repo": owner_repo, "repo_name": repo_name, "stages": {}}

    # Stage 1: Clone
    print(f"  [{repo_name}] Cloning...")
    clone_result = clone_repo(owner_repo)
    result["stages"]["clone"] = clone_result
    if clone_result["status"] not in ("cloned", "exists"):
        result["final_status"] = "clone_failed"
        return result

    # Stage 2: Parse
    print(f"  [{repo_name}] Parsing commits...")
    parse_result = run_parse(repo_name, workers=workers, lang=lang)
    result["stages"]["parse"] = parse_result
    if parse_result["status"] not in ("ok", "exists"):
        result["final_status"] = "parse_failed"
        return result

    # Stage 3: Filter
    print(f"  [{repo_name}] Filtering...")
    filter_result = run_filter(repo_name, lang=lang)
    result["stages"]["filter"] = filter_result
    if filter_result["status"] not in ("ok", "exists"):
        result["final_status"] = "filter_failed"
        return result

    # Stage 4: Generate tasks
    print(f"  [{repo_name}] Generating tasks...")
    gen_result = run_generate(repo_name, workers=workers)
    result["stages"]["generate"] = gen_result
    if gen_result["status"] == "ok":
        result["final_status"] = "ok"
        result["total_samples"] = gen_result.get("total", 0)
        print(f"  [{repo_name}] DONE: {gen_result.get('total', 0)} samples")
    else:
        result["final_status"] = "generate_failed"

    return result


def main():
    parser = argparse.ArgumentParser(description="Clone and process new repos")
    parser.add_argument("--repos", type=str, help="Comma-separated owner/repo list")
    parser.add_argument("--repo-list", type=str, help="File with owner/repo lines")
    parser.add_argument("--lang", default="python", help="Language filter (default: python)")
    parser.add_argument("--parallel", "-p", type=int, default=2,
                        help="Parallel repos (default: 2, keep low for clone)")
    parser.add_argument("--workers", "-w", type=int, default=4,
                        help="Workers per repo for parse/generate")
    args = parser.parse_args()

    # Collect repo list
    repos: list[str] = []
    if args.repos:
        repos = [r.strip() for r in args.repos.split(",") if r.strip()]
    elif args.repo_list:
        with open(args.repo_list) as f:
            repos = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    else:
        print("Error: provide --repos or --repo-list")
        sys.exit(1)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    REPOS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(repos)} repos (lang={args.lang}, parallel={args.parallel})")
    print("=" * 60)

    all_results = []
    total_samples = 0

    if args.parallel <= 1:
        for repo in repos:
            result = process_single_repo(repo, lang=args.lang, workers=args.workers)
            all_results.append(result)
            total_samples += result.get("total_samples", 0)
    else:
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(process_single_repo, repo, args.lang, args.workers): repo
                for repo in repos
            }
            for future in as_completed(futures):
                result = future.result()
                all_results.append(result)
                total_samples += result.get("total_samples", 0)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    ok = [r for r in all_results if r["final_status"] == "ok"]
    failed = [r for r in all_results if r["final_status"] != "ok"]
    print(f"  Succeeded: {len(ok)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Total samples: {total_samples:,}")

    for r in ok:
        print(f"    {r['repo_name']}: {r.get('total_samples', 0):,} samples")

    if failed:
        print("\n  Failed repos:")
        for r in failed:
            print(f"    {r['repo_name']}: {r['final_status']}")

    # Save results
    results_path = LOG_DIR / f"add_repos_{int(time.time())}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {results_path}")


if __name__ == "__main__":
    main()
