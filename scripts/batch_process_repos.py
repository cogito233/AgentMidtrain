#!/usr/bin/env python3
"""Batch processing script for the 3-stage pipeline (parse -> filter -> generate).

Discovers repos in the repos/ directory, runs all three stages for each repo
with configurable parallelism, error handling, and resume support.

Usage:
    python scripts/batch_process_repos.py --parallel 3
    python scripts/batch_process_repos.py --parallel 3 --force
    python scripts/batch_process_repos.py --repos django_django,sympy_sympy
    python scripts/batch_process_repos.py --stage generate
    python scripts/batch_process_repos.py --stage parse,filter --parallel 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path("/data_fast_v3/eremite/cogito_explore/AgentMidtrain")
REPOS_DIR = BASE_DIR / "repos"
DATA_DIR = BASE_DIR / "data"
FILTERED_DIR = DATA_DIR / "filtered_commits"
TASKS_DIR = DATA_DIR / "tasks"
SCRIPTS_DIR = BASE_DIR / "scripts"
LOG_DIR = BASE_DIR / "logs"

SKIP_REPOS = {"r2e_gym", "swe_bench"}

STAGES = ("parse", "filter", "generate")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_DIR.mkdir(parents=True, exist_ok=True)

log_file = LOG_DIR / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class StageStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class RepoResult:
    """Result of processing a single repo."""
    repo_name: str
    stages_run: list[str] = field(default_factory=list)
    stages_skipped: list[str] = field(default_factory=list)
    stages_failed: dict[str, str] = field(default_factory=dict)
    duration_secs: float = 0.0

    @property
    def success(self) -> bool:
        return len(self.stages_failed) == 0


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def raw_output_path(repo_name: str) -> Path:
    """Stage 1 output: raw parsed commits."""
    return FILTERED_DIR / f"{repo_name}_raw.jsonl"


def filtered_output_path(repo_name: str) -> Path:
    """Stage 2 output: filtered commits."""
    return FILTERED_DIR / f"{repo_name}.jsonl"


def tasks_output_dir(repo_name: str) -> Path:
    """Stage 3 output: task directory."""
    return TASKS_DIR / repo_name


def repo_path(repo_name: str) -> Path:
    """Path to the repo."""
    return REPOS_DIR / repo_name


# ---------------------------------------------------------------------------
# Stage checks (for skip logic)
# ---------------------------------------------------------------------------

def is_stage_done(repo_name: str, stage: str) -> bool:
    """Check if a stage has already produced output for a repo."""
    if stage == "parse":
        path = raw_output_path(repo_name)
        return path.exists() and path.stat().st_size > 0
    elif stage == "filter":
        path = filtered_output_path(repo_name)
        return path.exists() and path.stat().st_size > 0
    elif stage == "generate":
        task_dir = tasks_output_dir(repo_name)
        if not task_dir.exists():
            return False
        # Check if any JSONL output files exist
        jsonl_files = list(task_dir.glob("*.jsonl"))
        return len(jsonl_files) > 0
    return False


def can_run_stage(repo_name: str, stage: str) -> bool:
    """Check if prerequisites for a stage are met."""
    if stage == "parse":
        # Need the repo to exist
        rp = repo_path(repo_name)
        return rp.exists() and (rp / ".git").exists()
    elif stage == "filter":
        # Need raw output
        return raw_output_path(repo_name).exists()
    elif stage == "generate":
        # Need filtered output
        return filtered_output_path(repo_name).exists()
    return False


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def run_stage_parse(repo_name: str, workers: int = 8) -> subprocess.CompletedProcess:
    """Run stage 1: parse_commits.py."""
    FILTERED_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "parse_commits.py"),
        str(repo_path(repo_name)),
        "--output-file", str(raw_output_path(repo_name)),
        "--workers", str(workers),
    ]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(BASE_DIR),
        timeout=7200,  # 2 hour timeout for large repos
    )


def run_stage_filter(repo_name: str) -> subprocess.CompletedProcess:
    """Run stage 2: filter_commits.py."""
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "filter_commits.py"),
        str(raw_output_path(repo_name)),
        "-o", str(filtered_output_path(repo_name)),
        "--require-test",
        "--require-src",
        "--fix-keywords",
        "--python-only",
        "--max-src-files", "5",
    ]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(BASE_DIR),
        timeout=1800,  # 30 min timeout
    )


def run_stage_generate(repo_name: str, workers: int = 8) -> subprocess.CompletedProcess:
    """Run stage 3: generate_tasks.py."""
    output_dir = tasks_output_dir(repo_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "generate_tasks.py"),
        "--input", str(filtered_output_path(repo_name)),
        "--output-dir", str(output_dir),
        "--repo-path", str(repo_path(repo_name)),
        "--workers", str(workers),
    ]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(BASE_DIR),
        timeout=3600,  # 1 hour timeout
    )


STAGE_RUNNERS = {
    "parse": run_stage_parse,
    "filter": run_stage_filter,
    "generate": run_stage_generate,
}


# ---------------------------------------------------------------------------
# Per-repo processing
# ---------------------------------------------------------------------------

def process_repo(
    repo_name: str,
    stages: list[str],
    force: bool = False,
    workers: int = 8,
) -> RepoResult:
    """Process a single repo through specified stages.

    Args:
        repo_name: Name of the repo directory.
        stages: List of stages to run (subset of STAGES).
        force: If True, re-run even if output exists.
        workers: Number of workers for parse/generate stages.

    Returns:
        RepoResult with details of what happened.
    """
    result = RepoResult(repo_name=repo_name)
    start_time = time.time()

    for stage in stages:
        # Check skip conditions
        if not force and is_stage_done(repo_name, stage):
            result.stages_skipped.append(stage)
            logger.info(f"  [{repo_name}] Stage '{stage}' - SKIPPED (output exists)")
            continue

        # Check prerequisites
        if not can_run_stage(repo_name, stage):
            msg = f"Prerequisites not met for stage '{stage}'"
            result.stages_failed[stage] = msg
            logger.warning(f"  [{repo_name}] Stage '{stage}' - FAILED: {msg}")
            # If a prerequisite fails, skip downstream stages too
            break

        logger.info(f"  [{repo_name}] Stage '{stage}' - RUNNING...")
        stage_start = time.time()

        try:
            if stage == "filter":
                proc = run_stage_filter(repo_name)
            elif stage == "parse":
                proc = run_stage_parse(repo_name, workers=workers)
            elif stage == "generate":
                proc = run_stage_generate(repo_name, workers=workers)
            else:
                result.stages_failed[stage] = f"Unknown stage: {stage}"
                break

            stage_elapsed = time.time() - stage_start

            if proc.returncode != 0:
                error_msg = proc.stderr[-2000:] if proc.stderr else "No stderr output"
                result.stages_failed[stage] = error_msg
                logger.error(
                    f"  [{repo_name}] Stage '{stage}' - FAILED "
                    f"(exit code {proc.returncode}, {stage_elapsed:.1f}s)"
                )
                logger.debug(f"  [{repo_name}] stderr: {error_msg}")
                # Don't continue to later stages if this one failed
                break
            else:
                result.stages_run.append(stage)
                logger.info(
                    f"  [{repo_name}] Stage '{stage}' - DONE ({stage_elapsed:.1f}s)"
                )

        except subprocess.TimeoutExpired:
            result.stages_failed[stage] = "Timeout expired"
            logger.error(f"  [{repo_name}] Stage '{stage}' - TIMEOUT")
            break
        except Exception as e:
            result.stages_failed[stage] = str(e)
            logger.error(f"  [{repo_name}] Stage '{stage}' - ERROR: {e}")
            break

    result.duration_secs = time.time() - start_time
    return result


# ---------------------------------------------------------------------------
# Repo discovery
# ---------------------------------------------------------------------------

def discover_repos(specific_repos: Optional[list[str]] = None) -> list[str]:
    """Find all valid repos in the repos/ directory.

    Args:
        specific_repos: If provided, only include these repo names.

    Returns:
        Sorted list of repo directory names.
    """
    if not REPOS_DIR.exists():
        logger.error(f"Repos directory does not exist: {REPOS_DIR}")
        return []

    all_repos = []
    for entry in sorted(REPOS_DIR.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in SKIP_REPOS:
            continue
        # Must be a git repo
        if not (entry / ".git").exists():
            logger.debug(f"Skipping {entry.name}: not a git repo")
            continue
        all_repos.append(entry.name)

    if specific_repos:
        # Filter to only requested repos
        requested = set(specific_repos)
        found = [r for r in all_repos if r in requested]
        not_found = requested - set(found)
        if not_found:
            logger.warning(f"Requested repos not found: {not_found}")
        return found

    return all_repos


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_batch(
    repos: list[str],
    stages: list[str],
    parallel: int = 3,
    force: bool = False,
    workers: int = 8,
) -> list[RepoResult]:
    """Run the pipeline for multiple repos with parallelism.

    Args:
        repos: List of repo names to process.
        stages: Stages to run per repo.
        parallel: Max concurrent repos.
        force: Re-run even if output exists.
        workers: Workers per stage (parse/generate).

    Returns:
        List of RepoResult objects.
    """
    logger.info("=" * 70)
    logger.info(f"BATCH PROCESSING: {len(repos)} repos, stages={stages}, "
                f"parallel={parallel}, force={force}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 70)

    results: list[RepoResult] = []
    total = len(repos)
    completed = 0
    failed_repos = []
    skipped_repos = []

    batch_start = time.time()

    if parallel <= 1:
        # Sequential processing
        for i, repo_name in enumerate(repos, 1):
            logger.info(f"\n[{i}/{total}] Processing: {repo_name}")
            result = process_repo(repo_name, stages, force=force, workers=workers)
            results.append(result)
            completed += 1
            if not result.success:
                failed_repos.append(repo_name)
            elif not result.stages_run and result.stages_skipped:
                skipped_repos.append(repo_name)
            _print_progress(completed, total, failed_repos)
    else:
        # Parallel processing using ProcessPoolExecutor
        # Note: we use ProcessPoolExecutor because each "task" spawns subprocesses
        # and we want true parallelism
        futures = {}
        with ProcessPoolExecutor(max_workers=parallel) as executor:
            for repo_name in repos:
                future = executor.submit(
                    process_repo, repo_name, stages, force, workers
                )
                futures[future] = repo_name

            for future in as_completed(futures):
                repo_name = futures[future]
                completed += 1
                try:
                    result = future.result()
                    results.append(result)
                    if not result.success:
                        failed_repos.append(repo_name)
                    elif not result.stages_run and result.stages_skipped:
                        skipped_repos.append(repo_name)
                except Exception as e:
                    logger.error(f"[{repo_name}] Unexpected error: {e}")
                    result = RepoResult(
                        repo_name=repo_name,
                        stages_failed={"unknown": str(e)},
                    )
                    results.append(result)
                    failed_repos.append(repo_name)

                _print_progress(completed, total, failed_repos)

    batch_elapsed = time.time() - batch_start

    # Final summary
    _print_summary(results, batch_elapsed, stages)

    return results


def _print_progress(completed: int, total: int, failed: list[str]):
    """Print a progress line."""
    pct = (completed / total) * 100 if total > 0 else 0
    fail_str = f", {len(failed)} failed" if failed else ""
    logger.info(f"  Progress: {completed}/{total} ({pct:.0f}%){fail_str}")


def _print_summary(results: list[RepoResult], elapsed: float, stages: list[str]):
    """Print final batch summary."""
    logger.info("\n" + "=" * 70)
    logger.info("BATCH SUMMARY")
    logger.info("=" * 70)

    succeeded = [r for r in results if r.success and r.stages_run]
    skipped = [r for r in results if r.success and not r.stages_run]
    failed = [r for r in results if not r.success]

    logger.info(f"  Total repos:    {len(results)}")
    logger.info(f"  Succeeded:      {len(succeeded)}")
    logger.info(f"  Skipped (done): {len(skipped)}")
    logger.info(f"  Failed:         {len(failed)}")
    logger.info(f"  Total time:     {timedelta(seconds=int(elapsed))}")
    logger.info(f"  Stages run:     {', '.join(stages)}")

    if succeeded:
        logger.info("\n  Completed repos:")
        for r in sorted(succeeded, key=lambda x: x.repo_name):
            stages_str = ", ".join(r.stages_run)
            logger.info(f"    {r.repo_name} [{stages_str}] ({r.duration_secs:.1f}s)")

    if skipped:
        logger.info("\n  Skipped repos (all stages already done):")
        for r in sorted(skipped, key=lambda x: x.repo_name):
            logger.info(f"    {r.repo_name}")

    if failed:
        logger.info("\n  Failed repos:")
        for r in sorted(failed, key=lambda x: x.repo_name):
            for stage, error in r.stages_failed.items():
                # Truncate error for display
                short_err = error[:200].replace("\n", " ")
                logger.info(f"    {r.repo_name} [stage={stage}]: {short_err}")

    logger.info("=" * 70)

    # Write summary JSON
    summary_path = LOG_DIR / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        "stages": stages,
        "total": len(results),
        "succeeded": len(succeeded),
        "skipped": len(skipped),
        "failed": len(failed),
        "results": [
            {
                "repo": r.repo_name,
                "success": r.success,
                "stages_run": r.stages_run,
                "stages_skipped": r.stages_skipped,
                "stages_failed": r.stages_failed,
                "duration_secs": r.duration_secs,
            }
            for r in results
        ],
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    logger.info(f"\n  Summary written to: {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-process repos through the 3-stage pipeline "
                    "(parse -> filter -> generate).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--parallel", "-p",
        type=int,
        default=3,
        help="Number of repos to process concurrently (default: 3).",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-process repos even if output files already exist.",
    )
    parser.add_argument(
        "--repos", "-r",
        type=str,
        default=None,
        help="Comma-separated list of specific repo names to process. "
             "If not set, processes all discovered repos.",
    )
    parser.add_argument(
        "--stage", "-s",
        type=str,
        default=None,
        help="Comma-separated list of stages to run (parse,filter,generate). "
             "Default: all stages. Example: --stage generate",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=8,
        help="Workers for parse/generate stages (default: 8).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine stages to run
    if args.stage:
        stages = [s.strip() for s in args.stage.split(",")]
        invalid = [s for s in stages if s not in STAGES]
        if invalid:
            logger.error(f"Invalid stages: {invalid}. Valid: {list(STAGES)}")
            sys.exit(1)
    else:
        stages = list(STAGES)

    # Discover repos
    specific_repos = None
    if args.repos:
        specific_repos = [r.strip() for r in args.repos.split(",")]

    repos = discover_repos(specific_repos)

    if not repos:
        logger.error("No repos found to process.")
        sys.exit(1)

    logger.info(f"Discovered {len(repos)} repos: {', '.join(repos)}")

    # Ensure output directories exist
    FILTERED_DIR.mkdir(parents=True, exist_ok=True)
    TASKS_DIR.mkdir(parents=True, exist_ok=True)

    # Dry run mode
    if args.dry_run:
        logger.info("\n[DRY RUN] Would process the following:")
        for repo_name in repos:
            for stage in stages:
                done = is_stage_done(repo_name, stage)
                can_run = can_run_stage(repo_name, stage)
                status = "SKIP (done)" if (done and not args.force) else (
                    "RUN" if can_run else "BLOCKED (prerequisites)"
                )
                logger.info(f"  {repo_name} / {stage}: {status}")
        sys.exit(0)

    # Run the batch
    results = run_batch(
        repos=repos,
        stages=stages,
        parallel=args.parallel,
        force=args.force,
        workers=args.workers,
    )

    # Exit code: non-zero if any repo failed
    failed_count = sum(1 for r in results if not r.success)
    if failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
