#!/usr/bin/env python3
"""Parse commits from a git repository into structured JSON data.

Extracts commit metadata, diffs, and file classifications for downstream
filtering and data synthesis pipelines.

Usage:
    python parse_commits.py /path/to/repo --output-dir ./parsed/
    python parse_commits.py /path/to/repo --output-file commits.jsonl --sample 100
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Optional

try:
    from tqdm import tqdm
except ImportError:
    # Fallback: no progress bar
    def tqdm(iterable, **kwargs):
        return iterable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Delimiter unlikely to appear in commit messages
COMMIT_DELIMITER = "---COMMIT_BOUNDARY_7f3a9c2e---"
COMMIT_FORMAT = f"%H|%P|%aI%n%B%n{COMMIT_DELIMITER}"


def is_test_file(filepath: str) -> bool:
    """Heuristic to classify a file as a test file.

    Checks for common test patterns in the file path.
    """
    path_lower = filepath.lower()
    # Common test patterns
    patterns = [
        "/test_",
        "/tests/",
        "/test/",
        "/testing/",
        "_test.py",
        "_tests.py",
        "tests.py",
        "/conftest.py",
        "/testcase",
        "/spec/",
        "_spec.",
    ]
    return any(p in path_lower for p in patterns)


def run_git(args: list[str], cwd: str, timeout: int = 60) -> Optional[str]:
    """Run a git command and return stdout, or None on failure."""
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return None
        return result.stdout
    except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
        logger.debug(f"Git command failed: git {' '.join(args[:3])}... -> {e}")
        return None


def get_commit_hashes(repo_dir: str, sample: Optional[int] = None) -> list[str]:
    """Get list of commit hashes from the repository.

    Args:
        repo_dir: Path to the git repository.
        sample: If set, return only the most recent N commits.

    Returns:
        List of commit hash strings.
    """
    args = ["log", "--format=%H", "--no-merges"]
    if sample:
        args.extend(["-n", str(sample)])

    output = run_git(args, cwd=repo_dir, timeout=120)
    if output is None:
        logger.error(f"Failed to get commit hashes from {repo_dir}")
        return []

    hashes = [h.strip() for h in output.strip().split("\n") if h.strip()]
    return hashes


def parse_diff_stats(diff_text: str) -> tuple[int, int]:
    """Count total insertions and deletions from a unified diff.

    Args:
        diff_text: Unified diff string.

    Returns:
        Tuple of (insertions, deletions).
    """
    insertions = 0
    deletions = 0
    for line in diff_text.split("\n"):
        if line.startswith("+") and not line.startswith("+++"):
            insertions += 1
        elif line.startswith("-") and not line.startswith("---"):
            deletions += 1
    return insertions, deletions


def split_diff_by_file(diff_text: str) -> list[dict[str, str]]:
    """Split a combined diff into per-file segments.

    Args:
        diff_text: Full diff output from git diff.

    Returns:
        List of dicts with 'filepath' and 'patch' keys.
    """
    files = []
    # Split on 'diff --git' boundaries
    segments = re.split(r"^(diff --git .+)$", diff_text, flags=re.MULTILINE)

    # segments[0] is empty or preamble, then alternating: header, content
    i = 1
    while i < len(segments):
        header = segments[i]
        content = segments[i + 1] if i + 1 < len(segments) else ""

        # Extract file path from 'diff --git a/path b/path'
        match = re.match(r"diff --git a/(.+?) b/(.+)", header)
        if match:
            filepath = match.group(2)  # Use the 'b' path (new name)
            full_patch = header + content
            files.append({"filepath": filepath, "patch": full_patch})

        i += 2

    return files


def parse_single_commit(args: tuple[str, str]) -> Optional[dict[str, Any]]:
    """Parse a single commit into structured data.

    Args:
        args: Tuple of (commit_hash, repo_dir).

    Returns:
        Dict with commit data, or None if parsing fails.
    """
    commit_hash, repo_dir = args

    try:
        # Get commit metadata
        meta_output = run_git(
            ["log", "-1", "--format=%H|%P|%aI%n%B", commit_hash],
            cwd=repo_dir,
            timeout=30,
        )
        if meta_output is None:
            return None

        lines = meta_output.strip().split("\n")
        if not lines:
            return None

        # Parse first line: hash|parent|date
        meta_parts = lines[0].split("|", 2)
        if len(meta_parts) < 3:
            return None

        commit_hash_parsed = meta_parts[0].strip()
        parent_hash = meta_parts[1].strip().split()[0] if meta_parts[1].strip() else ""
        commit_date = meta_parts[2].strip()

        # Commit message is the rest
        commit_message = "\n".join(lines[1:]).strip()

        # Get the diff for this commit
        if parent_hash:
            diff_output = run_git(
                ["diff", parent_hash, commit_hash, "--", ".", ":(exclude)*.lock"],
                cwd=repo_dir,
                timeout=60,
            )
        else:
            # Root commit: diff against empty tree
            diff_output = run_git(
                ["diff", "--root", commit_hash, "--", ".", ":(exclude)*.lock"],
                cwd=repo_dir,
                timeout=60,
            )
            if diff_output is None:
                # Try alternative for root commit
                diff_output = run_git(
                    [
                        "diff",
                        "4b825dc642cb6eb9a060e54bf899d69f82cf0262",
                        commit_hash,
                        "--",
                        ".",
                    ],
                    cwd=repo_dir,
                    timeout=60,
                )

        if diff_output is None:
            diff_output = ""

        # Split diff by file
        file_diffs = split_diff_by_file(diff_output)

        # Classify files and build patches
        changed_files = []
        source_patches = []
        test_patches = []
        num_src_files = 0
        num_test_files = 0
        total_insertions = 0
        total_deletions = 0

        for fd in file_diffs:
            filepath = fd["filepath"]
            patch = fd["patch"]
            is_test = is_test_file(filepath)

            ins, dels = parse_diff_stats(patch)
            total_insertions += ins
            total_deletions += dels

            file_info = {
                "filepath": filepath,
                "is_test": is_test,
                "insertions": ins,
                "deletions": dels,
            }
            changed_files.append(file_info)

            if is_test:
                num_test_files += 1
                test_patches.append(patch)
            else:
                num_src_files += 1
                source_patches.append(patch)

        source_patch = "\n".join(source_patches)
        test_patch = "\n".join(test_patches)

        return {
            "commit_hash": commit_hash_parsed,
            "parent_hash": parent_hash,
            "commit_message": commit_message,
            "commit_date": commit_date,
            "changed_files": changed_files,
            "source_patch": source_patch,
            "test_patch": test_patch,
            "num_src_files": num_src_files,
            "num_test_files": num_test_files,
            "total_insertions": total_insertions,
            "total_deletions": total_deletions,
        }

    except Exception as e:
        logger.debug(f"Error parsing commit {commit_hash}: {e}")
        return None


def write_jsonl(data: list[dict], output_path: str) -> None:
    """Write list of dicts to a JSONL file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_individual_json(data: list[dict], output_dir: str) -> None:
    """Write each commit as an individual JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    for item in data:
        filepath = os.path.join(output_dir, f"{item['commit_hash']}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(item, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Parse git commits into structured JSON data for midtrain synthesis."
    )
    parser.add_argument(
        "repo_dir",
        type=str,
        help="Path to the git repository to parse.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output JSONL file path. If set, writes a single JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for individual JSON files (one per commit).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only process the N most recent commits (for testing).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose/debug logging.",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    repo_dir = os.path.abspath(args.repo_dir)
    if not (os.path.isdir(os.path.join(repo_dir, ".git")) or os.path.isfile(os.path.join(repo_dir, "HEAD"))):
        logger.error(f"Not a git repository: {repo_dir}")
        sys.exit(1)

    # Default output: JSONL file named after the repo
    if args.output_file is None and args.output_dir is None:
        repo_name = os.path.basename(repo_dir)
        args.output_file = f"{repo_name}_commits.jsonl"

    # Get commit hashes
    logger.info(f"Fetching commit hashes from {repo_dir}...")
    hashes = get_commit_hashes(repo_dir, sample=args.sample)
    logger.info(f"Found {len(hashes)} commits to process.")

    if not hashes:
        logger.warning("No commits found. Exiting.")
        sys.exit(0)

    # Prepare work items
    work_items = [(h, repo_dir) for h in hashes]

    # Process commits in parallel
    logger.info(f"Parsing commits with {args.workers} workers...")
    results = []

    if args.workers == 1:
        # Single-process mode for easier debugging
        for item in tqdm(work_items, desc="Parsing commits"):
            result = parse_single_commit(item)
            if result is not None:
                results.append(result)
    else:
        with Pool(processes=args.workers) as pool:
            for result in tqdm(
                pool.imap_unordered(parse_single_commit, work_items, chunksize=50),
                total=len(work_items),
                desc="Parsing commits",
            ):
                if result is not None:
                    results.append(result)

    logger.info(
        f"Successfully parsed {len(results)}/{len(hashes)} commits "
        f"({len(hashes) - len(results)} failed/skipped)."
    )

    # Sort by date (most recent first)
    results.sort(key=lambda x: x["commit_date"], reverse=True)

    # Write output
    if args.output_file:
        output_path = os.path.abspath(args.output_file)
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        write_jsonl(results, output_path)
        logger.info(f"Wrote {len(results)} commits to {output_path}")

    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
        write_individual_json(results, output_dir)
        logger.info(f"Wrote {len(results)} individual JSON files to {output_dir}")


if __name__ == "__main__":
    main()
