#!/usr/bin/env python3
"""Generate multi-step resolution training tasks from commit chains.

Reads commit chain JSONL (output of detect_commit_chains.py) and generates
training tasks that require iterative reasoning over multiple commits.

Task types:
  - iterative_fix: Given initial fix, predict the follow-up fix
  - revert_recovery: Given reverted commit + reason, predict proper fix
  - feature_step: Given feature progress so far, predict next step

Usage:
    python generate_multistep_tasks.py \
        --chains data/commit_chains_django.jsonl \
        --repo-path repos/django_django \
        --output-dir data/multistep_tasks/ \
        --workers 8
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Maximum diff size (chars) before we skip a chain
MAX_DIFF_CHARS = 10000


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def git_run(repo_path: str, args: list[str], timeout: int = 30) -> Optional[str]:
    """Run a git command and return stdout, or None on failure."""
    cmd = ["git", "-C", repo_path] + args
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return None
        return result.stdout
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
        return None


def get_diff_for_commit(repo_path: str, commit_hash: str) -> Optional[str]:
    """Get the unified diff for a specific commit (against its parent).

    Returns the full diff text or None if unavailable.
    """
    output = git_run(
        repo_path,
        ["diff", f"{commit_hash}~1", commit_hash, "--", "*.py"],
        timeout=60,
    )
    if output is None:
        # Fallback: try show with diff format
        output = git_run(
            repo_path,
            ["show", "--format=", "--patch", commit_hash, "--", "*.py"],
            timeout=60,
        )
    return output


def get_file_content_at_commit(
    repo_path: str, commit_hash: str, filepath: str
) -> Optional[str]:
    """Get file content at a specific commit."""
    return git_run(repo_path, ["show", f"{commit_hash}:{filepath}"])


def get_accumulated_diff(
    repo_path: str, base_commit: str, target_commit: str
) -> Optional[str]:
    """Get the accumulated diff between two commits (for Python files only)."""
    return git_run(
        repo_path,
        ["diff", base_commit, target_commit, "--", "*.py"],
        timeout=60,
    )


# ---------------------------------------------------------------------------
# Diff / text helpers
# ---------------------------------------------------------------------------


def diff_to_search_replace(diff_text: str) -> str:
    """Convert a unified diff into search/replace blocks.

    Each hunk becomes:
    <<<<<<< SEARCH
    old code here
    =======
    new code here
    >>>>>>> REPLACE

    Context lines (no +/-) are included in both SEARCH and REPLACE sections
    to provide anchoring.
    """
    blocks: list[str] = []
    lines = diff_text.split("\n")
    i = 0

    while i < len(lines):
        # Skip until we find a hunk header
        if not lines[i].startswith("@@"):
            i += 1
            continue

        i += 1  # skip the @@ line

        search_lines: list[str] = []
        replace_lines: list[str] = []

        # Process lines in this hunk until next hunk header or end
        while i < len(lines) and not lines[i].startswith("@@"):
            line = lines[i]
            if line.startswith("diff --git") or line.startswith("---") or line.startswith("+++") or line.startswith("index "):
                break

            if line.startswith("-"):
                search_lines.append(line[1:])  # Remove the - prefix
            elif line.startswith("+"):
                replace_lines.append(line[1:])  # Remove the + prefix
            elif line.startswith(" ") or line == "":
                # Context line - include in both
                content = line[1:] if line.startswith(" ") else ""
                search_lines.append(content)
                replace_lines.append(content)
            else:
                # No-newline marker or other; skip
                pass
            i += 1

        if search_lines or replace_lines:
            block = "<<<<<<< SEARCH\n"
            block += "\n".join(search_lines) + "\n"
            block += "=======\n"
            block += "\n".join(replace_lines) + "\n"
            block += ">>>>>>> REPLACE"
            blocks.append(block)

    return "\n\n".join(blocks)


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
    msg = re.sub(r"\(?(?:Refs|Fixed|Closes|Related to)\s+#\d+\)?", "", msg, flags=re.IGNORECASE)
    # Remove CVE references prefix like "Fixed CVE-2026-6907 -- "
    msg = re.sub(r"^Fixed\s+CVE-[\d-]+\s*--\s*", "", msg, flags=re.IGNORECASE)
    return msg.strip()


def split_diff_by_file(diff_text: str) -> dict[str, str]:
    """Split a unified diff into per-file segments.

    Returns dict mapping filepath -> diff segment for that file.
    """
    file_diffs: dict[str, str] = {}
    segments = re.split(r"^(diff --git .+)$", diff_text, flags=re.MULTILINE)
    i = 1
    while i < len(segments):
        header = segments[i]
        content = segments[i + 1] if i + 1 < len(segments) else ""
        match = re.match(r"diff --git a/(.+?) b/(.+)", header)
        if match:
            fpath = match.group(2)
            file_diffs[fpath] = header + content
        i += 2
    return file_diffs


# ---------------------------------------------------------------------------
# Task generators per chain type
# ---------------------------------------------------------------------------


def generate_iterative_fix_task(
    chain: dict[str, Any], repo_path: str
) -> Optional[dict[str, Any]]:
    """Generate an iterative_fix task from a follow_up chain.

    Input: issue description (from first commit msg) + initial fix (first commit's diff)
    Output: follow-up fix (second commit's diff in SEARCH/REPLACE) with context
    """
    commits = chain.get("commits", [])
    if len(commits) < 2:
        return None

    first_commit = commits[0]
    second_commit = commits[1]

    # Get diffs
    first_diff = get_diff_for_commit(repo_path, first_commit["hash"])
    second_diff = get_diff_for_commit(repo_path, second_commit["hash"])

    if not first_diff or not second_diff:
        return None

    # Quality check: skip if any diff is too large or empty
    if len(first_diff.strip()) == 0 or len(second_diff.strip()) == 0:
        return None
    if len(first_diff) > MAX_DIFF_CHARS or len(second_diff) > MAX_DIFF_CHARS:
        return None

    # Build input
    issue_desc = clean_commit_message(first_commit["message"])
    initial_fix_sr = diff_to_search_replace(first_diff)

    if not initial_fix_sr.strip():
        return None

    # Build context about what the follow-up addresses
    followup_desc = clean_commit_message(second_commit["message"])

    input_text = (
        f"Issue: {issue_desc}\n\n"
        f"An initial fix was applied:\n{initial_fix_sr}\n\n"
        f"However, the fix was incomplete. "
        f"The follow-up addresses: {followup_desc}\n\n"
        f"Provide the additional changes needed."
    )

    # Build output
    output_text = diff_to_search_replace(second_diff)
    if not output_text.strip():
        return None

    return {
        "task_type": "iterative_fix",
        "repo": os.path.basename(repo_path),
        "commits": [first_commit["hash"], second_commit["hash"]],
        "input": input_text,
        "output": output_text,
        "metadata": {
            "chain_type": "follow_up",
            "shared_ticket": chain.get("shared_ticket", ""),
            "shared_files": chain.get("shared_files", []),
            "first_commit_msg": first_commit["message"],
            "second_commit_msg": second_commit["message"],
            "num_commits_in_chain": len(commits),
        },
    }


def generate_revert_recovery_task(
    chain: dict[str, Any], repo_path: str
) -> Optional[dict[str, Any]]:
    """Generate a revert_recovery task from a revert_fix chain.

    Input: original commit msg + revert reason + original code context
    Output: proper fix (third commit's diff in SEARCH/REPLACE)

    Chain structure: [original_commit, revert_commit, proper_fix_commit]
    (original may be missing, in which case chain has 2 commits)
    """
    commits = chain.get("commits", [])

    # We need at least a revert and a fix
    if len(commits) < 2:
        return None

    # Identify commits based on chain structure
    if len(commits) >= 3:
        original_commit = commits[0]
        revert_commit = commits[1]
        fix_commit = commits[2]
    else:
        # Only revert + fix available (no original found)
        original_commit = None
        revert_commit = commits[0]
        fix_commit = commits[1]

    # Get the proper fix diff
    fix_diff = get_diff_for_commit(repo_path, fix_commit["hash"])
    if not fix_diff or len(fix_diff.strip()) == 0:
        return None
    if len(fix_diff) > MAX_DIFF_CHARS:
        return None

    # Build input
    parts: list[str] = []

    if original_commit:
        original_desc = clean_commit_message(original_commit["message"])
        parts.append(f"Original change: {original_desc}")

        # Get original diff for context
        original_diff = get_diff_for_commit(repo_path, original_commit["hash"])
        if original_diff and len(original_diff) <= MAX_DIFF_CHARS:
            original_sr = diff_to_search_replace(original_diff)
            if original_sr.strip():
                parts.append(f"\nOriginal patch:\n{original_sr}")

    # Revert reason
    revert_msg = revert_commit["message"]
    revert_desc = clean_commit_message(revert_msg)
    # Keep "Revert" keyword since it's informative
    if "revert" in revert_msg.lower():
        parts.append(f"\nThe original change was reverted: {revert_msg}")
    else:
        parts.append(f"\nRevert reason: {revert_desc}")

    # Code context: show files at the state after revert (before the proper fix)
    shared_files = chain.get("shared_files", [])
    if shared_files:
        # Show content of key files after the revert
        context_parts: list[str] = []
        for fpath in shared_files[:3]:  # Limit to 3 files
            content = get_file_content_at_commit(
                repo_path, f"{fix_commit['hash']}~1", fpath
            )
            if content:
                lines = content.split("\n")
                if len(lines) > 200:
                    # Truncate large files
                    content = "\n".join(lines[:200]) + f"\n... ({len(lines) - 200} more lines)"
                context_parts.append(f"File: {fpath}\n```python\n{content}\n```")
        if context_parts:
            parts.append("\nCurrent code (after revert):\n" + "\n\n".join(context_parts))

    parts.append("\nProvide the proper fix that addresses the original issue without the problems that caused the revert.")

    input_text = "\n".join(parts)

    # Build output
    output_text = diff_to_search_replace(fix_diff)
    if not output_text.strip():
        return None

    commit_list = [c["hash"] for c in commits]

    return {
        "task_type": "revert_recovery",
        "repo": os.path.basename(repo_path),
        "commits": commit_list,
        "input": input_text,
        "output": output_text,
        "metadata": {
            "chain_type": "revert_fix",
            "shared_ticket": chain.get("shared_ticket", ""),
            "shared_files": shared_files,
            "revert_commit_msg": revert_commit["message"],
            "fix_commit_msg": fix_commit["message"],
            "has_original": original_commit is not None,
        },
    }


def generate_feature_step_tasks(
    chain: dict[str, Any], repo_path: str
) -> list[dict[str, Any]]:
    """Generate feature_step tasks from a feature chain (3+ commits).

    For a chain of N commits, generates N-1 tasks:
      - Task 1: Given description + commit 1 state → predict commit 2 changes
      - Task 2: Given description + commits 1-2 state → predict commit 3 changes
      - ...

    Input: feature description + accumulated code so far
    Output: next step's changes in SEARCH/REPLACE
    """
    commits = chain.get("commits", [])
    if len(commits) < 3:
        return []

    # Build feature description from aggregated commit messages
    all_messages = [clean_commit_message(c["message"]) for c in commits]
    # Deduplicate while preserving order
    seen = set()
    unique_messages = []
    for msg in all_messages:
        if msg and msg not in seen:
            seen.add(msg)
            unique_messages.append(msg)

    feature_desc = "Feature development steps:\n" + "\n".join(
        f"  {i+1}. {msg}" for i, msg in enumerate(unique_messages)
    )

    ticket = chain.get("shared_ticket", "")
    if ticket:
        feature_desc = f"Ticket #{ticket}\n{feature_desc}"

    tasks: list[dict[str, Any]] = []

    # Generate one task per step (step N predicts commit N+1)
    for step_idx in range(1, len(commits)):
        current_commit = commits[step_idx]

        # Get the diff for this step
        step_diff = get_diff_for_commit(repo_path, current_commit["hash"])
        if not step_diff or len(step_diff.strip()) == 0:
            continue
        if len(step_diff) > MAX_DIFF_CHARS:
            continue

        # Build input: feature description + what has been done so far
        input_parts: list[str] = [feature_desc, ""]

        # Show what has been completed so far (prior steps)
        input_parts.append("Completed steps:")
        for prev_idx in range(step_idx):
            prev_commit = commits[prev_idx]
            prev_desc = clean_commit_message(prev_commit["message"])
            input_parts.append(f"  Step {prev_idx + 1}: {prev_desc}")

        input_parts.append("")

        # Show accumulated code state: diff from before first commit to before current commit
        # This shows what the code looks like after all prior steps
        if step_idx == 1:
            # For step 1, show the first commit's changes as context
            prev_diff = get_diff_for_commit(repo_path, commits[0]["hash"])
            if prev_diff and len(prev_diff) <= MAX_DIFF_CHARS:
                prev_sr = diff_to_search_replace(prev_diff)
                if prev_sr.strip():
                    input_parts.append("Changes from previous step:")
                    input_parts.append(prev_sr)
                    input_parts.append("")
        else:
            # For later steps, show accumulated diff from initial state
            acc_diff = get_accumulated_diff(
                repo_path,
                f"{commits[0]['hash']}~1",
                f"{current_commit['hash']}~1",
            )
            if acc_diff and len(acc_diff) <= MAX_DIFF_CHARS:
                acc_sr = diff_to_search_replace(acc_diff)
                if acc_sr.strip():
                    input_parts.append("Accumulated changes so far:")
                    input_parts.append(acc_sr)
                    input_parts.append("")

        # Current step description
        current_desc = clean_commit_message(current_commit["message"])
        input_parts.append(f"Now implement step {step_idx + 1}: {current_desc}")

        input_text = "\n".join(input_parts)

        # Build output
        output_text = diff_to_search_replace(step_diff)
        if not output_text.strip():
            continue

        tasks.append({
            "task_type": "feature_step",
            "repo": os.path.basename(repo_path),
            "commit": current_commit["hash"],
            "commits": [c["hash"] for c in commits],
            "input": input_text,
            "output": output_text,
            "metadata": {
                "chain_type": "feature",
                "shared_ticket": ticket,
                "shared_files": chain.get("shared_files", []),
                "step_number": step_idx + 1,
                "total_steps": len(commits),
                "step_commit_msg": current_commit["message"],
            },
        })

    return tasks


# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------


def process_chain(
    chain_json: str, repo_path: str
) -> list[dict[str, Any]]:
    """Process a single chain and generate tasks.

    This function runs in a worker process.
    Returns a list of generated task dicts.
    """
    try:
        chain = json.loads(chain_json)
    except json.JSONDecodeError:
        return []

    chain_type = chain.get("chain_type", "")
    results: list[dict[str, Any]] = []

    try:
        if chain_type == "follow_up":
            task = generate_iterative_fix_task(chain, repo_path)
            if task:
                results.append(task)

        elif chain_type == "revert_fix":
            task = generate_revert_recovery_task(chain, repo_path)
            if task:
                results.append(task)

        elif chain_type == "feature":
            tasks = generate_feature_step_tasks(chain, repo_path)
            results.extend(tasks)

    except Exception as e:
        logger.debug(f"Error processing chain: {e}")

    return results


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def load_chains(chains_path: str) -> list[str]:
    """Load chain JSON lines from input file."""
    lines: list[str] = []
    with open(chains_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines.append(line)
    return lines


def run(
    chains_path: str,
    repo_path: str,
    output_dir: str,
    workers: int,
) -> None:
    """Main execution: load chains, process in parallel, write outputs."""
    # Validate inputs
    if not os.path.exists(chains_path):
        logger.error(f"Chains file not found: {chains_path}")
        sys.exit(1)

    repo_path = os.path.abspath(repo_path)
    if not os.path.isdir(os.path.join(repo_path, ".git")):
        logger.error(f"Not a git repo: {repo_path}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Load chains
    logger.info(f"Loading chains from {chains_path}")
    chain_lines = load_chains(chains_path)
    logger.info(f"Loaded {len(chain_lines)} chains")

    if not chain_lines:
        logger.warning("No chains to process")
        return

    # Count chain types
    type_counts: dict[str, int] = {}
    for line in chain_lines:
        try:
            ct = json.loads(line).get("chain_type", "unknown")
            type_counts[ct] = type_counts.get(ct, 0) + 1
        except json.JSONDecodeError:
            pass
    logger.info(f"Chain type distribution: {type_counts}")

    # Output file paths (one combined file + per-type files)
    combined_path = os.path.join(output_dir, "multistep_tasks_all.jsonl")
    type_paths = {
        "iterative_fix": os.path.join(output_dir, "iterative_fix.jsonl"),
        "revert_recovery": os.path.join(output_dir, "revert_recovery.jsonl"),
        "feature_step": os.path.join(output_dir, "feature_step.jsonl"),
    }

    # Counters
    counters: dict[str, int] = {
        "iterative_fix": 0,
        "revert_recovery": 0,
        "feature_step": 0,
    }
    total_generated = 0
    errors = 0

    # Process chains in parallel
    logger.info(f"Processing with {workers} workers...")

    # Open output files
    writers: dict[str, Any] = {}
    writers["all"] = open(combined_path, "w", encoding="utf-8")
    for task_type, path in type_paths.items():
        writers[task_type] = open(path, "w", encoding="utf-8")

    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_chain, chain_line, repo_path): i
                for i, chain_line in enumerate(chain_lines)
            }

            with tqdm(total=len(futures), desc="Processing chains") as pbar:
                for future in as_completed(futures):
                    try:
                        results = future.result(timeout=120)
                        for task in results:
                            task_type = task.get("task_type", "unknown")
                            task_json = json.dumps(task, ensure_ascii=False) + "\n"

                            # Write to combined file
                            writers["all"].write(task_json)

                            # Write to type-specific file
                            if task_type in writers:
                                writers[task_type].write(task_json)

                            counters[task_type] = counters.get(task_type, 0) + 1
                            total_generated += 1
                    except Exception as e:
                        errors += 1
                        logger.debug(f"Worker error: {e}")
                    pbar.update(1)
    finally:
        for w in writers.values():
            w.close()

    # Summary
    logger.info("=" * 60)
    logger.info("Generation complete!")
    logger.info(f"  Chains processed: {len(chain_lines)}")
    logger.info(f"  Total tasks generated: {total_generated}")
    logger.info(f"  Errors/skipped: {errors}")
    logger.info("")
    for task_type, count in sorted(counters.items()):
        path = type_paths.get(task_type, "")
        logger.info(f"  {task_type}: {count} tasks -> {path}")
    logger.info(f"  Combined: {total_generated} tasks -> {combined_path}")
    logger.info("=" * 60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate multi-step resolution training tasks from commit chains.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--chains",
        required=True,
        help="Path to commit chains JSONL file (output of detect_commit_chains.py)",
    )
    parser.add_argument(
        "--repo-path",
        required=True,
        help="Path to the git repository",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for output JSONL files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        chains_path=args.chains,
        repo_path=args.repo_path,
        output_dir=args.output_dir,
        workers=args.workers,
    )
