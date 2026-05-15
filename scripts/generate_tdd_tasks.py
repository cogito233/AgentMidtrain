#!/usr/bin/env python3
"""Generate TDD (Test-Driven Development) simulation tasks from filtered commits.

For commits that modify BOTH test files and source files, this script reverses
the development order to create TDD-style tasks:
  - Input: NEW test code (added lines) + failure description + source context (pre-fix)
  - Output: Source fix in SEARCH/REPLACE format

Works with ANY repository — no hardcoded commit hashes or path prefixes.

Usage:
    python generate_tdd_tasks.py \
        --input data/filtered_commits/django_django.jsonl \
        --output-dir data/tasks/ \
        --repo-path repos/django_django \
        --workers 8 \
        --sample 100
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """Runtime configuration."""

    input_path: str
    output_dir: str
    repo_path: str
    workers: int = 4
    sample: Optional[int] = None
    max_diff_chars: int = 5000
    max_context_lines: int = 150


# ---------------------------------------------------------------------------
# Test file detection
# ---------------------------------------------------------------------------

# Patterns that identify a file as a test file
TEST_PATH_PATTERNS = [
    r"(^|/)tests?/",        # contains /test/ or /tests/
    r"(^|/)test_[^/]+$",    # starts with test_ (at any directory level)
    r"(^|/)conftest\.py$",  # pytest conftest
    r"_test\.py$",          # ends with _test.py
    r"(^|/)testing/",       # /testing/ directory
]

_TEST_RE = re.compile("|".join(TEST_PATH_PATTERNS))


def is_test_file(filepath: str) -> bool:
    """Return True if the filepath looks like a test file."""
    return bool(_TEST_RE.search(filepath))


def is_source_file(filepath: str) -> bool:
    """Return True if the filepath is a non-test Python source file."""
    if not filepath.endswith(".py"):
        return False
    if is_test_file(filepath):
        return False
    # Exclude docs, migrations, config-only files
    if "/migrations/" in filepath:
        return False
    return True


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def git_run(
    repo_path: str, args: list[str], timeout: int = 30
) -> Optional[str]:
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


def get_file_at_commit(
    repo_path: str, commit_hash: str, filepath: str
) -> Optional[str]:
    """Get file content at a specific commit."""
    return git_run(repo_path, ["show", f"{commit_hash}:{filepath}"])


def get_diff_for_paths(
    repo_path: str, commit_hash: str, paths: list[str]
) -> Optional[str]:
    """Get unified diff for specific paths in a commit."""
    if not paths:
        return None
    out = git_run(
        repo_path,
        ["diff", f"{commit_hash}^..{commit_hash}", "--"] + paths,
        timeout=60,
    )
    return out.strip() if out else None


# ---------------------------------------------------------------------------
# Diff parsing / conversion helpers
# ---------------------------------------------------------------------------


def clean_commit_message(msg: str) -> str:
    """Strip ticket references, hash refs, and cleanup commit messages.

    Removes patterns like:
      - "Fixed #37095 -- "
      - "Refs #35870 -- "
      - "Closes #12345 -- "
      - "[prefix] message"
      - "Follow-up to 63c56cda..."
    """
    # Remove "Fixed/Refs/Closes #NNN(, Fixed/Refs/Closes #NNN)* -- " prefix
    msg = re.sub(
        r"^(?:(?:Fixed|Refs|Closes)\s+#\d+(?:,\s*)?)+\s*--\s*",
        "",
        msg,
        flags=re.IGNORECASE,
    )
    # Remove bracketed prefixes like "[3.2.x]"
    msg = re.sub(r"^\[[^\]]+\]\s*", "", msg)
    # Remove "Follow-up to <hash>..." references
    msg = re.sub(r"Follow-up to [0-9a-f]{6,40}\.?", "", msg, flags=re.IGNORECASE)
    # Remove standalone ticket references like "(#12345)"
    msg = re.sub(r"\(#\d+\)", "", msg)
    # Remove inline ticket references
    msg = re.sub(
        r"\(?(?:Refs|Fixed|Closes|Related to)\s+#\d+\)?", "", msg, flags=re.IGNORECASE
    )
    # Remove CVE references prefix
    msg = re.sub(r"^Fixed\s+CVE-[\d-]+\s*--\s*", "", msg, flags=re.IGNORECASE)
    return msg.strip()


def diff_to_search_replace(diff_text: str) -> str:
    """Convert a unified diff into SEARCH/REPLACE blocks.

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

    current_file = ""

    while i < len(lines):
        line = lines[i]

        # Track current file for multi-file diffs
        if line.startswith("diff --git"):
            match = re.match(r"diff --git a/.+ b/(.+)", line)
            if match:
                current_file = match.group(1)
            i += 1
            continue

        # Skip diff metadata lines
        if line.startswith("---") or line.startswith("+++") or line.startswith("index "):
            i += 1
            continue

        # Process hunk
        if not line.startswith("@@"):
            i += 1
            continue

        i += 1  # skip the @@ line

        search_lines: list[str] = []
        replace_lines: list[str] = []

        # Process lines in this hunk until next hunk/file header or end
        while i < len(lines) and not lines[i].startswith("@@"):
            hline = lines[i]
            if (
                hline.startswith("diff --git")
                or hline.startswith("---")
                or hline.startswith("+++")
                or hline.startswith("index ")
            ):
                break

            if hline.startswith("-"):
                search_lines.append(hline[1:])
            elif hline.startswith("+"):
                replace_lines.append(hline[1:])
            elif hline.startswith(" ") or hline == "":
                # Context line - include in both
                content = hline[1:] if hline.startswith(" ") else ""
                search_lines.append(content)
                replace_lines.append(content)
            else:
                # No-newline marker or other; skip
                pass
            i += 1

        if search_lines or replace_lines:
            file_header = f"# File: {current_file}\n" if current_file else ""
            block = file_header
            block += "<<<<<<< SEARCH\n"
            block += "\n".join(search_lines) + "\n"
            block += "=======\n"
            block += "\n".join(replace_lines) + "\n"
            block += ">>>>>>> REPLACE"
            blocks.append(block)

    return "\n\n".join(blocks)


def extract_added_lines(patch: str) -> str:
    """Extract only the added lines (without '+' prefix) from a unified diff.

    Skips file headers (+++) and returns clean code.
    """
    added: list[str] = []
    for line in patch.split("\n"):
        if line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
    return "\n".join(added)


def has_meaningful_additions(patch: str) -> bool:
    """Return True if the patch has non-trivial added lines."""
    added = extract_added_lines(patch)
    # Filter out blank lines and single-char lines
    meaningful = [
        l for l in added.split("\n")
        if l.strip() and len(l.strip()) > 1
    ]
    return len(meaningful) > 0


def extract_file_diff(full_diff: str, filepath: str) -> Optional[str]:
    """Extract the diff section for a specific file from a combined diff."""
    sections = re.split(r"(?=^diff --git )", full_diff, flags=re.MULTILINE)
    for section in sections:
        if filepath in section:
            return section
    return None


# ---------------------------------------------------------------------------
# Source context extraction
# ---------------------------------------------------------------------------


def extract_relevant_source_context(
    repo_path: str,
    commit_hash: str,
    src_files: list[str],
    src_patch: str,
    max_lines: int = 150,
) -> str:
    """Get the relevant source code BEFORE the fix, with context around changed hunks.

    Shows the pre-fix code (at parent commit) with line numbers, providing
    enough context to understand where and what needs to change.
    """
    contexts: list[str] = []

    for filepath in src_files:
        # Get file content at parent (pre-fix state)
        content = get_file_at_commit(repo_path, f"{commit_hash}^", filepath)
        if content is None:
            # File might be newly created in this commit — skip
            continue

        lines = content.split("\n")
        total_lines = len(lines)

        # Extract the diff section for this specific file
        file_diff = extract_file_diff(src_patch, filepath)
        if not file_diff:
            continue

        # Parse hunk headers to find changed line ranges
        hunk_ranges: list[tuple[int, int]] = []
        for match in re.finditer(
            r"@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@", file_diff
        ):
            start = int(match.group(1))
            count = int(match.group(2)) if match.group(2) else 1
            hunk_ranges.append((start, start + count - 1))

        if not hunk_ranges:
            # Fallback: show first N lines
            context_lines = lines[: min(60, total_lines)]
            contexts.append(
                f"# {filepath}\n"
                + "\n".join(f"{i+1:>4} | {l}" for i, l in enumerate(context_lines))
            )
            continue

        # Extract context around each hunk (20 lines before, 10 lines after)
        CONTEXT_BEFORE = 20
        CONTEXT_AFTER = 10
        included_ranges: list[tuple[int, int]] = []

        for start, end in hunk_ranges:
            range_start = max(0, start - 1 - CONTEXT_BEFORE)
            range_end = min(total_lines, end + CONTEXT_AFTER)
            included_ranges.append((range_start, range_end))

        # Merge overlapping ranges
        included_ranges.sort()
        merged = [included_ranges[0]]
        for s, e in included_ranges[1:]:
            if s <= merged[-1][1] + 5:  # merge if within 5 lines
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        # Build output
        file_context_lines = [f"# {filepath}"]
        for idx, (range_start, range_end) in enumerate(merged):
            if idx > 0:
                file_context_lines.append("    ...")
            for line_idx in range(range_start, min(range_end, total_lines)):
                file_context_lines.append(f"{line_idx+1:>4} | {lines[line_idx]}")

        contexts.append("\n".join(file_context_lines))

    result = "\n\n".join(contexts)

    # Truncate if too long
    result_lines = result.split("\n")
    if len(result_lines) > max_lines:
        result = "\n".join(result_lines[:max_lines]) + "\n... (truncated)"

    return result


# ---------------------------------------------------------------------------
# Test failure inference heuristics
# ---------------------------------------------------------------------------


def infer_test_failure(test_added_code: str, src_patch: str) -> str:
    """Infer what kind of test failure would occur against the old code.

    Uses heuristics based on assertion types found in the test.
    """
    failures: list[str] = []
    test_text = test_added_code

    # Heuristic 1: assertRaises / assertRaisesMessage / assertRaisesRegex
    raises_match = re.findall(
        r"(?:assertRaises|assertRaisesMessage|assertRaisesRegex|pytest\.raises)"
        r"\s*\(\s*(\w+)(?:,\s*[\"'](.+?)[\"'])?",
        test_text,
    )
    if raises_match:
        for exc_class, msg in raises_match[:2]:  # Limit to avoid verbosity
            if msg:
                failures.append(
                    f"Expected {exc_class} with message containing \"{msg}\" "
                    f"to be raised, but the current code does not raise it."
                )
            else:
                failures.append(
                    f"Expected {exc_class} to be raised, but the current code "
                    f"does not raise this exception."
                )

    # Heuristic 2: assertEqual / assertEquals / assert ... ==
    if not failures:
        equal_matches = re.findall(
            r"(?:assertEqual|assertEquals|assert_equal)\s*\(", test_text
        )
        if equal_matches:
            failures.append(
                "Equality assertion fails because the current code returns "
                "an incorrect value."
            )

    # Heuristic 3: assertIn / assertNotIn
    if not failures:
        if "assertNotIn(" in test_text or "not in" in test_text:
            failures.append(
                "Assertion fails because the current code incorrectly "
                "includes an element that should not be present."
            )
        elif "assertIn(" in test_text:
            failures.append(
                "Assertion fails because the current code does not include "
                "an expected element."
            )

    # Heuristic 4: assertTrue / assertFalse
    if not failures:
        if "assertTrue(" in test_text or "assert " in test_text:
            failures.append(
                "Boolean assertion fails because the current code returns "
                "an unexpected falsy/truthy value."
            )
        elif "assertFalse(" in test_text:
            failures.append(
                "assertFalse fails because the current code returns True "
                "where False is expected."
            )

    # Heuristic 5: assertWarns
    if not failures:
        warns_match = re.findall(
            r"(?:assertWarns|assertWarnsMessage|pytest\.warns)\s*\(\s*(\w+)",
            test_text,
        )
        if warns_match:
            for warn_class in warns_match[:2]:
                failures.append(
                    f"Expected {warn_class} warning to be emitted, "
                    f"but the current code does not emit it."
                )

    # Heuristic 6: Check for status_code / response assertions (web frameworks)
    if not failures:
        status_match = re.findall(
            r"(?:status_code|response\.status)\s*(?:==|,)\s*(\d+)", test_text
        )
        if status_match:
            failures.append(
                f"Expected HTTP status code {status_match[0]} but the current "
                f"code returns a different status."
            )

    # Fallback: generic failure from source changes
    if not failures:
        src_removed = []
        for line in src_patch.split("\n"):
            if line.startswith("-") and not line.startswith("---"):
                content = line[1:].strip()
                if content and not content.startswith("#"):
                    src_removed.append(content)

        if src_removed:
            failures.append(
                "Test expects behavior that differs from the current implementation. "
                f"The current code contains logic like: {'; '.join(src_removed[:2])}"
            )
        else:
            failures.append(
                "Test expects behavior that the current code does not implement."
            )

    return " ".join(failures)


# ---------------------------------------------------------------------------
# Test code extraction
# ---------------------------------------------------------------------------


def extract_test_method(test_patch: str) -> Optional[str]:
    """Try to extract a clean test method from the patch (added lines only).

    Looks for `def test_*` patterns in added lines and captures the full method.
    """
    lines = test_patch.split("\n")
    method_lines: list[str] = []
    capturing = False
    indent_level: Optional[int] = None

    for line in lines:
        if line.startswith("+") and not line.startswith("+++"):
            content = line[1:]  # strip '+'
            # Detect start of a new test method
            if re.match(r"\s+def test_", content) or re.match(r"def test_", content):
                if capturing and method_lines:
                    # Already captured one method, stop here (return first one)
                    break
                capturing = True
                indent_level = len(content) - len(content.lstrip())
                method_lines = [content]
                continue
            if capturing:
                stripped = content.lstrip()
                current_indent = len(content) - len(stripped)
                if content.strip() == "":
                    method_lines.append(content)
                elif current_indent <= indent_level and stripped and not stripped.startswith("#"):
                    # We've exited the method (dedented)
                    break
                else:
                    method_lines.append(content)
        elif capturing and line.startswith(" "):
            # Context line within our captured method
            content = line
            stripped = content.lstrip()
            current_indent = len(content) - len(stripped)
            if stripped and indent_level is not None and current_indent <= indent_level:
                break
            method_lines.append(content)

    if method_lines and len(method_lines) >= 3:
        return "\n".join(method_lines)
    return None


# ---------------------------------------------------------------------------
# Eligibility detection
# ---------------------------------------------------------------------------


def classify_commit_files(
    commit: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Classify changed files into test files and source files.

    Returns (test_files, src_files).
    """
    test_files: list[str] = []
    src_files: list[str] = []

    changed_files = commit.get("changed_files", [])
    for f in changed_files:
        filepath = f.get("filepath", f.get("path", ""))
        if not filepath:
            continue
        if is_test_file(filepath):
            test_files.append(filepath)
        elif is_source_file(filepath):
            src_files.append(filepath)

    return test_files, src_files


def is_tdd_eligible(
    commit: dict[str, Any],
    test_files: list[str],
    src_files: list[str],
    max_diff_chars: int,
) -> tuple[bool, str]:
    """Check if a commit is eligible for TDD simulation task generation.

    Returns (eligible, skip_reason).
    """
    if not test_files:
        return False, "no test files"
    if not src_files:
        return False, "no source files"

    # Check test_patch has meaningful additions
    test_patch = commit.get("test_patch", "")
    if not test_patch:
        return False, "empty test_patch"
    if not has_meaningful_additions(test_patch):
        return False, "test patch has no meaningful additions"

    # Check source_patch exists
    source_patch = commit.get("source_patch", commit.get("src_patch", ""))
    if not source_patch:
        return False, "empty source_patch"

    # Check total diff size
    total_size = len(test_patch) + len(source_patch)
    if total_size > max_diff_chars:
        return False, f"diff too large ({total_size} > {max_diff_chars})"

    return True, ""


# ---------------------------------------------------------------------------
# Task generation core
# ---------------------------------------------------------------------------


def generate_tdd_task(
    commit: dict[str, Any],
    repo_path: str,
    config: Config,
) -> Optional[dict[str, Any]]:
    """Generate a single TDD simulation task from a commit.

    Returns the task dict or None if generation fails.
    """
    # Classify files
    test_files, src_files = classify_commit_files(commit)

    # Eligibility check
    eligible, reason = is_tdd_eligible(
        commit, test_files, src_files, config.max_diff_chars
    )
    if not eligible:
        return None

    commit_hash = commit.get("commit_hash", "")
    test_patch = commit.get("test_patch", "")
    source_patch = commit.get("source_patch", commit.get("src_patch", ""))
    message = commit.get("commit_message", commit.get("message", ""))
    repo_name = commit.get("repo", os.path.basename(repo_path))

    # Extract test code: prefer clean method, fall back to all additions
    test_method = extract_test_method(test_patch)
    if test_method:
        test_code = test_method
    else:
        test_code = extract_added_lines(test_patch)

    if not test_code.strip():
        return None

    # Get source context (pre-fix code around changed hunks)
    source_context = extract_relevant_source_context(
        repo_path,
        commit_hash,
        src_files,
        source_patch,
        max_lines=config.max_context_lines,
    )

    if not source_context.strip():
        return None

    # Infer test failure description
    failure_desc = infer_test_failure(test_code, source_patch)

    # Convert source patch to SEARCH/REPLACE format for output
    src_search_replace = diff_to_search_replace(source_patch)
    if not src_search_replace.strip():
        # Fallback: use raw patch if conversion fails
        src_search_replace = source_patch

    # Build task input
    issue_desc = clean_commit_message(message)
    input_text = (
        f"## Failing Test\n\n"
        f"The following test should pass but currently fails:\n\n"
        f"```python\n{test_code}\n```\n\n"
        f"## Failure Description\n\n"
        f"{failure_desc}\n\n"
    )
    if issue_desc:
        input_text += f"## Issue Context\n\n{issue_desc}\n\n"
    input_text += (
        f"## Relevant Source Code (current, pre-fix)\n\n"
        f"```python\n{source_context}\n```\n\n"
        f"## Task\n\n"
        f"Fix the source code so that the failing test passes. "
        f"Provide your fix using SEARCH/REPLACE blocks."
    )

    return {
        "task_type": "tdd_simulation",
        "repo": repo_name,
        "commit": commit_hash,
        "input": input_text,
        "output": src_search_replace,
        "metadata": {
            "commit_message": message,
            "test_files": test_files,
            "src_files": src_files,
            "issue_description": issue_desc,
        },
    }


# ---------------------------------------------------------------------------
# Worker function (called in subprocess)
# ---------------------------------------------------------------------------


def process_commit(
    commit_json: str,
    repo_path: str,
    max_diff_chars: int,
    max_context_lines: int,
) -> Optional[dict[str, Any]]:
    """Process a single commit line and return a TDD task or None.

    This function runs in a worker process.
    """
    config = Config(
        input_path="",
        output_dir="",
        repo_path=repo_path,
        max_diff_chars=max_diff_chars,
        max_context_lines=max_context_lines,
    )

    try:
        commit = json.loads(commit_json)
    except json.JSONDecodeError:
        return None

    try:
        return generate_tdd_task(commit, repo_path, config)
    except Exception:
        # Don't crash the worker on individual commit failures
        return None


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def load_commits(input_path: str, sample: Optional[int] = None) -> list[str]:
    """Load commit JSON lines from input file."""
    lines: list[str] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines.append(line)
            if sample and len(lines) >= sample:
                break
    return lines


def run(config: Config) -> None:
    """Main execution: load commits, process in parallel, write output."""
    # Validate inputs
    input_path = Path(config.input_path)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    repo_path = os.path.abspath(config.repo_path)
    if not os.path.isdir(os.path.join(repo_path, ".git")):
        logger.error(f"Not a git repo: {repo_path}")
        sys.exit(1)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tdd_simulation.jsonl"

    # Load commits
    logger.info(f"Loading commits from {input_path}")
    commit_lines = load_commits(str(input_path), config.sample)
    logger.info(f"Loaded {len(commit_lines)} commits")

    if not commit_lines:
        logger.warning("No commits to process")
        return

    # Process commits in parallel
    logger.info(
        f"Processing with {config.workers} workers "
        f"(max_diff_chars={config.max_diff_chars})"
    )

    generated = 0
    skipped = 0
    errors = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            futures = {
                executor.submit(
                    process_commit,
                    commit_line,
                    repo_path,
                    config.max_diff_chars,
                    config.max_context_lines,
                ): i
                for i, commit_line in enumerate(commit_lines)
            }

            with tqdm(total=len(futures), desc="Generating TDD tasks") as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=120)
                        if result is not None:
                            out_f.write(
                                json.dumps(result, ensure_ascii=False) + "\n"
                            )
                            generated += 1
                        else:
                            skipped += 1
                    except Exception:
                        errors += 1
                    pbar.update(1)

    # Summary
    logger.info("=" * 60)
    logger.info("TDD Simulation Task Generation Complete")
    logger.info(f"  Commits processed: {len(commit_lines)}")
    logger.info(f"  Tasks generated:   {generated}")
    logger.info(f"  Skipped:           {skipped}")
    logger.info(f"  Errors:            {errors}")
    logger.info(f"  Output:            {output_path}")
    logger.info(f"  Yield rate:        {generated / max(len(commit_lines), 1) * 100:.1f}%")
    logger.info("=" * 60)


def parse_args() -> Config:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate TDD simulation tasks from filtered commits. "
            "For commits with both test and source changes, creates tasks "
            "where the input is the failing test and the output is the fix."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to filtered commits JSONL file (output of filter_commits.py)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for output JSONL file (tdd_simulation.jsonl)",
    )
    parser.add_argument(
        "--repo-path",
        required=True,
        help="Path to the git repository",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only process first N commits (for testing/debugging)",
    )
    parser.add_argument(
        "--max-diff-chars",
        type=int,
        default=5000,
        help="Skip commits where total diff exceeds this many chars (default: 5000)",
    )
    parser.add_argument(
        "--max-context-lines",
        type=int,
        default=150,
        help="Max lines of source context to include (default: 150)",
    )

    args = parser.parse_args()

    return Config(
        input_path=args.input,
        output_dir=args.output_dir,
        repo_path=args.repo_path,
        workers=args.workers,
        sample=args.sample,
        max_diff_chars=args.max_diff_chars,
        max_context_lines=args.max_context_lines,
    )


if __name__ == "__main__":
    config = parse_args()
    run(config)
