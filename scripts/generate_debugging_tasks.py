#!/usr/bin/env python3
"""Generate iterative debugging training tasks from filtered commits.

Given a bug manifestation (error/failure description) + code context, the model
must diagnose the root cause and produce a fix.

Task type: error_diagnosis
  - Examines commit's test changes to infer what error/failure was happening
  - Constructs a synthetic "error report" with test code, error type, and source context
  - Expected output: structured diagnosis (root cause, location, fix in SEARCH/REPLACE)

Usage:
    python generate_debugging_tasks.py \
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
# Git helpers (self-contained)
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


def get_file_content(
    repo_path: str, commit_hash: str, filepath: str
) -> Optional[str]:
    """Get file content at a specific commit."""
    return git_run(repo_path, ["show", f"{commit_hash}:{filepath}"])


# ---------------------------------------------------------------------------
# Text helpers (self-contained)
# ---------------------------------------------------------------------------


def clean_commit_message(msg: str) -> str:
    """Strip ticket references, hash refs, and cleanup commit messages."""
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
    # Remove inline ticket references
    msg = re.sub(
        r"\(?(?:Refs|Fixed|Closes|Related to)\s+#\d+\)?", "", msg, flags=re.IGNORECASE
    )
    # Remove CVE references prefix
    msg = re.sub(r"^Fixed\s+CVE-[\d-]+\s*--\s*", "", msg, flags=re.IGNORECASE)
    return msg.strip()


def diff_to_search_replace(diff_text: str) -> str:
    """Convert a unified diff into search/replace blocks.

    Each hunk becomes:
    <<<<<<< SEARCH
    old code here
    =======
    new code here
    >>>>>>> REPLACE
    """
    blocks: list[str] = []
    lines = diff_text.split("\n")
    i = 0

    while i < len(lines):
        if not lines[i].startswith("@@"):
            i += 1
            continue

        i += 1  # skip the @@ line
        search_lines: list[str] = []
        replace_lines: list[str] = []

        while i < len(lines) and not lines[i].startswith("@@"):
            line = lines[i]
            if line.startswith("diff --git") or line.startswith("---") or line.startswith("+++") or line.startswith("index "):
                break

            if line.startswith("-"):
                search_lines.append(line[1:])
            elif line.startswith("+"):
                replace_lines.append(line[1:])
            elif line.startswith(" ") or line == "":
                content = line[1:] if line.startswith(" ") else ""
                search_lines.append(content)
                replace_lines.append(content)
            i += 1

        if search_lines or replace_lines:
            block = "<<<<<<< SEARCH\n"
            block += "\n".join(search_lines) + "\n"
            block += "=======\n"
            block += "\n".join(replace_lines) + "\n"
            block += ">>>>>>> REPLACE"
            blocks.append(block)

    return "\n\n".join(blocks)


def parse_hunk_headers(diff_text: str) -> list[tuple[int, int]]:
    """Extract changed line ranges from diff hunk headers (old side)."""
    ranges: list[tuple[int, int]] = []
    for match in re.finditer(r"@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@", diff_text):
        start = int(match.group(1))
        count = int(match.group(2)) if match.group(2) else 1
        end = start + count - 1
        ranges.append((start, end))
    return ranges


def extract_context_around_changes(
    file_content: str, diff_text: str, context_lines: int = 30
) -> str:
    """Extract file content around the changed regions with line numbers."""
    lines = file_content.split("\n")
    total_lines = len(lines)

    if total_lines == 0:
        return file_content

    hunk_ranges = parse_hunk_headers(diff_text)
    if not hunk_ranges:
        return "\n".join(lines[: context_lines * 2])

    included: set[int] = set()
    for start, end in hunk_ranges:
        range_start = max(0, start - 1 - context_lines)
        range_end = min(total_lines, end + context_lines)
        for i in range(range_start, range_end):
            included.add(i)

    if not included:
        return "\n".join(lines[: context_lines * 2])

    sorted_indices = sorted(included)
    output_lines: list[str] = []
    prev_idx = -2

    for idx in sorted_indices:
        if idx > prev_idx + 1:
            if output_lines:
                output_lines.append("... (lines omitted) ...")
        output_lines.append(f"{idx + 1:>6} | {lines[idx]}")
        prev_idx = idx

    return "\n".join(output_lines)


# ---------------------------------------------------------------------------
# Error type inference from test code
# ---------------------------------------------------------------------------

# Pattern: self.assertRaises(SomeException) or with self.assertRaises(SomeException):
RE_ASSERT_RAISES = re.compile(
    r"(?:self\.)?assertRaises\(\s*([A-Za-z_]\w*(?:\.\w+)*)\s*\)"
)
# Pattern: pytest.raises(SomeException)
RE_PYTEST_RAISES = re.compile(
    r"pytest\.raises\(\s*([A-Za-z_]\w*(?:\.\w+)*)\s*\)"
)
# Pattern: with self.assertRaisesMessage(SomeException, ...)
RE_ASSERT_RAISES_MSG = re.compile(
    r"(?:self\.)?assertRaisesMessage\(\s*([A-Za-z_]\w*(?:\.\w+)*)\s*,"
)

# Assertion patterns -> error description
ASSERTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(?:self\.)?assertEqual\("), "AssertionError: values not equal"),
    (re.compile(r"(?:self\.)?assertNotEqual\("), "AssertionError: values unexpectedly equal"),
    (re.compile(r"(?:self\.)?assertIn\("), "AssertionError: item not found in container"),
    (re.compile(r"(?:self\.)?assertNotIn\("), "AssertionError: item unexpectedly found in container"),
    (re.compile(r"(?:self\.)?assertTrue\("), "AssertionError: condition is not True"),
    (re.compile(r"(?:self\.)?assertFalse\("), "AssertionError: condition is not False"),
    (re.compile(r"(?:self\.)?assertIsNone\("), "AssertionError: value is not None"),
    (re.compile(r"(?:self\.)?assertIsNotNone\("), "AssertionError: value is unexpectedly None"),
    (re.compile(r"(?:self\.)?assertGreater\("), "AssertionError: value not greater"),
    (re.compile(r"(?:self\.)?assertLess\("), "AssertionError: value not less"),
    (re.compile(r"(?:self\.)?assertContains\("), "AssertionError: content not found in response"),
    (re.compile(r"(?:self\.)?assertNotContains\("), "AssertionError: content unexpectedly found in response"),
    (re.compile(r"(?:self\.)?assertQuerySetEqual\("), "AssertionError: queryset mismatch"),
    (re.compile(r"\bassert\s+.+"), "AssertionError"),
]

# Recognizable assertion/raise patterns for eligibility check
RE_ASSERTION_LINE = re.compile(
    r"^\+\s*(?:self\.)?(?:assert\w+|raise\s+\w+|pytest\.raises)"
    r"|^\+\s*assert\s+"
)


def infer_error_type(test_code: str) -> str:
    """Infer the error type from test code patterns.

    Returns a string describing the likely error type.
    """
    # Check for assertRaises / pytest.raises first (explicit exception)
    m = RE_ASSERT_RAISES.search(test_code)
    if m:
        return m.group(1)

    m = RE_PYTEST_RAISES.search(test_code)
    if m:
        return m.group(1)

    m = RE_ASSERT_RAISES_MSG.search(test_code)
    if m:
        return m.group(1)

    # Check assertion patterns
    for pattern, error_desc in ASSERTION_PATTERNS:
        if pattern.search(test_code):
            return error_desc

    return "Test failure"


def extract_test_info(test_patch: str) -> list[dict[str, Any]]:
    """Extract test file, class, method, and assertion info from test patch.

    Returns a list of dicts with keys: file, class_name, method_name, code_snippet, error_type
    """
    results: list[dict[str, Any]] = []

    # Split by file
    file_sections = re.split(r"^diff --git a/(.+?) b/(.+?)$", test_patch, flags=re.MULTILINE)

    # file_sections: ['', a_path, b_path, content, a_path2, b_path2, content2, ...]
    i = 1
    while i < len(file_sections):
        if i + 2 >= len(file_sections):
            break
        file_path = file_sections[i + 1]  # b/ path
        section_content = file_sections[i + 2]
        i += 3

        if not file_path.endswith(".py"):
            continue

        # Check if this file has added assertion lines
        added_lines = []
        for line in section_content.split("\n"):
            if line.startswith("+") and not line.startswith("+++"):
                added_lines.append(line)

        has_assertions = any(RE_ASSERTION_LINE.match(line) for line in added_lines)
        if not has_assertions:
            continue

        # Extract added test methods and their assertion code
        # Look for added test method definitions
        test_methods = _extract_test_methods_from_patch(section_content, file_path)
        if test_methods:
            results.extend(test_methods)
        else:
            # Fallback: treat the whole added section as a test snippet
            added_code = "\n".join(line[1:] for line in added_lines if line.startswith("+"))
            if added_code.strip():
                error_type = infer_error_type(added_code)
                results.append({
                    "file": file_path,
                    "class_name": None,
                    "method_name": None,
                    "code_snippet": added_code,
                    "error_type": error_type,
                })

    return results


def _extract_test_methods_from_patch(patch_section: str, file_path: str) -> list[dict[str, Any]]:
    """Extract individual test methods from a diff section.

    Looks for added/modified test method definitions and their assertion bodies.
    """
    results: list[dict[str, Any]] = []

    # Collect all added lines with their context
    lines = patch_section.split("\n")
    added_blocks: list[list[str]] = []
    current_block: list[str] = []

    for line in lines:
        if line.startswith("+") and not line.startswith("+++"):
            current_block.append(line[1:])  # strip the +
        elif line.startswith(" "):
            if current_block:
                current_block.append(line[1:] if len(line) > 1 else "")
            # Context line - include if we're in a block
        else:
            if current_block:
                added_blocks.append(current_block)
                current_block = []

    if current_block:
        added_blocks.append(current_block)

    # Parse each block for test method patterns
    current_class: Optional[str] = None
    current_method: Optional[str] = None
    method_lines: list[str] = []

    all_added_code = "\n".join(
        line[1:] for line in lines
        if line.startswith("+") and not line.startswith("+++")
    )

    # Try to find class and method names from the full patch section (including context)
    for line in lines:
        stripped = ""
        if line.startswith("+") and not line.startswith("+++"):
            stripped = line[1:]
        elif line.startswith(" "):
            stripped = line[1:] if len(line) > 1 else ""
        elif line.startswith("@@"):
            # Hunk headers sometimes contain function/class context
            ctx_match = re.search(r"@@.*@@\s*(class\s+(\w+)|def\s+(\w+))", line)
            if ctx_match:
                if ctx_match.group(2):
                    current_class = ctx_match.group(2)
                elif ctx_match.group(3):
                    current_method = ctx_match.group(3)
            continue
        else:
            continue

        # Check for class definition
        class_match = re.match(r"^class\s+(\w+)", stripped)
        if class_match:
            current_class = class_match.group(1)
            continue

        # Check for method/function definition
        method_match = re.match(r"^\s*def\s+(test\w*)\s*\(", stripped)
        if method_match:
            # Save previous method if exists
            if current_method and method_lines:
                code = "\n".join(method_lines)
                error_type = infer_error_type(code)
                if any(RE_ASSERTION_LINE.match("+" + ml.lstrip()) for ml in method_lines):
                    results.append({
                        "file": file_path,
                        "class_name": current_class,
                        "method_name": current_method,
                        "code_snippet": code,
                        "error_type": error_type,
                    })
            current_method = method_match.group(1)
            method_lines = [stripped]
            continue

        if current_method:
            method_lines.append(stripped)

    # Don't forget last method
    if current_method and method_lines:
        code = "\n".join(method_lines)
        error_type = infer_error_type(code)
        if any(RE_ASSERTION_LINE.match("+" + ml.lstrip()) for ml in method_lines):
            results.append({
                "file": file_path,
                "class_name": current_class,
                "method_name": current_method,
                "code_snippet": code,
                "error_type": error_type,
            })

    # If no structured methods found, fallback to whole added code
    if not results and all_added_code.strip():
        error_type = infer_error_type(all_added_code)
        # Check for any class/method in hunk headers
        hunk_class = None
        hunk_method = None
        for line in lines:
            if line.startswith("@@"):
                ctx = re.search(r"@@.*@@\s*(?:class\s+(\w+)|def\s+(test\w*))", line)
                if ctx:
                    if ctx.group(1):
                        hunk_class = ctx.group(1)
                    if ctx.group(2):
                        hunk_method = ctx.group(2)

        results.append({
            "file": file_path,
            "class_name": hunk_class or current_class,
            "method_name": hunk_method or current_method,
            "code_snippet": all_added_code,
            "error_type": error_type,
        })

    return results


# ---------------------------------------------------------------------------
# Eligibility check
# ---------------------------------------------------------------------------


def is_eligible(commit: dict[str, Any]) -> bool:
    """Check if a commit is eligible for error_diagnosis task generation.

    Requirements:
    - At least one test file with added assertion lines (not just refactoring)
    - At least one source file change
    - The test must contain recognizable assertion patterns
    """
    # Must have source changes
    src_files = commit.get("src_files", [])
    has_source_change = any(
        f.get("path", "").endswith(".py") and f.get("diff", "").strip()
        for f in src_files
    )
    if not has_source_change:
        return False

    # Must have test patch with assertions
    test_patch = commit.get("test_patch", "")
    if not test_patch:
        return False

    # Check for added assertion lines in test patch
    for line in test_patch.split("\n"):
        if RE_ASSERTION_LINE.match(line):
            return True

    return False


# ---------------------------------------------------------------------------
# Task generator
# ---------------------------------------------------------------------------


def generate_error_diagnosis(
    commit: dict[str, Any], repo_path: str
) -> list[dict[str, Any]]:
    """Generate error_diagnosis tasks from a commit.

    For each test change with assertions, construct a synthetic error report
    paired with the diagnosis (root cause + fix).
    """
    parent_hash = commit.get("parent_hash")
    if not parent_hash:
        return []

    test_patch = commit.get("test_patch", "")
    if not test_patch:
        return []

    # Extract test info (file, class, method, assertions, error type)
    test_infos = extract_test_info(test_patch)
    if not test_infos:
        return []

    # Get source file changes for the fix
    src_files = commit.get("src_files", [])
    py_src_files = [
        f for f in src_files
        if f.get("path", "").endswith(".py") and f.get("diff", "").strip()
    ]
    if not py_src_files:
        return []

    # Prepare the fix output (combined source changes)
    fix_parts: list[str] = []
    location_parts: list[str] = []

    for src_file in py_src_files:
        filepath = src_file["path"]
        diff_text = src_file.get("diff", "")
        if not diff_text:
            continue

        # Get line range from hunk headers
        hunk_ranges = parse_hunk_headers(diff_text)
        if hunk_ranges:
            line_start = min(r[0] for r in hunk_ranges)
            line_end = max(r[1] for r in hunk_ranges)
            location_parts.append(f"{filepath}:{line_start}-{line_end}")
        else:
            location_parts.append(filepath)

        # Convert diff to SEARCH/REPLACE format
        sr_block = diff_to_search_replace(diff_text)
        if sr_block.strip():
            fix_parts.append(f"File: {filepath}\n{sr_block}")

    if not fix_parts:
        return []

    # Get source context (code BEFORE fix) for each source file
    source_contexts: list[str] = []
    for src_file in py_src_files:
        filepath = src_file["path"]
        diff_text = src_file.get("diff", "")
        if not diff_text:
            continue

        content = get_file_content(repo_path, parent_hash, filepath)
        if content is None:
            continue

        # Extract context around the changed lines (30 lines)
        context = extract_context_around_changes(content, diff_text, context_lines=30)
        if context.strip():
            source_contexts.append(f"File: {filepath}\n```python\n{context}\n```")

    if not source_contexts:
        return []

    # Build root cause from commit message
    root_cause = clean_commit_message(commit.get("message", "") or commit.get("commit_message", ""))
    if not root_cause:
        return []

    # Combined location string
    location_str = "; ".join(location_parts)

    # Combined fix
    fix_str = "\n\n".join(fix_parts)

    # Generate one task per test info (or combine if multiple tests point to same fix)
    results: list[dict[str, Any]] = []
    seen_methods: set[str] = set()

    for test_info in test_infos:
        # Avoid duplicates
        method_key = f"{test_info['file']}::{test_info.get('class_name', '')}::{test_info.get('method_name', '')}"
        if method_key in seen_methods:
            continue
        seen_methods.add(method_key)

        # Build error location string
        error_location_parts = [test_info["file"]]
        if test_info.get("class_name"):
            error_location_parts.append(test_info["class_name"])
        if test_info.get("method_name"):
            error_location_parts.append(test_info["method_name"])
        error_location = "::".join(error_location_parts)

        # Build test code snippet (truncate if too long)
        test_snippet = test_info["code_snippet"]
        snippet_lines = test_snippet.split("\n")
        if len(snippet_lines) > 50:
            test_snippet = "\n".join(snippet_lines[:50]) + "\n... (truncated)"

        # Construct synthetic error report (input)
        input_text = (
            f"Error in {error_location}\n"
            f"\n"
            f"The following test fails:\n"
            f"```python\n"
            f"{test_snippet}\n"
            f"```\n"
            f"\n"
            f"Error type: {test_info['error_type']}\n"
            f"\n"
            f"Relevant source code:\n"
            + "\n".join(source_contexts)
        )

        # Construct structured diagnosis (output)
        output_text = (
            f"Root cause: {root_cause}\n"
            f"Location: {location_str}\n"
            f"Fix:\n{fix_str}"
        )

        results.append({
            "task_type": "error_diagnosis",
            "repo": commit.get("repo", ""),
            "commit": commit.get("commit_hash", ""),
            "input": input_text,
            "output": output_text,
            "metadata": {
                "test_file": test_info["file"],
                "test_class": test_info.get("class_name"),
                "test_method": test_info.get("method_name"),
                "error_type": test_info["error_type"],
                "n_source_files": len(py_src_files),
                "source_files": [f["path"] for f in py_src_files],
            },
        })

    return results


# ---------------------------------------------------------------------------
# Commit normalization (same as generate_tasks.py)
# ---------------------------------------------------------------------------


def normalize_commit(commit: dict[str, Any], repo_path: str) -> dict[str, Any]:
    """Normalize field names from parse_commits.py output to what generators expect."""
    if "message" not in commit and "commit_message" in commit:
        commit["message"] = commit["commit_message"]

    if "repo" not in commit:
        commit["repo"] = os.path.basename(repo_path)

    if "src_files" not in commit and "changed_files" in commit:
        source_patch = commit.get("source_patch", "")
        file_patches: dict[str, str] = {}
        if source_patch:
            segments = re.split(r"^(diff --git .+)$", source_patch, flags=re.MULTILINE)
            i = 1
            while i < len(segments):
                header = segments[i]
                content = segments[i + 1] if i + 1 < len(segments) else ""
                match = re.match(r"diff --git a/(.+?) b/(.+)", header)
                if match:
                    fpath = match.group(2)
                    file_patches[fpath] = header + content
                i += 2

        src_files = []
        for f in commit["changed_files"]:
            if not f.get("is_test", False):
                fpath = f.get("filepath", f.get("path", ""))
                src_files.append({
                    "path": fpath,
                    "diff": file_patches.get(fpath, ""),
                })
        commit["src_files"] = src_files

    if "src_patch" not in commit and "source_patch" in commit:
        commit["src_patch"] = commit["source_patch"]

    return commit


# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------


def process_commit(
    commit_json: str,
    repo_path: str,
) -> list[dict[str, Any]]:
    """Process a single commit and generate error_diagnosis tasks.

    Runs in a worker process.
    """
    try:
        commit = json.loads(commit_json)
    except json.JSONDecodeError:
        return []

    commit = normalize_commit(commit, repo_path)

    if not is_eligible(commit):
        return []

    try:
        return generate_error_diagnosis(commit, repo_path)
    except Exception as e:
        logger.debug(f"Error processing commit {commit.get('commit_hash', '?')}: {e}")
        return []


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


def run(
    input_path: str,
    output_dir: str,
    repo_path: str,
    workers: int,
    sample: Optional[int] = None,
) -> None:
    """Main execution: load commits, process in parallel, write output."""
    # Validate inputs
    if not Path(input_path).exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    repo_path = os.path.abspath(repo_path)
    if not os.path.isdir(os.path.join(repo_path, ".git")):
        logger.error(f"Not a git repo: {repo_path}")
        sys.exit(1)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load commits
    logger.info(f"Loading commits from {input_path}")
    commit_lines = load_commits(input_path, sample)
    logger.info(f"Loaded {len(commit_lines)} commits")

    if not commit_lines:
        logger.warning("No commits to process")
        return

    output_path = out_dir / "error_diagnosis.jsonl"
    total_tasks = 0
    eligible_count = 0
    errors = 0

    logger.info(f"Processing with {workers} workers")
    logger.info(f"Output: {output_path}")

    with open(output_path, "w", encoding="utf-8") as out_f:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_commit, commit_line, repo_path): i
                for i, commit_line in enumerate(commit_lines)
            }

            with tqdm(total=len(futures), desc="Generating debugging tasks") as pbar:
                for future in as_completed(futures):
                    try:
                        tasks = future.result(timeout=120)
                        if tasks:
                            eligible_count += 1
                            for task in tasks:
                                out_f.write(json.dumps(task, ensure_ascii=False) + "\n")
                                total_tasks += 1
                    except Exception as e:
                        errors += 1
                        logger.debug(f"Worker error: {e}")
                    pbar.update(1)

    # Summary
    logger.info("=" * 60)
    logger.info("Generation complete!")
    logger.info(f"  Commits processed: {len(commit_lines)}")
    logger.info(f"  Eligible commits: {eligible_count}")
    logger.info(f"  Tasks generated: {total_tasks}")
    logger.info(f"  Errors/skipped: {errors}")
    logger.info(f"  Output: {output_path}")
    logger.info("=" * 60)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate iterative debugging training tasks from filtered commits.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to filtered commits JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for output JSONL file (error_diagnosis.jsonl)",
    )
    parser.add_argument(
        "--repo-path",
        required=True,
        help="Path to the git repository (for file content retrieval)",
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
        help="Only process first N commits (for testing)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        input_path=args.input,
        output_dir=args.output_dir,
        repo_path=args.repo_path,
        workers=args.workers,
        sample=args.sample,
    )
