#!/usr/bin/env python3
"""Generate code comprehension tasks from filtered commits.

Reads filtered commit JSONL (output of filter_commits.py) and generates
two JSONL output files for code explanation training:

Task types:
  - code_explanation: Given a code change (diff), explain what it does and why.
  - dependency_analysis: Given a modified file, identify dependencies and dependents.

Usage:
    python generate_explanation_tasks.py \
        --input data/filtered_commits/django_django.jsonl \
        --output-dir data/tasks/ \
        --repo-path repos/django_django \
        --workers 8
"""

from __future__ import annotations

import argparse
import ast
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

ALL_TASK_TYPES = [
    "code_explanation",
    "dependency_analysis",
]


# ---------------------------------------------------------------------------
# Git helpers (reimplemented inline)
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


def get_commit_files(repo_path: str, commit_hash: str) -> Optional[list[str]]:
    """Get all files changed in a specific commit."""
    raw = git_run(
        repo_path,
        ["diff-tree", "--no-commit-id", "--name-only", "-r", commit_hash],
        timeout=30,
    )
    if raw is None:
        return None
    return [f.strip() for f in raw.strip().split("\n") if f.strip()]


# ---------------------------------------------------------------------------
# Diff / commit message helpers (reimplemented inline)
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
    msg = re.sub(r"\(?(?:Refs|Fixed|Closes|Related to)\s+#\d+\)?", "", msg, flags=re.IGNORECASE)
    # Remove CVE references prefix like "Fixed CVE-2026-6907 -- "
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


def extract_imports(source: str) -> list[str]:
    """Extract import statements from Python source code."""
    imports: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Fallback: regex-based extraction
        for line in source.split("\n"):
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                imports.append(stripped)
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = ", ".join(a.name for a in node.names[:5])
            if len(node.names) > 5:
                names += ", ..."
            imports.append(f"from {module} import {names}")

    return imports


def extract_changed_definitions(diff_text: str, source: str) -> list[str]:
    """Extract function/class names that were modified based on the diff.

    Uses hunk headers and AST to identify which definitions overlap with changes.
    """
    # Parse hunk headers to get changed line ranges (new file side)
    changed_lines: set[int] = set()
    for match in re.finditer(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", diff_text):
        start = int(match.group(1))
        count = int(match.group(2)) if match.group(2) else 1
        for line_no in range(start, start + count):
            changed_lines.add(line_no)

    if not changed_lines:
        return []

    # Parse AST to find definitions
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Fallback: regex extraction from diff
        defs: list[str] = []
        for line in diff_text.split("\n"):
            if line.startswith("+") or line.startswith("-"):
                content = line[1:]
                m = re.match(r"\s*(def|class)\s+(\w+)", content)
                if m:
                    defs.append(f"{m.group(1)} {m.group(2)}")
        return list(dict.fromkeys(defs))  # dedupe preserving order

    definitions: list[str] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            class_end = node.end_lineno or node.lineno
            class_lines = set(range(node.lineno, class_end + 1))
            if class_lines & changed_lines:
                # Check if specific methods are changed
                method_found = False
                for item in ast.iter_child_nodes(node):
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        item_end = item.end_lineno or item.lineno
                        item_lines = set(range(item.lineno, item_end + 1))
                        if item_lines & changed_lines:
                            definitions.append(f"{node.name}.{item.name}()")
                            method_found = True
                if not method_found:
                    definitions.append(f"class {node.name}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node_end = node.end_lineno or node.lineno
            node_lines = set(range(node.lineno, node_end + 1))
            if node_lines & changed_lines:
                definitions.append(f"{node.name}()")

    return definitions


def classify_change_category(filepath: str, definitions: list[str]) -> str:
    """Classify the change into a category based on filepath and definitions."""
    path_lower = filepath.lower()

    if "test" in path_lower:
        return "test"
    elif any(kw in path_lower for kw in ["config", "settings", "conf", "setup.py", "setup.cfg", "pyproject"]):
        return "configuration"
    elif any(kw in path_lower for kw in ["__init__", "api", "views", "urls", "serializer"]):
        return "public API"
    else:
        # Check if definitions look like internal helpers (leading underscore)
        if definitions:
            public_defs = [d for d in definitions if not any(
                part.startswith("_") for part in d.replace("()", "").split(".")
            )]
            if not public_defs:
                return "internal API"
        return "public API"


def classify_commit_type(message: str) -> str:
    """Classify commit type from message keywords."""
    msg_lower = message.lower()

    if any(kw in msg_lower for kw in ["fixed", "fix", "bug", "crash", "error", "prevent", "cve"]):
        return "bug_fix"
    elif any(kw in msg_lower for kw in ["added", "add", "support", "implement", "new"]):
        return "feature"
    elif any(kw in msg_lower for kw in ["refactor", "clean", "simplify", "reorganiz", "move"]):
        return "refactor"
    elif any(kw in msg_lower for kw in ["doc", "readme", "comment", "typo"]):
        return "docs"
    elif any(kw in msg_lower for kw in ["test", "assert"]):
        return "test"
    elif any(kw in msg_lower for kw in ["deprecat", "remov", "adjust", "updat", "improv"]):
        return "enhancement"
    else:
        return "enhancement"


def determine_impact(commit: dict[str, Any], filepath: str) -> str:
    """Determine which components/functions are affected by this change."""
    src_files = commit.get("src_files", [])
    # Find the specific file entry
    for sf in src_files:
        if sf.get("path") == filepath:
            diff_text = sf.get("diff", "")
            break
    else:
        diff_text = ""

    # Extract function/class names from diff hunk headers (the @@ ... @@ function_name part)
    affected: list[str] = []
    for match in re.finditer(r"@@ .+? @@\s*(.+?)$", diff_text, re.MULTILINE):
        context = match.group(1).strip()
        if context:
            affected.append(context)

    if affected:
        return "; ".join(dict.fromkeys(affected))  # dedupe

    # Fallback: just the file path components
    parts = filepath.rsplit("/", 1)
    if len(parts) > 1:
        return f"module {parts[0]}, file {parts[1]}"
    return f"file {filepath}"


# ---------------------------------------------------------------------------
# Normalization (same as generate_tasks.py)
# ---------------------------------------------------------------------------


def normalize_commit(commit: dict[str, Any], repo_path: str) -> dict[str, Any]:
    """Normalize field names from parse_commits.py output to what generators expect."""
    # message field
    if "message" not in commit and "commit_message" in commit:
        commit["message"] = commit["commit_message"]

    # repo field (derive from repo_path if missing)
    if "repo" not in commit:
        commit["repo"] = os.path.basename(repo_path)

    # src_files: list of {path, diff} for non-test files
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

    # Normalize patch field names
    if "src_patch" not in commit and "source_patch" in commit:
        commit["src_patch"] = commit["source_patch"]

    return commit


# ---------------------------------------------------------------------------
# Task generators
# ---------------------------------------------------------------------------


def generate_code_explanation(
    commit: dict[str, Any], repo_path: str
) -> list[dict[str, Any]]:
    """Generate code explanation tasks: explain what a change does and why.

    Only for commits with clean commit messages > 50 chars (need good explanations).
    Each changed source file produces one task.
    """
    cleaned_message = clean_commit_message(commit.get("message", ""))
    if len(cleaned_message) < 50:
        return []

    src_files = commit.get("src_files", [])
    if not src_files:
        return []

    commit_type = classify_commit_type(commit.get("message", ""))

    results: list[dict[str, Any]] = []

    for src_file in src_files:
        filepath = src_file["path"]
        diff_text = src_file.get("diff", "")

        if not diff_text:
            continue

        # Convert diff to SEARCH/REPLACE format for the input
        search_replace = diff_to_search_replace(diff_text)
        if not search_replace.strip():
            continue

        # Determine impact (which components are affected)
        impact = determine_impact(commit, filepath)

        # Build input
        input_text = (
            f"File: {filepath}\n\n"
            f"Change:\n{search_replace}"
        )

        # Build structured output
        # What: 1-2 sentence summary - use first sentence of cleaned message
        what_parts = cleaned_message.split(".")
        what = what_parts[0].strip() + "." if what_parts[0].strip() else cleaned_message[:100]

        # Why: motivation from cleaned commit message
        why = cleaned_message

        output_text = (
            f"What: {what}\n"
            f"Why: {why}\n"
            f"Impact: {impact}"
        )

        results.append({
            "task_type": "code_explanation",
            "repo": commit.get("repo", ""),
            "commit": commit.get("commit_hash", ""),
            "input": input_text,
            "output": output_text,
            "metadata": {
                "filepath": filepath,
                "commit_type": commit_type,
                "message_length": len(cleaned_message),
            },
        })

    return results


def generate_dependency_analysis(
    commit: dict[str, Any], repo_path: str
) -> list[dict[str, Any]]:
    """Generate dependency analysis tasks.

    Given a modified file, identify which other files depend on it or it depends on.
    Uses co-changed files in the same commit as a proxy for dependency information.
    """
    commit_hash = commit.get("commit_hash", "")
    if not commit_hash:
        return []

    src_files = commit.get("src_files", [])
    if not src_files:
        return []

    # Get all files changed in this commit (from git directly)
    all_commit_files = get_commit_files(repo_path, commit_hash)
    if all_commit_files is None:
        # Fallback: use src_files + any test files from commit data
        all_commit_files = [sf["path"] for sf in src_files]
        test_patch = commit.get("test_patch", "")
        if test_patch:
            test_files = re.findall(r"^diff --git a/.+ b/(.+)$", test_patch, re.MULTILINE)
            all_commit_files.extend(test_files)

    results: list[dict[str, Any]] = []

    for src_file in src_files:
        filepath = src_file["path"]
        diff_text = src_file.get("diff", "")

        if not filepath.endswith(".py"):
            continue

        # Get file content at the commit (after the change)
        content = get_file_content(repo_path, commit_hash, filepath)
        if content is None:
            continue

        # Extract imports
        imports = extract_imports(content)
        if not imports:
            continue

        # Extract changed definitions
        changed_defs = extract_changed_definitions(diff_text, content) if diff_text else []

        # Build input
        imports_text = "\n".join(imports)
        defs_text = "\n".join(changed_defs) if changed_defs else "(no specific function/class changes identified)"

        input_text = (
            f"File: {filepath}\n\n"
            f"Imports in this file:\n{imports_text}\n\n"
            f"Modified functions/classes:\n{defs_text}"
        )

        # Build output

        # Direct dependencies (from imports)
        dep_modules: list[str] = []
        for imp in imports:
            if imp.startswith("from "):
                m = re.match(r"from\s+([\w.]+)", imp)
                if m:
                    dep_modules.append(m.group(1))
            elif imp.startswith("import "):
                m = re.match(r"import\s+([\w.]+)", imp)
                if m:
                    dep_modules.append(m.group(1))
        dep_modules = list(dict.fromkeys(dep_modules))  # dedupe preserving order

        # Likely dependents: other files changed in the same commit (proxy for deps)
        other_files = [f for f in all_commit_files if f != filepath]

        # Classify change category
        category = classify_change_category(filepath, changed_defs)

        deps_text = "\n".join(f"- {m}" for m in dep_modules) if dep_modules else "- (none identified)"
        dependents_text = "\n".join(f"- {f}" for f in other_files) if other_files else "- (none - isolated change)"

        output_text = (
            f"Direct dependencies (imports from):\n{deps_text}\n\n"
            f"Likely dependents (files that may need updates):\n{dependents_text}\n\n"
            f"Change category: {category}"
        )

        results.append({
            "task_type": "dependency_analysis",
            "repo": commit.get("repo", ""),
            "commit": commit_hash,
            "input": input_text,
            "output": output_text,
            "metadata": {
                "filepath": filepath,
                "n_imports": len(imports),
                "n_changed_defs": len(changed_defs),
                "n_co_changed_files": len(other_files),
                "change_category": category,
            },
        })

    return results


# ---------------------------------------------------------------------------
# Worker function (processes one commit, returns task results)
# ---------------------------------------------------------------------------


def process_commit(
    commit_json: str,
    repo_path: str,
    task_types: list[str],
) -> dict[str, list[dict[str, Any]]]:
    """Process a single commit and generate all requested task types.

    This function runs in a worker process. Returns a dict mapping
    task_type -> list of generated QA pairs.
    """
    try:
        commit = json.loads(commit_json)
    except json.JSONDecodeError:
        return {}

    # Normalize field names to match what generators expect
    commit = normalize_commit(commit, repo_path)

    results: dict[str, list[dict[str, Any]]] = {t: [] for t in task_types}

    try:
        if "code_explanation" in task_types:
            tasks = generate_code_explanation(commit, repo_path)
            results["code_explanation"].extend(tasks)

        if "dependency_analysis" in task_types:
            tasks = generate_dependency_analysis(commit, repo_path)
            results["dependency_analysis"].extend(tasks)

    except Exception as e:
        # Log but don't crash the worker
        logger.debug(
            f"Error processing commit {commit.get('commit_hash', '?')}: {e}"
        )

    return results


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
    task_types: list[str],
    workers: int,
    sample: Optional[int],
) -> None:
    """Main execution: load commits, process in parallel, write outputs."""
    # Validate inputs
    input_file = Path(input_path)
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)

    repo_path = os.path.abspath(repo_path)
    if not os.path.isdir(os.path.join(repo_path, ".git")):
        logger.error(f"Not a git repo: {repo_path}")
        sys.exit(1)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load commits
    logger.info(f"Loading commits from {input_file}")
    commit_lines = load_commits(str(input_file), sample)
    logger.info(f"Loaded {len(commit_lines)} commits")

    if not commit_lines:
        logger.warning("No commits to process")
        return

    # Open output files
    output_files: dict[str, Path] = {}
    for task_type in task_types:
        output_files[task_type] = out_dir / f"{task_type}.jsonl"

    # Counters
    counters: dict[str, int] = {t: 0 for t in task_types}
    errors = 0

    # Process commits in parallel
    logger.info(
        f"Processing with {workers} workers, "
        f"task types: {task_types}"
    )

    # Open all output files for writing
    writers: dict[str, Any] = {}
    for task_type, path in output_files.items():
        writers[task_type] = open(path, "w", encoding="utf-8")

    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_commit,
                    commit_line,
                    repo_path,
                    task_types,
                ): i
                for i, commit_line in enumerate(commit_lines)
            }

            with tqdm(total=len(futures), desc="Processing commits") as pbar:
                for future in as_completed(futures):
                    try:
                        results = future.result(timeout=120)
                        for task_type, tasks in results.items():
                            for task in tasks:
                                writers[task_type].write(
                                    json.dumps(task, ensure_ascii=False) + "\n"
                                )
                                counters[task_type] += 1
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
    logger.info(f"  Commits processed: {len(commit_lines)}")
    logger.info(f"  Errors/skipped: {errors}")
    for task_type in task_types:
        count = counters[task_type]
        path = output_files[task_type]
        logger.info(f"  {task_type}: {count} pairs -> {path}")
    logger.info("=" * 60)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate code comprehension tasks (explanation + dependency analysis).",
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
        help="Directory for output JSONL files",
    )
    parser.add_argument(
        "--repo-path",
        required=True,
        help="Path to the git repository (for file content retrieval)",
    )
    parser.add_argument(
        "--task-types",
        default=",".join(ALL_TASK_TYPES),
        help=(
            f"Comma-separated list of task types to generate. "
            f"Default: all ({','.join(ALL_TASK_TYPES)})"
        ),
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

    args = parser.parse_args()

    # Validate task types
    task_types = [t.strip() for t in args.task_types.split(",")]
    invalid = [t for t in task_types if t not in ALL_TASK_TYPES]
    if invalid:
        parser.error(
            f"Invalid task types: {invalid}. Valid: {ALL_TASK_TYPES}"
        )

    args.task_types_list = task_types
    return args


if __name__ == "__main__":
    args = parse_args()
    run(
        input_path=args.input,
        output_dir=args.output_dir,
        repo_path=args.repo_path,
        task_types=args.task_types_list,
        workers=args.workers,
        sample=args.sample,
    )
