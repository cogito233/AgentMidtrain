#!/usr/bin/env python3
"""Filter parsed commit data by quality and relevance criteria.

Reads JSONL output from parse_commits.py and applies configurable filters
to select high-quality commits suitable for midtrain data synthesis.

Usage:
    python filter_commits.py input.jsonl --output filtered.jsonl --require-test --max-src-files 5
    python filter_commits.py input.jsonl -o filtered.jsonl --fix-keywords --python-only
"""

import argparse
import json
import logging
import os
import re
import sys
from typing import Any

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Keywords that suggest a bug-fix commit
FIX_KEYWORDS_PATTERN = re.compile(
    r"\b(fix|bug|repair|patch|resolve|closes?|fixes|fixed|hotfix|defect|issue)\b",
    re.IGNORECASE,
)

# Pattern to detect comment/docstring-only changes
COMMENT_PATTERNS = re.compile(
    r"^\s*([#].*|\"\"\".*\"\"\"|\'\'\'.*\'\'\'|/\*.*\*/|//.*|\*.*|\"\"\"|\'\'\')$"
)


def load_jsonl(filepath: str) -> list[dict[str, Any]]:
    """Load commits from a JSONL file.

    Args:
        filepath: Path to JSONL file.

    Returns:
        List of commit dicts.
    """
    commits = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                commits.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed JSON at line {line_num}: {e}")
    return commits


def is_substantive_change(patch: str) -> bool:
    """Check if a patch contains substantive (non-comment, non-whitespace) changes.

    Heuristic: looks at added/removed lines and checks whether any of them
    are not purely comments, docstrings, or whitespace.

    Args:
        patch: Unified diff string for source files.

    Returns:
        True if the patch contains substantive code changes.
    """
    if not patch.strip():
        return False

    for line in patch.split("\n"):
        # Only look at added/removed lines (not context or headers)
        if not (line.startswith("+") or line.startswith("-")):
            continue
        if line.startswith("+++") or line.startswith("---"):
            continue

        # Strip the +/- prefix
        content = line[1:].strip()

        # Skip empty lines
        if not content:
            continue

        # Skip pure comment/docstring lines
        if COMMENT_PATTERNS.match(content):
            continue

        # Found a substantive line
        return True

    return False


def get_source_edit_lines(commit: dict[str, Any]) -> int:
    """Count total lines changed in source (non-test) files.

    Args:
        commit: Parsed commit dict.

    Returns:
        Total insertions + deletions in source files.
    """
    total = 0
    for f in commit.get("changed_files", []):
        if not f.get("is_test", False):
            total += f.get("insertions", 0) + f.get("deletions", 0)
    return total


def all_python_files(commit: dict[str, Any]) -> bool:
    """Check if all changed files are Python files.

    Args:
        commit: Parsed commit dict.

    Returns:
        True if all changed files end with .py.
    """
    files = commit.get("changed_files", [])
    if not files:
        return False
    return all(f["filepath"].endswith(".py") for f in files)


def all_go_files(commit: dict[str, Any]) -> bool:
    """Check if all changed files are Go source files (excluding tests and vendor).

    Args:
        commit: Parsed commit dict.

    Returns:
        True if all changed files end with .go, are not _test.go, and not in vendor/.
    """
    files = commit.get("changed_files", [])
    if not files:
        return False
    for f in files:
        filepath = f["filepath"]
        if not filepath.endswith(".go"):
            return False
        if filepath.endswith("_test.go"):
            return False
        if filepath.startswith("vendor/") or "/vendor/" in filepath:
            return False
    return True


# TypeScript/JavaScript file extensions
TS_EXTENSIONS = (".ts", ".tsx", ".js", ".jsx", ".mts", ".mjs")

# Directories to exclude for TypeScript projects
TS_EXCLUDED_DIRS = ("node_modules/", "dist/", "build/", ".next/")

# Config file patterns to exclude for TypeScript projects
TS_CONFIG_PATTERNS = re.compile(
    r"(.*\.config\.(js|ts|mjs|cjs|mts)$"
    r"|jest\.config\..*"
    r"|vitest\.config\..*"
    r"|webpack\.config\..*"
    r"|rollup\.config\..*"
    r"|babel\.config\..*"
    r"|tsconfig\..*\.json$"
    r"|\.eslintrc\..*"
    r"|\.prettierrc\..*"
    r"|tailwind\.config\..*"
    r"|next\.config\..*"
    r"|vite\.config\..*)",
    re.IGNORECASE,
)

# TypeScript test file patterns
TS_TEST_PATTERNS = re.compile(
    r"(.*\.test\.(ts|tsx|js|jsx)$"
    r"|.*\.spec\.(ts|tsx|js|jsx)$"
    r"|.*/__tests__/.*"
    r"|.*/test/.*"
    r"|.*/tests/.*)",
    re.IGNORECASE,
)


def all_typescript_files(commit: dict[str, Any]) -> bool:
    """Check if all changed files are TypeScript/JavaScript files.

    Validates that:
    - All files have TS/JS extensions (.ts, .tsx, .js, .jsx, .mts, .mjs)
    - No files are in excluded directories (node_modules/, dist/, build/, .next/)
    - No files are config files (*.config.js, *.config.ts, jest.config.*, etc.)

    Args:
        commit: Parsed commit dict.

    Returns:
        True if all changed files are valid TypeScript/JavaScript source files.
    """
    files = commit.get("changed_files", [])
    if not files:
        return False

    for f in files:
        filepath = f.get("filepath", f.get("path", ""))
        # Check extension
        if not any(filepath.endswith(ext) for ext in TS_EXTENSIONS):
            return False
        # Check excluded directories
        if any(excluded in filepath for excluded in TS_EXCLUDED_DIRS):
            return False
        # Check config files
        basename = filepath.rsplit("/", 1)[-1] if "/" in filepath else filepath
        if TS_CONFIG_PATTERNS.match(basename):
            return False

    return True


def apply_filters(
    commit: dict[str, Any],
    max_src_files: int,
    max_edit_lines: int,
    max_patch_length: int,
    require_test: bool,
    require_src: bool,
    python_only: bool,
    exclude_docs: bool,
    fix_keywords: bool,
    language: str = "python",
) -> bool:
    """Apply all configured filters to a single commit.

    Args:
        commit: Parsed commit dict.
        max_src_files: Maximum number of source files changed.
        max_edit_lines: Maximum total lines changed in source.
        max_patch_length: Maximum patch string length (characters).
        require_test: Require at least one test file changed.
        require_src: Require at least one source file changed.
        python_only: All changed files must be .py (legacy, use language instead).
        exclude_docs: Exclude commits with only comment/docstring changes.
        fix_keywords: Only include commits with fix-related keywords.
        language: Language filter - "python", "go", "typescript", or "all".

    Returns:
        True if the commit passes all filters.
    """
    # Filter: max source files
    if commit.get("num_src_files", 0) > max_src_files:
        return False

    # Filter: max edit lines in source
    src_edit_lines = get_source_edit_lines(commit)
    if src_edit_lines > max_edit_lines:
        return False

    # Filter: max patch length
    source_patch = commit.get("source_patch", "")
    if len(source_patch) > max_patch_length:
        return False

    # Filter: require test files
    if require_test and commit.get("num_test_files", 0) == 0:
        return False

    # Filter: require source files
    if require_src and commit.get("num_src_files", 0) == 0:
        return False

    # Filter: python only (legacy flag)
    if python_only and not all_python_files(commit):
        return False

    # Filter: language-specific source file check
    if language == "python" and not python_only:
        if not all_python_files(commit):
            return False
    elif language == "go":
        if not all_go_files(commit):
            return False
    elif language == "typescript":
        if not all_typescript_files(commit):
            return False
    # language == "all" -> no language-specific filter

    # Filter: exclude docs-only changes
    if exclude_docs:
        if not is_substantive_change(source_patch):
            return False

    # Filter: fix keywords
    if fix_keywords:
        message = commit.get("commit_message", "")
        if not FIX_KEYWORDS_PATTERN.search(message):
            return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Filter parsed commits by quality and relevance criteria."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Input JSONL file (output of parse_commits.py).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output filtered JSONL file path.",
    )
    parser.add_argument(
        "--max-src-files",
        type=int,
        default=5,
        help="Max number of source files changed (default: 5).",
    )
    parser.add_argument(
        "--max-edit-lines",
        type=int,
        default=200,
        help="Max total lines changed in source files (default: 200).",
    )
    parser.add_argument(
        "--max-patch-length",
        type=int,
        default=10000,
        help="Max source patch string length in characters (default: 10000).",
    )
    parser.add_argument(
        "--require-test",
        action="store_true",
        help="Require at least one test file changed.",
    )
    parser.add_argument(
        "--require-src",
        action="store_true",
        help="Require at least one source file changed.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="python",
        choices=["python", "go", "all"],
        help="Language filter: 'python', 'go', or 'all' (default: python).",
    )
    parser.add_argument(
        "--python-only",
        action="store_true",
        help="All changed files must be .py files (shortcut for --language python).",
    )
    parser.add_argument(
        "--exclude-docs",
        action="store_true",
        help="Exclude commits that only change comments/docstrings.",
    )
    parser.add_argument(
        "--fix-keywords",
        action="store_true",
        help="Only include commits with fix/bug/repair keywords in message.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only process the first N commits from input (for testing).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    input_path = os.path.abspath(args.input_file)
    if not os.path.isfile(input_path):
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Default output name
    if args.output is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        args.output = f"{base}_filtered.jsonl"

    output_path = os.path.abspath(args.output)

    # Load commits
    logger.info(f"Loading commits from {input_path}...")
    commits = load_jsonl(input_path)
    logger.info(f"Loaded {len(commits)} commits.")

    # Apply sample limit
    if args.sample is not None:
        commits = commits[: args.sample]
        logger.info(f"Sampling first {args.sample} commits.")

    # If --python-only is used, treat it as --language python
    language = args.language
    if args.python_only:
        language = "python"

    # Apply filters
    logger.info("Applying filters...")
    logger.info(f"  max_src_files: {args.max_src_files}")
    logger.info(f"  max_edit_lines: {args.max_edit_lines}")
    logger.info(f"  max_patch_length: {args.max_patch_length}")
    logger.info(f"  require_test: {args.require_test}")
    logger.info(f"  require_src: {args.require_src}")
    logger.info(f"  language: {language}")
    logger.info(f"  python_only: {args.python_only}")
    logger.info(f"  exclude_docs: {args.exclude_docs}")
    logger.info(f"  fix_keywords: {args.fix_keywords}")

    filtered = []
    for commit in tqdm(commits, desc="Filtering"):
        if apply_filters(
            commit,
            max_src_files=args.max_src_files,
            max_edit_lines=args.max_edit_lines,
            max_patch_length=args.max_patch_length,
            require_test=args.require_test,
            require_src=args.require_src,
            python_only=args.python_only,
            exclude_docs=args.exclude_docs,
            fix_keywords=args.fix_keywords,
            language=language,
        ):
            filtered.append(commit)

    logger.info(
        f"Filter results: {len(filtered)}/{len(commits)} commits passed "
        f"({len(commits) - len(filtered)} rejected, "
        f"{len(filtered) / len(commits) * 100:.1f}% pass rate)."
        if commits
        else "No commits to filter."
    )

    # Write output
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        for item in filtered:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(filtered)} filtered commits to {output_path}")

    # Print summary statistics
    if filtered:
        avg_src = sum(c["num_src_files"] for c in filtered) / len(filtered)
        avg_test = sum(c["num_test_files"] for c in filtered) / len(filtered)
        avg_ins = sum(c["total_insertions"] for c in filtered) / len(filtered)
        avg_del = sum(c["total_deletions"] for c in filtered) / len(filtered)
        print(f"\n--- Summary ---")
        print(f"Total passed: {len(filtered)}")
        print(f"Avg source files: {avg_src:.1f}")
        print(f"Avg test files: {avg_test:.1f}")
        print(f"Avg insertions: {avg_ins:.1f}")
        print(f"Avg deletions: {avg_del:.1f}")


if __name__ == "__main__":
    main()
