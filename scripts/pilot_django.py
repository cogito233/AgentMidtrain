#!/usr/bin/env python3
"""
Pilot analysis of Django repository for mid-training data synthesis.
Extracts commits, applies quality filters, generates QA pairs for 5 task types.
"""

import subprocess
import json
import re
import os
from datetime import datetime
from collections import defaultdict
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────────────────
REPO_DIR = "/data_fast_v3/eremite/cogito_explore/AgentMidtrain/repos/django_django"
OUTPUT_DIR = "/data_fast_v3/eremite/cogito_explore/AgentMidtrain/data"
NUM_COMMITS = 2000
MAX_SRC_FILES = 5
MAX_PATCH_LEN = 10000
SAMPLES_PER_TASK = 5

BUG_KEYWORDS = re.compile(
    r'\b(fix|bug|issue|error|crash|fault|defect|regression|broken|incorrect|wrong|fail|patch)\b',
    re.IGNORECASE
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def git(*args, timeout=30):
    """Run a git command in the repo directory."""
    result = subprocess.run(
        ["git", "-C", REPO_DIR] + list(args),
        capture_output=True, text=True, timeout=timeout
    )
    return result.stdout, result.returncode


def classify_file(path):
    """Classify a file as src or test."""
    if not path.endswith('.py'):
        return 'other'
    parts = path.lower().split('/')
    # Django test files are in tests/ directory or named test_*.py
    if 'tests' in parts or 'test' in parts or any(p.startswith('test_') for p in parts):
        return 'test'
    return 'src'


def get_commits(n):
    """Get last n commits with hash, message, date, parent."""
    fmt = "%H|%P|%aI|%s"
    out, _ = git("log", f"--format={fmt}", f"-{n}", "--first-parent")
    commits = []
    for line in out.strip().split('\n'):
        if not line.strip():
            continue
        parts = line.split('|', 3)
        if len(parts) < 4:
            continue
        hash_, parents, date, message = parts
        parent = parents.split()[0] if parents.strip() else None
        commits.append({
            'hash': hash_,
            'parent': parent,
            'date': date,
            'message': message
        })
    return commits


def get_changed_files(commit_hash, parent_hash):
    """Get list of changed files with their status."""
    if not parent_hash:
        return []
    out, rc = git("diff", "--name-status", f"{parent_hash}..{commit_hash}")
    if rc != 0:
        return []
    files = []
    for line in out.strip().split('\n'):
        if not line.strip():
            continue
        parts = line.split('\t')
        if len(parts) >= 2:
            status = parts[0][0]  # M, A, D, R, etc.
            filepath = parts[-1]  # For renames, take the new name
            files.append({'path': filepath, 'status': status, 'type': classify_file(filepath)})
    return files


def get_file_diff(commit_hash, parent_hash, filepath):
    """Get the diff for a specific file."""
    out, _ = git("diff", f"{parent_hash}..{commit_hash}", "--", filepath)
    return out


def get_file_content_before(parent_hash, filepath):
    """Get file content before the change."""
    out, rc = git("show", f"{parent_hash}:{filepath}")
    if rc != 0:
        return None
    return out


def get_directory_tree(commit_hash, max_dirs=50):
    """Get directory structure (not full file listing) for context."""
    out, rc = git("ls-tree", "-r", "--name-only", commit_hash)
    if rc != 0:
        return ""
    # Extract unique directory paths for manageable size
    dirs = set()
    for line in out.strip().split('\n')[:5000]:
        if '/' in line:
            parts = line.split('/')
            for i in range(1, min(len(parts), 4)):
                dirs.add('/'.join(parts[:i]) + '/')
    sorted_dirs = sorted(dirs)[:max_dirs]
    return '\n'.join(sorted_dirs)


def get_context_around_changes(file_content, diff_text, context_lines=10):
    """Extract the region of the file around the changed lines."""
    if not file_content or not diff_text:
        return file_content[:2000] if file_content else ""

    # Parse diff to find changed line numbers
    changed_lines = set()
    for line in diff_text.split('\n'):
        match = re.match(r'^@@ -(\d+)(?:,(\d+))? ', line)
        if match:
            start = int(match.group(1))
            count = int(match.group(2) or 1)
            for i in range(max(1, start - context_lines), start + count + context_lines):
                changed_lines.add(i)

    if not changed_lines:
        return file_content[:2000]

    lines = file_content.split('\n')
    min_line = max(0, min(changed_lines) - 1)
    max_line = min(len(lines), max(changed_lines))

    result_lines = []
    for i in range(min_line, max_line):
        result_lines.append(f"{i+1}: {lines[i]}")

    result = '\n'.join(result_lines)
    return result[:3000]


def quality_filter(commit, files):
    """Apply quality filters. Returns True if commit passes."""
    # Must have bug-related keyword
    if not BUG_KEYWORDS.search(commit['message']):
        return False

    # Only .py files
    py_files = [f for f in files if f['path'].endswith('.py')]
    if len(py_files) != len(files):
        return False

    # Has both src and test changes
    src_files = [f for f in files if f['type'] == 'src']
    test_files = [f for f in files if f['type'] == 'test']
    if not src_files or not test_files:
        return False

    # ≤5 src files changed
    if len(src_files) > MAX_SRC_FILES:
        return False

    return True


def generate_localization_pair(commit, files, tree):
    """Task: Given message + file tree, predict changed files."""
    src_files = [f['path'] for f in files if f['type'] == 'src']
    test_files = [f['path'] for f in files if f['type'] == 'test']

    input_text = f"""## Task: Localization
Given the following bug report/commit message and repository structure, identify which files need to be modified.

### Commit Message
{commit['message']}

### Repository Structure (top-level directories)
{tree}

### Question
Which source files and test files need to be modified to fix this issue?
"""

    output_text = f"""### Files to Modify

**Source files:**
{chr(10).join('- ' + f for f in src_files)}

**Test files:**
{chr(10).join('- ' + f for f in test_files)}
"""
    return {"task": "localization", "input": input_text, "output": output_text,
            "commit": commit['hash'], "message": commit['message']}


def generate_edit_pair(commit, files, parent_hash):
    """Task: Given message + pre-change content, predict patch."""
    src_files = [f for f in files if f['type'] == 'src']
    if not src_files:
        return None

    target_file = src_files[0]
    file_content = get_file_content_before(parent_hash, target_file['path'])
    diff = get_file_diff(commit['hash'], parent_hash, target_file['path'])

    if not file_content or not diff:
        return None

    context = get_context_around_changes(file_content, diff)

    input_text = f"""## Task: Code Edit
Apply the fix described in the commit message to the code below.

### Commit Message
{commit['message']}

### File: {target_file['path']}
```python
{context}
```

### Question
What changes need to be made to fix this issue? Provide the patch.
"""

    output_text = f"""### Patch for {target_file['path']}
```diff
{diff[:3000]}
```
"""
    return {"task": "edit", "input": input_text, "output": output_text,
            "commit": commit['hash'], "message": commit['message']}


def generate_test_writing_pair(commit, files, parent_hash):
    """Task: Given message + src patch, predict test patch."""
    src_files = [f for f in files if f['type'] == 'src']
    test_files = [f for f in files if f['type'] == 'test']

    if not src_files or not test_files:
        return None

    src_diff = get_file_diff(commit['hash'], parent_hash, src_files[0]['path'])
    test_diff = get_file_diff(commit['hash'], parent_hash, test_files[0]['path'])

    if not src_diff or not test_diff:
        return None

    input_text = f"""## Task: Test Writing
Given the following bug fix, write appropriate test cases.

### Commit Message
{commit['message']}

### Source Code Patch
```diff
{src_diff[:3000]}
```

### Question
Write test cases that verify this fix works correctly and prevents regression.
"""

    output_text = f"""### Test Patch ({test_files[0]['path']})
```diff
{test_diff[:3000]}
```
"""
    return {"task": "test_writing", "input": input_text, "output": output_text,
            "commit": commit['hash'], "message": commit['message']}


def generate_commit_message_pair(commit, files, parent_hash):
    """Task: Given full diff, predict commit message."""
    # Collect all diffs
    all_diffs = []
    for f in files[:3]:  # Limit to first 3 files for manageable size
        diff = get_file_diff(commit['hash'], parent_hash, f['path'])
        if diff:
            all_diffs.append(diff[:2000])

    if not all_diffs:
        return None

    combined_diff = '\n'.join(all_diffs)

    input_text = f"""## Task: Commit Message Generation
Given the following code changes, write an appropriate commit message.

### Changed Files
{chr(10).join('- ' + f['path'] + ' (' + f['status'] + ')' for f in files)}

### Diff
```diff
{combined_diff[:5000]}
```

### Question
Write a concise, descriptive commit message for these changes.
"""

    output_text = f"""{commit['message']}"""

    return {"task": "commit_message", "input": input_text, "output": output_text,
            "commit": commit['hash'], "message": commit['message']}


def generate_bug_detection_pair(commit, files, parent_hash):
    """Task: Given buggy code region, predict bug description."""
    src_files = [f for f in files if f['type'] == 'src']
    if not src_files:
        return None

    target_file = src_files[0]
    file_content = get_file_content_before(parent_hash, target_file['path'])
    diff = get_file_diff(commit['hash'], parent_hash, target_file['path'])

    if not file_content or not diff:
        return None

    context = get_context_around_changes(file_content, diff, context_lines=15)

    input_text = f"""## Task: Bug Detection
Analyze the following code and identify the bug.

### File: {target_file['path']}
```python
{context}
```

### Question
What bug exists in this code? Describe the issue and how it should be fixed.
"""

    # Extract what was changed from the diff to describe the bug
    removed_lines = [l[1:] for l in diff.split('\n') if l.startswith('-') and not l.startswith('---')]
    added_lines = [l[1:] for l in diff.split('\n') if l.startswith('+') and not l.startswith('+++')]

    output_text = f"""### Bug Description
**Commit:** {commit['message']}

**Problem:** The code has a bug that was fixed in commit {commit['hash'][:8]}.

**Buggy code (lines removed):**
```python
{chr(10).join(removed_lines[:10])}
```

**Fixed code (lines added):**
```python
{chr(10).join(added_lines[:10])}
```

**Explanation:** {commit['message']}
"""
    return {"task": "bug_detection", "input": input_text, "output": output_text,
            "commit": commit['hash'], "message": commit['message']}


def main():
    print("=" * 70)
    print("PILOT ANALYSIS: Django Repository - Mid-Training Data Synthesis")
    print("=" * 70)

    # Step 1: Get commits
    print(f"\n[1/5] Extracting last {NUM_COMMITS} commits...")
    commits = get_commits(NUM_COMMITS)
    print(f"  Got {len(commits)} commits")
    print(f"  Date range: {commits[-1]['date'][:10]} to {commits[0]['date'][:10]}")

    # Step 2: Extract changed files and apply filter
    print(f"\n[2/5] Analyzing changed files and applying quality filters...")
    filtered_commits = []
    stats = defaultdict(int)

    for i, commit in enumerate(commits):
        if i % 200 == 0:
            print(f"  Processing commit {i}/{len(commits)}...")

        if not commit['parent']:
            stats['no_parent'] += 1
            continue

        files = get_changed_files(commit['hash'], commit['parent'])
        if not files:
            stats['no_files'] += 1
            continue

        # Check total patch length
        total_diff, _ = git("diff", "--stat", f"{commit['parent']}..{commit['hash']}")
        full_diff, _ = git("diff", f"{commit['parent']}..{commit['hash']}")
        if len(full_diff) > MAX_PATCH_LEN:
            stats['too_long'] += 1
            continue

        if not quality_filter(commit, files):
            if not BUG_KEYWORDS.search(commit['message']):
                stats['no_keyword'] += 1
            elif any(not f['path'].endswith('.py') for f in files):
                stats['non_py'] += 1
            elif not [f for f in files if f['type'] == 'src']:
                stats['no_src'] += 1
            elif not [f for f in files if f['type'] == 'test']:
                stats['no_test'] += 1
            elif len([f for f in files if f['type'] == 'src']) > MAX_SRC_FILES:
                stats['too_many_src'] += 1
            else:
                stats['other_filter'] += 1
            continue

        filtered_commits.append({
            'commit': commit,
            'files': files,
            'diff_len': len(full_diff)
        })

    print(f"\n  Filter results:")
    print(f"    Total commits analyzed: {len(commits)}")
    print(f"    Passed quality filter: {len(filtered_commits)}")
    print(f"    Rejected - no bug keyword: {stats['no_keyword']}")
    print(f"    Rejected - non-.py files: {stats['non_py']}")
    print(f"    Rejected - no src changes: {stats['no_src']}")
    print(f"    Rejected - no test changes: {stats['no_test']}")
    print(f"    Rejected - too many src files: {stats['too_many_src']}")
    print(f"    Rejected - patch too long: {stats['too_long']}")
    print(f"    Rejected - no parent/files: {stats['no_parent'] + stats['no_files']}")

    # Step 3: Generate QA pairs
    print(f"\n[3/5] Generating QA pairs ({SAMPLES_PER_TASK} per task type)...")

    # Get directory tree once (from HEAD)
    tree = get_directory_tree("HEAD")

    task_generators = {
        'localization': generate_localization_pair,
        'edit': generate_edit_pair,
        'test_writing': generate_test_writing_pair,
        'commit_message': generate_commit_message_pair,
        'bug_detection': generate_bug_detection_pair,
    }

    all_examples = []
    task_counts = defaultdict(int)

    for item in filtered_commits:
        commit = item['commit']
        files = item['files']
        parent = commit['parent']

        for task_name, generator in task_generators.items():
            if task_counts[task_name] >= SAMPLES_PER_TASK:
                continue

            try:
                if task_name == 'localization':
                    pair = generator(commit, files, tree)
                else:
                    pair = generator(commit, files, parent)
            except Exception as e:
                continue

            if pair:
                all_examples.append(pair)
                task_counts[task_name] += 1

        # Stop if we have enough for all tasks
        if all(c >= SAMPLES_PER_TASK for c in task_counts.values()):
            break

    print(f"  Generated examples per task:")
    for task, count in sorted(task_counts.items()):
        print(f"    {task}: {count}")

    # Step 4: Save results
    print(f"\n[4/5] Saving results to {OUTPUT_DIR}/pilot_examples.jsonl")
    output_path = os.path.join(OUTPUT_DIR, "pilot_examples.jsonl")
    with open(output_path, 'w') as f:
        for example in all_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    print(f"  Saved {len(all_examples)} examples")

    # Step 5: Print statistics and sample examples
    print(f"\n[5/5] Summary Statistics")
    print("=" * 70)
    print(f"  Repository: Django")
    print(f"  Commits analyzed: {len(commits)}")
    print(f"  Commits passing filter: {len(filtered_commits)} ({100*len(filtered_commits)/len(commits):.1f}%)")
    print(f"  Total QA pairs generated: {len(all_examples)}")
    print(f"  Output file: {output_path}")

    # Print distribution of filtered commits by year-month
    print(f"\n  Filtered commits by month:")
    monthly = defaultdict(int)
    for item in filtered_commits:
        ym = item['commit']['date'][:7]
        monthly[ym] += 1
    for ym in sorted(monthly.keys(), reverse=True)[:10]:
        print(f"    {ym}: {monthly[ym]}")

    # Print some file type stats
    print(f"\n  File change distribution (filtered commits):")
    src_count = sum(len([f for f in item['files'] if f['type'] == 'src']) for item in filtered_commits)
    test_count = sum(len([f for f in item['files'] if f['type'] == 'test']) for item in filtered_commits)
    print(f"    Total src file changes: {src_count}")
    print(f"    Total test file changes: {test_count}")
    print(f"    Avg src files per commit: {src_count/max(1,len(filtered_commits)):.1f}")
    print(f"    Avg test files per commit: {test_count/max(1,len(filtered_commits)):.1f}")

    # Print sample QA pairs
    print("\n" + "=" * 70)
    print("SAMPLE QA PAIRS (one per task type)")
    print("=" * 70)

    shown_tasks = set()
    for example in all_examples:
        if example['task'] not in shown_tasks:
            shown_tasks.add(example['task'])
            print(f"\n{'─' * 70}")
            print(f"TASK: {example['task'].upper()}")
            print(f"COMMIT: {example['commit'][:12]} - {example['message']}")
            print(f"{'─' * 70}")
            print(f"\n--- INPUT ---")
            print(example['input'][:2000])
            print(f"\n--- OUTPUT ---")
            print(example['output'][:2000])
            print()


if __name__ == "__main__":
    main()
