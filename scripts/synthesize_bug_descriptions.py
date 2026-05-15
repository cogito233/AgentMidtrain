"""
Synthesize better bug descriptions using LLM for bug_detection task data.

This script explores 3 strategies for generating natural language bug descriptions
from commit diffs, replacing the low-quality mechanical diff pasting approach.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

# Config
GATEWAY_URL = "http://106.54.223.20:8000"
MODEL = "claude-sonnet-4.5"  # Good balance of quality and cost
REPO_DIR = "/data_fast_v3/eremite/cogito_explore/AgentMidtrain/repos/django_django"
OUTPUT_DIR = "/data_fast_v3/eremite/cogito_explore/AgentMidtrain/data/synthesis_exploration/bug_descriptions"

# 5 selected commits with clean, meaningful bug fixes
COMMITS = [
    {
        "hash": "335c6d0129",
        "message": "Fixed #37095 -- Checked maximum redirect lengths against percent-encoded URLs.",
        "source_file": "django/http/response.py",
    },
    {
        "hash": "21c51c2623",
        "message": "Fixed #37060 -- Propagated AlterField through attname-based to_field references.",
        "source_file": "django/db/backends/base/schema.py",
    },
    {
        "hash": "1085e5e17b",
        "message": "Fixed #36300 -- Restored the semantic where RemoteUserMiddleware.header corresponds to request.META under ASGI.",
        "source_file": "django/contrib/auth/middleware.py",
    },
    {
        "hash": "c79bdfc135",
        "message": "Fixed CVE-2026-6907 -- Prevented caching of requests when Vary header contains an asterisk.",
        "source_file": "django/middleware/cache.py",
    },
    {
        "hash": "8096b52510",
        "message": "Fixed #37085 -- Added support for object-based form media stylesheet assets.",
        "source_file": "django/forms/widgets.py",
    },
]


def run_git(args, cwd=REPO_DIR):
    """Run a git command and return output."""
    result = subprocess.run(
        ["git"] + args, capture_output=True, text=True, cwd=cwd
    )
    if result.returncode != 0:
        print(f"  [git error] {result.stderr.strip()}", file=sys.stderr)
    return result.stdout.strip()


def get_buggy_code_region(commit_hash, source_file, context_lines=30):
    """Extract the buggy code region (before fix) with context around changed lines."""
    # Get the parent commit
    parent = run_git(["rev-parse", f"{commit_hash}^"])

    # Get the full file content at parent (buggy version)
    buggy_content = run_git(["show", f"{parent}:{source_file}"])

    # Get the diff to find changed line numbers
    diff_output = run_git(["diff", "-U0", f"{parent}..{commit_hash}", "--", source_file])

    # Parse hunk headers to find changed lines in the old file
    changed_lines = []
    for line in diff_output.split("\n"):
        if line.startswith("@@"):
            # Parse @@ -start,count +start,count @@
            parts = line.split()
            old_range = parts[1]  # e.g., -642,12
            start = int(old_range.split(",")[0].replace("-", ""))
            count = int(old_range.split(",")[1]) if "," in old_range else 1
            changed_lines.extend(range(start, start + count))

    if not changed_lines:
        return buggy_content[:3000]  # fallback

    # Extract region with context
    lines = buggy_content.split("\n")
    min_line = max(0, min(changed_lines) - context_lines - 1)
    max_line = min(len(lines), max(changed_lines) + context_lines)

    region_lines = lines[min_line:max_line]
    # Add line numbers for clarity
    numbered = []
    for i, line in enumerate(region_lines, start=min_line + 1):
        marker = ">>>" if i in changed_lines else "   "
        numbered.append(f"{marker} {i:4d} | {line}")

    return "\n".join(numbered)


def get_diff_summary(commit_hash, source_file):
    """Get a brief summary of what the diff changes."""
    parent = run_git(["rev-parse", f"{commit_hash}^"])
    diff = run_git(["diff", f"{parent}..{commit_hash}", "--", source_file])
    return diff


def call_llm(prompt, temperature=0.3):
    """Call the LLM via gateway."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    resp = requests.post(
        f"{GATEWAY_URL}/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 1024,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def strategy1_root_cause(buggy_code, diff_summary, commit_msg):
    """Strategy 1: Root Cause Analysis"""
    prompt = f"""Given this buggy code (before fix):

```python
{buggy_code}
```

The fix was described as: {commit_msg}

Here is the actual diff:
```
{diff_summary}
```

Explain the root cause of this bug in 2-3 sentences. What would a developer notice as wrong behavior? What is the underlying logical error?

Be concise and specific. Do NOT mention the fix itself - only describe the bug."""
    return call_llm(prompt)


def strategy2_user_report(buggy_code, commit_msg):
    """Strategy 2: User-facing Bug Report"""
    prompt = f"""A bug was found in this Django code:

```python
{buggy_code}
```

The commit message says: {commit_msg}

Write a bug report from a user's perspective. Include:
- What they were trying to do
- What went wrong (symptoms)
- Expected vs actual behavior

Do NOT mention the fix or how to fix it. 3-5 sentences. Write as if you are a user who encountered this bug in production."""
    return call_llm(prompt)


def strategy3_code_review(buggy_code, commit_msg):
    """Strategy 3: Code Review Style"""
    prompt = f"""Review this Django code for potential bugs:

```python
{buggy_code}
```

Context: {commit_msg}

Identify the specific bug in the code marked with >>> markers. Explain why it's wrong, and suggest the correct approach. Be specific about which lines are problematic and what the correct logic should be.

Keep your response to 3-5 sentences. Be precise and technical."""
    return call_llm(prompt)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results_s1 = []
    results_s2 = []
    results_s3 = []

    print("=" * 80)
    print("BUG DESCRIPTION SYNTHESIS EXPLORATION")
    print(f"Model: {MODEL}")
    print(f"Commits: {len(COMMITS)} from django/django")
    print("=" * 80)

    for i, commit in enumerate(COMMITS):
        print(f"\n{'='*80}")
        print(f"[{i+1}/{len(COMMITS)}] Commit: {commit['hash']}")
        print(f"  Message: {commit['message']}")
        print(f"  File: {commit['source_file']}")
        print(f"{'='*80}")

        # Extract buggy code region
        buggy_code = get_buggy_code_region(commit["hash"], commit["source_file"])
        diff_summary = get_diff_summary(commit["hash"], commit["source_file"])

        print(f"\n  Buggy code region ({len(buggy_code)} chars)")

        # Strategy 1: Root Cause Analysis
        print("\n  --- Strategy 1: Root Cause Analysis ---")
        t0 = time.time()
        s1_result = strategy1_root_cause(buggy_code, diff_summary, commit["message"])
        t1 = time.time()
        print(f"  [{t1-t0:.1f}s] {s1_result}")
        results_s1.append({
            "commit_hash": commit["hash"],
            "commit_message": commit["message"],
            "source_file": commit["source_file"],
            "strategy": "root_cause",
            "description": s1_result,
            "model": MODEL,
        })

        # Strategy 2: User Report
        print("\n  --- Strategy 2: User-facing Bug Report ---")
        t0 = time.time()
        s2_result = strategy2_user_report(buggy_code, commit["message"])
        t1 = time.time()
        print(f"  [{t1-t0:.1f}s] {s2_result}")
        results_s2.append({
            "commit_hash": commit["hash"],
            "commit_message": commit["message"],
            "source_file": commit["source_file"],
            "strategy": "user_report",
            "description": s2_result,
            "model": MODEL,
        })

        # Strategy 3: Code Review
        print("\n  --- Strategy 3: Code Review Style ---")
        t0 = time.time()
        s3_result = strategy3_code_review(buggy_code, commit["message"])
        t1 = time.time()
        print(f"  [{t1-t0:.1f}s] {s3_result}")
        results_s3.append({
            "commit_hash": commit["hash"],
            "commit_message": commit["message"],
            "source_file": commit["source_file"],
            "strategy": "code_review",
            "description": s3_result,
            "model": MODEL,
        })

    # Save results
    for filename, results in [
        ("strategy1_root_cause.jsonl", results_s1),
        ("strategy2_user_report.jsonl", results_s2),
        ("strategy3_code_review.jsonl", results_s3),
    ]:
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "w") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"\nSaved {len(results)} entries to {path}")

    # Final comparison
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    for i, commit in enumerate(COMMITS):
        print(f"\n{'─'*60}")
        print(f"Commit: {commit['hash']} - {commit['message'][:60]}")
        print(f"{'─'*60}")
        print(f"\n  [S1 Root Cause]:")
        print(f"    {results_s1[i]['description'][:200]}...")
        print(f"\n  [S2 User Report]:")
        print(f"    {results_s2[i]['description'][:200]}...")
        print(f"\n  [S3 Code Review]:")
        print(f"    {results_s3[i]['description'][:200]}...")


if __name__ == "__main__":
    main()
