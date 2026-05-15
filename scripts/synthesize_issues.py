#!/usr/bin/env python3
"""
Synthesize high-quality GitHub issue descriptions from commit data using Claude Haiku
via the gateway at http://106.54.223.20:8000.

Three prompt strategies (inspired by R2E-Gym's build_syn_issue.py):
  A) Full context: commit msg + src patch + test patch
  B) Minimal context: commit msg + test patch only (no src patch => avoids leaking fix)
  C) Bug-focused: commit msg + buggy code region (before fix)

Outputs:
  - data/synthesis_exploration/strategy_a.jsonl
  - data/synthesis_exploration/strategy_b.jsonl
  - data/synthesis_exploration/strategy_c.jsonl
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

# ─── Config ────────────────────────────────────────────────────────────────────
GATEWAY_URL = "http://106.54.223.20:8000"
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = "claude-haiku-4.5"
REPO_DIR = "/data_fast_v3/eremite/cogito_explore/AgentMidtrain/repos/django_django"
OUTPUT_DIR = "/data_fast_v3/eremite/cogito_explore/AgentMidtrain/data/synthesis_exploration"

# 10 selected commits with clean, meaningful bug fixes from django/django
COMMITS = [
    {
        "hash": "335c6d0129",
        "message": "Fixed #37095 -- Checked maximum redirect lengths against percent-encoded URLs.",
        "src_files": ["django/http/response.py"],
        "test_files": ["tests/httpwrappers/tests.py"],
    },
    {
        "hash": "62fa9b8976",
        "message": "Fixed #37084 -- Added CSP nonce context processor system check.",
        "src_files": ["django/core/checks/security/base.py"],
        "test_files": ["tests/check_framework/test_security.py"],
    },
    {
        "hash": "21c51c2623",
        "message": "Fixed #37060 -- Propagated AlterField through attname-based to_field references.",
        "src_files": ["django/db/backends/base/schema.py", "django/db/backends/sqlite3/schema.py"],
        "test_files": ["tests/migrations/test_operations.py"],
    },
    {
        "hash": "1085e5e17b",
        "message": "Fixed #36300 -- Restored the semantic where RemoteUserMiddleware.header corresponds to request.META under ASGI.",
        "src_files": ["django/contrib/auth/middleware.py"],
        "test_files": ["tests/auth_tests/test_remote_user.py"],
    },
    {
        "hash": "ed54863769",
        "message": "Fixed #37092, Refs #35870 -- Added missing deprecation warnings for USE_BLANK_CHOICE_DASH.",
        "src_files": ["django/conf/__init__.py"],
        "test_files": ["tests/deprecation/test_use_blank_choice_dash.py"],
    },
    {
        "hash": "8096b52510",
        "message": "Fixed #37085 -- Added support for object-based form media stylesheet assets.",
        "src_files": ["django/forms/widgets.py"],
        "test_files": ["tests/forms_tests/tests/test_media.py"],
    },
    {
        "hash": "a284a49153",
        "message": "Fixed #37047 -- Fixed crash in Query.orderby_issubset_groupby for descending and random order_by strings.",
        "src_files": ["django/db/models/sql/query.py"],
        "test_files": ["tests/aggregation_regress/tests.py"],
    },
    {
        "hash": "386257b33e",
        "message": "Refs #36494 -- Prevented crash in JSONField numeric lookups with expressions.",
        "src_files": ["django/db/models/fields/json.py"],
        "test_files": ["tests/model_fields/test_jsonfield.py"],
    },
    {
        "hash": "8b7ea2bcdd",
        "message": "Refs #36913 -- Maintained error message determinism in MultipleChoiceField.validate().",
        "src_files": ["django/forms/fields.py"],
        "test_files": ["tests/forms_tests/field_tests/test_multiplechoicefield.py"],
    },
    {
        "hash": "9c655e9800",
        "message": "Fixed #36938 -- Removed unnecessary ordering from compound queries.",
        "src_files": ["django/db/models/sql/compiler.py"],
        "test_files": [],  # no test file changes for this commit
    },
]


# ─── Git helpers ───────────────────────────────────────────────────────────────

def run_git(args: list[str], cwd: str = REPO_DIR) -> str:
    """Run a git command and return stdout."""
    result = subprocess.run(
        ["git"] + args, capture_output=True, text=True, cwd=cwd
    )
    if result.returncode != 0:
        print(f"  [git error] {result.stderr.strip()}", file=sys.stderr)
    return result.stdout.strip()


def get_src_patch(commit_hash: str, src_files: list[str]) -> str:
    """Get the source-only diff for a commit."""
    parent = run_git(["rev-parse", f"{commit_hash}^"])
    patches = []
    for f in src_files:
        diff = run_git(["diff", f"{parent}..{commit_hash}", "--", f])
        if diff:
            patches.append(diff)
    return "\n".join(patches)


def get_test_patch(commit_hash: str, test_files: list[str]) -> str:
    """Get the test-only diff for a commit."""
    parent = run_git(["rev-parse", f"{commit_hash}^"])
    patches = []
    for f in test_files:
        diff = run_git(["diff", f"{parent}..{commit_hash}", "--", f])
        if diff:
            patches.append(diff)
    return "\n".join(patches)


def get_buggy_code_region(commit_hash: str, src_files: list[str],
                          context_lines: int = 25) -> str:
    """Extract buggy code region (before fix) with context around changed lines."""
    parent = run_git(["rev-parse", f"{commit_hash}^"])
    regions = []

    for src_file in src_files:
        buggy_content = run_git(["show", f"{parent}:{src_file}"])
        if not buggy_content:
            continue

        # Get diff to find changed line numbers
        diff_output = run_git(["diff", "-U0", f"{parent}..{commit_hash}", "--", src_file])

        changed_lines = set()
        for line in diff_output.split("\n"):
            if line.startswith("@@"):
                parts = line.split()
                old_range = parts[1]  # e.g., -642,12
                start_str = old_range.split(",")[0].replace("-", "")
                start = int(start_str)
                count = int(old_range.split(",")[1]) if "," in old_range else 1
                for ln in range(start, start + count):
                    changed_lines.add(ln)

        if not changed_lines:
            # Fallback: return first 60 lines
            regions.append(f"# {src_file}\n" + "\n".join(buggy_content.split("\n")[:60]))
            continue

        lines = buggy_content.split("\n")
        min_line = max(0, min(changed_lines) - context_lines - 1)
        max_line = min(len(lines), max(changed_lines) + context_lines)

        numbered = [f"# {src_file} (lines {min_line+1}-{max_line})"]
        for i, line in enumerate(lines[min_line:max_line], start=min_line + 1):
            marker = ">>> " if i in changed_lines else "    "
            numbered.append(f"{marker}{i:4d} | {line}")

        regions.append("\n".join(numbered))

    return "\n\n".join(regions)


# ─── LLM caller ───────────────────────────────────────────────────────────────

_active_model = MODEL  # will be updated by main() after connectivity test


def call_llm(prompt: str, temperature: float = 0.3, max_tokens: int = 1024) -> str:
    """Call the LLM gateway."""
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    resp = requests.post(
        f"{GATEWAY_URL}/v1/chat/completions",
        json={
            "model": _active_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        headers=headers,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ─── Prompt strategies ─────────────────────────────────────────────────────────

def strategy_a_full_context(commit_msg: str, src_patch: str, test_patch: str) -> str:
    """Strategy A: Full context (commit msg + src patch + test patch)."""
    prompt = f"""You are creating a GitHub issue based on a real bug fix commit in the Django web framework.

Commit message: {commit_msg}

Source patch (the fix):
```diff
{src_patch[:4000]}
```

Test patch (tests added/changed):
```diff
{test_patch[:4000]}
```

Write a natural, concise GitHub issue that:
1. Describes the bug from a user's perspective (not the fix)
2. Includes a minimal code example or scenario showing the problem
3. Shows expected vs actual behavior
4. Does NOT reveal the solution or mention how to fix it
5. Is 5-15 lines long

Output the issue in [ISSUE]...[/ISSUE] tags."""
    return call_llm(prompt)


def strategy_b_minimal_context(commit_msg: str, test_patch: str) -> str:
    """Strategy B: Minimal context (only commit msg + test patch, no src patch).
    Harder but avoids leaking the fix."""
    prompt = f"""You are creating a GitHub issue for the Django web framework. You know about a bug from its test case, but you should NOT reveal implementation details.

Commit message: {commit_msg}

Test patch (tests that expose the bug):
```diff
{test_patch[:4000]}
```

Based on what the tests check, write a GitHub issue from a user's perspective that:
1. Describes the symptoms of the bug without referencing internal code
2. Includes a minimal reproduction scenario (code example or steps)
3. Shows expected vs actual behavior
4. Does NOT mention internal function names, file paths, or the fix
5. Is 5-15 lines long

Output the issue in [ISSUE]...[/ISSUE] tags."""
    return call_llm(prompt)


def strategy_c_bug_focused(commit_msg: str, buggy_code: str) -> str:
    """Strategy C: Bug-focused (commit msg + buggy code region before fix)."""
    prompt = f"""You are creating a GitHub issue based on buggy code in the Django web framework. The lines marked with >>> contain the bug.

Commit message (describes the fix): {commit_msg}

Buggy code (BEFORE the fix was applied):
```python
{buggy_code[:5000]}
```

Write a natural, concise GitHub issue that:
1. Describes the problem a user would encounter due to this bug
2. Includes a concrete example or scenario showing the failure
3. Shows expected vs actual behavior
4. Does NOT describe the fix or point to specific buggy lines
5. Is 5-15 lines long

Output the issue in [ISSUE]...[/ISSUE] tags."""
    return call_llm(prompt)


def extract_issue(text: str) -> str:
    """Extract content between [ISSUE] and [/ISSUE] tags."""
    if "[ISSUE]" in text and "[/ISSUE]" in text:
        start = text.index("[ISSUE]") + len("[ISSUE]")
        end = text.index("[/ISSUE]")
        return text[start:end].strip()
    return text.strip()


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Test gateway connectivity
    print("=" * 80)
    print("STEP 1: Testing gateway connectivity")
    print("=" * 80)
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    # Try primary model and fallbacks
    active_model = MODEL
    models_to_try = [MODEL, "gpt-4o-mini", "gpt-5-nano", "claude-sonnet-4.5"]
    connected = False
    for try_model in models_to_try:
        try:
            resp = requests.post(
                f"{GATEWAY_URL}/v1/chat/completions",
                json={
                    "model": try_model,
                    "messages": [{"role": "user", "content": "Hello, respond with just 'OK'"}],
                    "max_tokens": 10,
                },
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            print(f"  Gateway OK! Model={try_model}, Response={content!r}")
            active_model = try_model
            connected = True
            break
        except Exception as e:
            print(f"  Model {try_model} failed: {e}")

    if not connected:
        print("  All models failed. Exiting.")
        sys.exit(1)

    # Update the global active model
    global _active_model
    _active_model = active_model

    # Step 2: Extract commit data
    print(f"\n{'='*80}")
    print(f"STEP 2: Extracting commit data for {len(COMMITS)} commits")
    print("=" * 80)

    commit_data = []
    for c in COMMITS:
        print(f"  Extracting {c['hash'][:10]}... ", end="", flush=True)
        src_patch = get_src_patch(c["hash"], c["src_files"])
        test_patch = get_test_patch(c["hash"], c["test_files"]) if c["test_files"] else ""
        buggy_code = get_buggy_code_region(c["hash"], c["src_files"])
        print(f"src={len(src_patch)}B, test={len(test_patch)}B, buggy={len(buggy_code)}B")
        commit_data.append({
            "hash": c["hash"],
            "message": c["message"],
            "src_files": c["src_files"],
            "test_files": c["test_files"],
            "src_patch": src_patch,
            "test_patch": test_patch,
            "buggy_code": buggy_code,
        })

    # Step 3: Run all 3 strategies
    results_a = []
    results_b = []
    results_c = []

    print(f"\n{'='*80}")
    print("STEP 3: Synthesizing issues with 3 strategies")
    print(f"  Model: {active_model}")
    print("=" * 80)

    for i, cd in enumerate(commit_data):
        print(f"\n{'─'*70}")
        print(f"[{i+1}/{len(commit_data)}] {cd['hash'][:10]} - {cd['message'][:65]}")
        print(f"{'─'*70}")

        # Strategy A: Full context
        print(f"\n  [Strategy A] Full context...", end=" ", flush=True)
        t0 = time.time()
        try:
            raw_a = strategy_a_full_context(cd["message"], cd["src_patch"], cd["test_patch"])
            issue_a = extract_issue(raw_a)
        except Exception as e:
            issue_a = f"[ERROR] {e}"
            raw_a = issue_a
        t1 = time.time()
        print(f"({t1-t0:.1f}s, {len(issue_a)} chars)")
        results_a.append({
            "commit_hash": cd["hash"],
            "commit_message": cd["message"],
            "src_files": cd["src_files"],
            "test_files": cd["test_files"],
            "strategy": "A_full_context",
            "issue": issue_a,
            "raw_response": raw_a,
            "model": active_model,
            "latency_s": round(t1 - t0, 2),
        })

        # Strategy B: Minimal context (no src patch)
        if cd["test_patch"]:
            print(f"  [Strategy B] Minimal context (test-only)...", end=" ", flush=True)
            t0 = time.time()
            try:
                raw_b = strategy_b_minimal_context(cd["message"], cd["test_patch"])
                issue_b = extract_issue(raw_b)
            except Exception as e:
                issue_b = f"[ERROR] {e}"
                raw_b = issue_b
            t1 = time.time()
            print(f"({t1-t0:.1f}s, {len(issue_b)} chars)")
        else:
            issue_b = "[SKIPPED] No test patch available for this commit."
            raw_b = issue_b
            print(f"  [Strategy B] SKIPPED (no test patch)")

        results_b.append({
            "commit_hash": cd["hash"],
            "commit_message": cd["message"],
            "src_files": cd["src_files"],
            "test_files": cd["test_files"],
            "strategy": "B_minimal_context",
            "issue": issue_b,
            "raw_response": raw_b,
            "model": active_model,
            "latency_s": round(t1 - t0, 2) if cd["test_patch"] else 0,
        })

        # Strategy C: Bug-focused (buggy code)
        print(f"  [Strategy C] Bug-focused (pre-fix code)...", end=" ", flush=True)
        t0 = time.time()
        try:
            raw_c = strategy_c_bug_focused(cd["message"], cd["buggy_code"])
            issue_c = extract_issue(raw_c)
        except Exception as e:
            issue_c = f"[ERROR] {e}"
            raw_c = issue_c
        t1 = time.time()
        print(f"({t1-t0:.1f}s, {len(issue_c)} chars)")
        results_c.append({
            "commit_hash": cd["hash"],
            "commit_message": cd["message"],
            "src_files": cd["src_files"],
            "test_files": cd["test_files"],
            "strategy": "C_bug_focused",
            "issue": issue_c,
            "raw_response": raw_c,
            "model": active_model,
            "latency_s": round(t1 - t0, 2),
        })

    # Step 4: Save results
    print(f"\n{'='*80}")
    print("STEP 4: Saving results")
    print("=" * 80)

    for filename, results in [
        ("strategy_a.jsonl", results_a),
        ("strategy_b.jsonl", results_b),
        ("strategy_c.jsonl", results_c),
    ]:
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "w") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  Saved {len(results)} entries -> {path}")

    # Step 5: Quality comparison (3 commits side-by-side)
    print(f"\n{'='*80}")
    print("STEP 5: QUALITY COMPARISON - 3 commits x 3 strategies")
    print("=" * 80)

    comparison_commits = [0, 1, 3]  # indices: redirect, CSP, ASGI middleware
    for idx in comparison_commits:
        cd = commit_data[idx]
        print(f"\n{'#'*70}")
        print(f"COMMIT: {cd['hash'][:10]}")
        print(f"MESSAGE: {cd['message']}")
        print(f"FILES: {', '.join(cd['src_files'])}")
        print(f"{'#'*70}")

        print(f"\n  ┌─ Strategy A: Full Context {'─'*40}")
        for line in results_a[idx]["issue"].split("\n"):
            print(f"  │ {line}")
        print(f"  └{'─'*60}")

        print(f"\n  ┌─ Strategy B: Minimal Context (test-only) {'─'*25}")
        for line in results_b[idx]["issue"].split("\n"):
            print(f"  │ {line}")
        print(f"  └{'─'*60}")

        print(f"\n  ┌─ Strategy C: Bug-Focused (pre-fix code) {'─'*25}")
        for line in results_c[idx]["issue"].split("\n"):
            print(f"  │ {line}")
        print(f"  └{'─'*60}")

    # Step 6: Summary stats
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print("=" * 80)

    for label, results in [("A (full)", results_a), ("B (minimal)", results_b), ("C (bug-focused)", results_c)]:
        valid = [r for r in results if not r["issue"].startswith("[ERROR]") and not r["issue"].startswith("[SKIPPED]")]
        lengths = [len(r["issue"]) for r in valid]
        latencies = [r["latency_s"] for r in valid]
        line_counts = [len(r["issue"].split("\n")) for r in valid]
        if valid:
            print(f"\n  Strategy {label}:")
            print(f"    Successful: {len(valid)}/{len(results)}")
            print(f"    Avg length: {sum(lengths)/len(lengths):.0f} chars")
            print(f"    Avg lines:  {sum(line_counts)/len(line_counts):.1f}")
            print(f"    Avg latency: {sum(latencies)/len(latencies):.1f}s")
        else:
            print(f"\n  Strategy {label}: no valid results")

    print(f"\nDone! Results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
