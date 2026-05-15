#!/usr/bin/env python3
"""Refine bug_detection samples using LLM to validate and rewrite Bug descriptions.

Takes raw bug_detection.jsonl from generate_tasks.py, and for each sample:
1. Sends the code + diff info to the LLM
2. Asks: "Is this bug identifiable from the code alone?"
3. If yes, generates a clear Bug description
4. Outputs only validated samples with high-quality descriptions

Usage:
    python scripts/refine_bug_detection.py \
        --input data/tasks/django_v4_bugdet/bug_detection.jsonl \
        --output data/tasks/django_v4_bugdet/bug_detection_refined.jsonl \
        --gateway-url http://106.54.223.20:8000 \
        --model claude-haiku-4
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

DEFAULT_GATEWAY_URL = "http://106.54.223.20:8000"
DEFAULT_MODEL = "claude-haiku-4"
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


VALIDATION_PROMPT = """\
You are a strict evaluator determining whether a code snippet contains a bug that is CLEARLY identifiable from reading the code alone.

## Code snippet (pre-fix version)
{code_snippet}

## Context about the change
The commit that modified this code says: "{commit_msg}"
Lines changed: {location}
Code that was modified: `{problematic_code}`

## Strict criteria - answer YES ONLY if ALL of these are true:

1. **Clearly wrong from code alone**: A skilled developer reading ONLY this snippet (without knowing it was "fixed" later) would identify something as clearly incorrect. NOT: subtle design choices, edge cases requiring system knowledge, or debatable style issues.

2. **Functional bug, not style/improvement**: The issue causes WRONG BEHAVIOR (crashes, wrong output, data corruption, security issue). NOT: non-determinism that doesn't affect correctness, missing features, defensive improvements, cosmetic issues, or debatable design choices.

3. **Verifiable from the snippet**: Someone can point to a specific line and explain WHY it's wrong using only the visible code. NOT: issues that require knowing how other parts of the system behave, what inputs are possible, or how the code is called.

4. **Not a false positive**: The "problematic code" is actually incorrect (not just unusual or suboptimal). Be skeptical - many commit "fixes" are actually enhancements, not bug fixes.

## If YES (strict pass), provide:
- **bug_description**: States the specific logical error (e.g., "The condition X should be Y because Z"). Verifiable by reading the code. Does NOT require external context to understand.
- **location**: The exact line number(s) where the bug is. Use the line numbers shown in the code snippet.
- **problematic_code**: The exact code expression/statement that is buggy (copy from the snippet).

## If NO (any doubt = NO), say SKIP.

Respond in exact JSON (no markdown fences):
{{"identifiable": true/false, "bug_description": "..." or null, "location_line": <int or null>, "problematic_code": "..." or null, "reasoning": "1 sentence"}}
"""


def call_llm(prompt: str, gateway_url: str, model: str,
             temperature: float = 0.2, max_tokens: int = 300) -> str:
    """Call the LLM gateway."""
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    resp = requests.post(
        f"{gateway_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        headers=headers,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def parse_llm_response(response: str) -> dict | None:
    """Parse JSON response from LLM."""
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return None


def refine_sample(sample: dict, gateway_url: str, model: str) -> dict | None:
    """Validate and refine a single bug_detection sample.

    Returns the refined sample if valid, None if should be skipped.
    """
    input_text = sample["input"]
    output_text = sample["output"]

    # Extract code snippet from input
    code_start = input_text.find("```python\n")
    code_end = input_text.rfind("```")
    if code_start < 0 or code_end < 0:
        return None
    code_snippet = input_text[code_start + len("```python\n"):code_end].strip()

    # Parse output fields
    lines = output_text.strip().split("\n")
    location = ""
    problematic_code = ""
    for line in lines:
        if line.startswith("Location:"):
            location = line[len("Location:"):].strip()
        elif line.startswith("Problematic code:"):
            problematic_code = line[len("Problematic code:"):].strip().strip("`")

    # Get commit message from metadata or reconstruct
    commit_msg = ""
    for line in lines:
        if line.startswith("Bug:"):
            commit_msg = line[len("Bug:"):].strip()
            break

    # Truncate code if too long
    if len(code_snippet) > 3000:
        code_snippet = code_snippet[:3000] + "\n... [truncated]"

    prompt = VALIDATION_PROMPT.format(
        code_snippet=code_snippet,
        commit_msg=commit_msg,
        location=location,
        problematic_code=problematic_code,
    )

    try:
        raw_response = call_llm(prompt, gateway_url, model)
        result = parse_llm_response(raw_response)
        if result is None:
            return None

        if not result.get("identifiable", False):
            return None

        bug_description = result.get("bug_description")
        if not bug_description or len(bug_description) < 10:
            return None

        # Use LLM-provided location and problematic_code if available
        llm_line = result.get("location_line")
        llm_problematic = result.get("problematic_code")

        # Extract filepath from original location
        filepath = ""
        if location and ":" in location:
            filepath = location.split(":")[0]

        # Build new output with LLM-generated fields
        new_output_parts = [f"Bug: {bug_description}"]

        # Location: use LLM line if provided, otherwise keep original
        if llm_line and filepath:
            new_output_parts.append(f"Location: {filepath}:{llm_line}")
        elif location:
            new_output_parts.append(f"Location: {location}")

        # Problematic code: use LLM's if provided, otherwise keep original
        if llm_problematic and len(llm_problematic) > 3:
            new_output_parts.append(f"Problematic code: `{llm_problematic}`")
        elif problematic_code:
            new_output_parts.append(f"Problematic code: `{problematic_code}`")

        refined_sample = dict(sample)
        refined_sample["output"] = "\n".join(new_output_parts)
        refined_sample["metadata"] = dict(sample.get("metadata", {}))
        refined_sample["metadata"]["llm_validated"] = True
        refined_sample["metadata"]["original_bug_desc"] = commit_msg
        return refined_sample

    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Refine bug_detection samples with LLM validation")
    parser.add_argument("--input", required=True, help="Input bug_detection.jsonl")
    parser.add_argument("--output", required=True, help="Output refined JSONL")
    parser.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=0, help="Max samples to process (0=all)")
    args = parser.parse_args()

    # Load samples
    samples = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    if args.limit > 0:
        samples = samples[:args.limit]

    print(f"Loaded {len(samples)} samples from {args.input}")
    print(f"Model: {args.model}, Gateway: {args.gateway_url}")
    print()

    # Test connectivity
    try:
        test_resp = call_llm("Say OK", args.gateway_url, args.model, max_tokens=10)
        print(f"Gateway OK: {test_resp!r}")
    except Exception as e:
        print(f"ERROR: Gateway failed: {e}")
        sys.exit(1)

    # Process samples
    refined = []
    skipped = 0
    errors = 0

    for i, sample in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] ", end="", flush=True)
        t0 = time.time()
        result = refine_sample(sample, args.gateway_url, args.model)
        elapsed = time.time() - t0

        if result is not None:
            refined.append(result)
            print(f"KEEP ({elapsed:.1f}s) - {result['output'].split(chr(10))[0][:80]}")
        else:
            skipped += 1
            print(f"SKIP ({elapsed:.1f}s)")

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for item in refined:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Input:    {len(samples)} samples")
    print(f"  Kept:     {len(refined)} ({len(refined)/len(samples)*100:.0f}%)")
    print(f"  Skipped:  {skipped}")
    print(f"  Errors:   {errors}")
    print(f"  Output:   {args.output}")


if __name__ == "__main__":
    main()
