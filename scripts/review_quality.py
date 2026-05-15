#!/usr/bin/env python3
"""
Review data quality across task types by randomly sampling tasks and
sending them to Claude Haiku via the gateway for structured evaluation.

Assesses: Coherence, Correctness, Leakage, Format, Difficulty, Overall Score (1-5).

Usage:
  python scripts/review_quality.py --samples-per-type 3
  python scripts/review_quality.py --task-types bug_detection,code_review --samples-per-type 5
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

# ─── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR = "/data_fast_v3/eremite/cogito_explore/AgentMidtrain/data/tasks"
DEFAULT_GATEWAY_URL = "http://106.54.223.20:8000"
DEFAULT_MODEL = "claude-sonnet-4.6"
DEFAULT_SAMPLES_PER_TYPE = 10
ALL_TASK_TYPES = [
    "bug_detection",
    "code_review",
    "commit_message",
    "edit_generation",
    "localization",
    "test_writing",
]

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


# ─── Review Prompt ────────────────────────────────────────────────────────────

REVIEW_PROMPT_TEMPLATE = """\
You are evaluating the quality of a training data sample for an AI coding assistant. The sample is of type "{task_type}".

## Task Description (System Prompt given to the model)
{prompt}

## Input (what the model receives)
{input}

## Expected Output (ground truth answer)
{output}

---

Evaluate this sample on 5 dimensions. For each, give a brief rationale (1 sentence) and a score from 1 (worst) to 5 (best).

1. **Coherence**: Does the input make sense as a task description? Is it well-formed and unambiguous?
2. **Correctness**: Is the output a valid/reasonable answer to the input? Is it factually correct?
3. **Leakage**: Does the input contain the answer or make it trivially obvious? (5 = no leakage, 1 = answer is fully given away)
4. **Format**: Is the output in the expected format for this task type?
5. **Difficulty**: Is the task at an appropriate difficulty level? (1 = trivial/impossible, 5 = well-calibrated challenge)

Then provide an **Overall** score (1-5) reflecting the sample's fitness for training.

Respond ONLY in this exact JSON format (no markdown fences, no extra text):
{{
  "coherence": {{"score": <1-5>, "rationale": "<1 sentence>"}},
  "correctness": {{"score": <1-5>, "rationale": "<1 sentence>"}},
  "leakage": {{"score": <1-5>, "rationale": "<1 sentence>"}},
  "format": {{"score": <1-5>, "rationale": "<1 sentence>"}},
  "difficulty": {{"score": <1-5>, "rationale": "<1 sentence>"}},
  "overall": {{"score": <1-5>, "rationale": "<1 sentence>"}}
}}"""


# ─── LLM caller ───────────────────────────────────────────────────────────────

def call_llm(prompt: str, gateway_url: str, model: str,
             temperature: float = 0.2, max_tokens: int = 512) -> str:
    """Call the LLM gateway (OpenAI-compatible chat completions)."""
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
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_samples(data_dir: str, task_type: str, n_samples: int) -> list[dict]:
    """Load random samples for a given task type from random repos."""
    data_path = Path(data_dir)
    # Find all repos that have this task type (subdirectory structure)
    repo_files = list(data_path.glob(f"*/{task_type}.jsonl"))
    # Also check for files directly in the data_dir (flat structure)
    direct_file = data_path / f"{task_type}.jsonl"
    if direct_file.exists() and direct_file not in repo_files:
        repo_files.append(direct_file)
    if not repo_files:
        print(f"  WARNING: No data found for task type '{task_type}'")
        return []

    # Collect all lines with repo info (lazy: read a random subset of repos)
    random.shuffle(repo_files)
    all_samples = []

    for repo_file in repo_files:
        repo_name = repo_file.parent.name
        with open(repo_file, "r") as f:
            lines = f.readlines()
        for line in lines:
            if line.strip():
                try:
                    item = json.loads(line)
                    item["_repo"] = repo_name
                    item["_source_file"] = str(repo_file)
                    all_samples.append(item)
                except json.JSONDecodeError:
                    continue
        # Early stop if we have way more than needed
        if len(all_samples) > n_samples * 20:
            break

    if not all_samples:
        return []

    # Random sample
    n = min(n_samples, len(all_samples))
    return random.sample(all_samples, n)


# ─── Review logic ─────────────────────────────────────────────────────────────

def parse_review(response: str) -> dict | None:
    """Parse JSON review response from LLM."""
    # Strip markdown fences if present
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON in the response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return None


def truncate(s: str, max_len: int = 3000) -> str:
    """Truncate string for prompt inclusion."""
    if len(s) <= max_len:
        return s
    return s[:max_len] + "\n... [truncated]"


def review_sample(sample: dict, gateway_url: str, model: str) -> dict:
    """Review a single sample and return structured result."""
    task_type = sample.get("task_type", "unknown")
    prompt_text = sample.get("prompt", "")
    input_text = sample.get("input", "")
    output_text = sample.get("output", "")

    review_prompt = REVIEW_PROMPT_TEMPLATE.format(
        task_type=task_type,
        prompt=truncate(prompt_text, 1000),
        input=truncate(input_text, 3000),
        output=truncate(output_text, 2000),
    )

    t0 = time.time()
    try:
        raw_response = call_llm(review_prompt, gateway_url, model)
        latency = time.time() - t0
        review = parse_review(raw_response)
        if review is None:
            return {
                "success": False,
                "error": "Failed to parse JSON response",
                "raw_response": raw_response,
                "latency_s": round(latency, 2),
            }
        return {
            "success": True,
            "review": review,
            "latency_s": round(latency, 2),
        }
    except Exception as e:
        latency = time.time() - t0
        return {
            "success": False,
            "error": str(e),
            "latency_s": round(latency, 2),
        }


# ─── Reporting ────────────────────────────────────────────────────────────────

def print_sample_review(idx: int, sample: dict, result: dict):
    """Print a single sample's review."""
    repo = sample.get("_repo", "?")
    commit = sample.get("commit", "?")[:10]
    task_type = sample.get("task_type", "?")

    print(f"\n  {'─'*60}")
    print(f"  Sample {idx+1} | repo={repo} | commit={commit} | type={task_type}")
    print(f"  Input preview: {sample.get('input', '')[:100]}...")
    print(f"  Output preview: {sample.get('output', '')[:100]}...")

    if not result["success"]:
        print(f"  ERROR: {result['error']}")
        if "raw_response" in result:
            print(f"  Raw: {result['raw_response'][:200]}")
        return

    review = result["review"]
    dims = ["coherence", "correctness", "leakage", "format", "difficulty", "overall"]
    for dim in dims:
        if dim in review:
            d = review[dim]
            score = d.get("score", "?")
            rationale = d.get("rationale", "")
            marker = "***" if isinstance(score, int) and score <= 2 else "   "
            print(f"  {marker} {dim:12s}: {score}/5 - {rationale}")


def print_aggregate_report(all_results: dict[str, list]):
    """Print aggregate statistics across all task types."""
    print(f"\n{'='*80}")
    print("AGGREGATE QUALITY REPORT")
    print(f"{'='*80}")

    dims = ["coherence", "correctness", "leakage", "format", "difficulty", "overall"]

    # Per task type
    print(f"\n{'─'*80}")
    print(f"  {'Task Type':<20s}", end="")
    for dim in dims:
        print(f" {dim[:6]:>6s}", end="")
    print(f" {'N':>4s}")
    print(f"{'─'*80}")

    global_scores = defaultdict(list)

    for task_type, results in sorted(all_results.items()):
        successful = [r for r in results if r["result"]["success"]]
        if not successful:
            print(f"  {task_type:<20s}  (no successful reviews)")
            continue

        print(f"  {task_type:<20s}", end="")
        n = len(successful)
        for dim in dims:
            scores = []
            for r in successful:
                review = r["result"]["review"]
                if dim in review and "score" in review[dim]:
                    s = review[dim]["score"]
                    if isinstance(s, (int, float)):
                        scores.append(s)
                        global_scores[dim].append(s)
            if scores:
                avg = sum(scores) / len(scores)
                print(f" {avg:6.2f}", end="")
            else:
                print(f" {'N/A':>6s}", end="")
        print(f" {n:>4d}")

    # Global average
    print(f"{'─'*80}")
    print(f"  {'GLOBAL AVG':<20s}", end="")
    for dim in dims:
        if global_scores[dim]:
            avg = sum(global_scores[dim]) / len(global_scores[dim])
            print(f" {avg:6.2f}", end="")
        else:
            print(f" {'N/A':>6s}", end="")
    total = sum(len([r for r in results if r["result"]["success"]]) for results in all_results.values())
    print(f" {total:>4d}")
    print(f"{'─'*80}")

    # Score distribution for overall
    if global_scores["overall"]:
        print(f"\n  Overall Score Distribution:")
        for score_val in range(1, 6):
            count = global_scores["overall"].count(score_val)
            pct = count / len(global_scores["overall"]) * 100
            bar = "#" * int(pct / 2)
            print(f"    {score_val}/5: {count:3d} ({pct:5.1f}%) {bar}")

    # Low-quality samples (overall <= 2)
    low_quality = []
    for task_type, results in all_results.items():
        for r in results:
            if r["result"]["success"]:
                overall = r["result"]["review"].get("overall", {}).get("score", 5)
                if isinstance(overall, (int, float)) and overall <= 2:
                    low_quality.append((task_type, r["sample"], r["result"]["review"]))

    if low_quality:
        print(f"\n  LOW QUALITY SAMPLES (overall <= 2): {len(low_quality)} found")
        for task_type, sample, review in low_quality[:5]:
            print(f"    - [{task_type}] repo={sample.get('_repo','?')} commit={sample.get('commit','?')[:10]}")
            print(f"      Overall: {review['overall']['score']}/5 - {review['overall']['rationale']}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Review data quality across task types using Claude Haiku"
    )
    parser.add_argument(
        "--data-dir", default=DEFAULT_DATA_DIR,
        help=f"Path to tasks directory (default: {DEFAULT_DATA_DIR})"
    )
    parser.add_argument(
        "--samples-per-type", type=int, default=DEFAULT_SAMPLES_PER_TYPE,
        help=f"Number of samples per task type (default: {DEFAULT_SAMPLES_PER_TYPE})"
    )
    parser.add_argument(
        "--task-types", default=",".join(ALL_TASK_TYPES),
        help="Comma-separated task types to review (default: all 6)"
    )
    parser.add_argument(
        "--gateway-url", default=DEFAULT_GATEWAY_URL,
        help=f"Gateway URL (default: {DEFAULT_GATEWAY_URL})"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Model to use for review (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    task_types = [t.strip() for t in args.task_types.split(",")]
    random.seed(args.seed)

    print("=" * 80)
    print("DATA QUALITY REVIEW")
    print("=" * 80)
    print(f"  Data dir:        {args.data_dir}")
    print(f"  Task types:      {task_types}")
    print(f"  Samples/type:    {args.samples_per_type}")
    print(f"  Model:           {args.model}")
    print(f"  Gateway:         {args.gateway_url}")
    print(f"  Seed:            {args.seed}")

    # Test connectivity
    print(f"\n{'─'*80}")
    print("Testing gateway connectivity...")
    try:
        test_resp = call_llm("Say OK", args.gateway_url, args.model, max_tokens=10)
        print(f"  Gateway OK! Response: {test_resp!r}")
    except Exception as e:
        print(f"  ERROR: Gateway failed: {e}")
        sys.exit(1)

    # Process each task type
    all_results: dict[str, list] = {}

    for task_type in task_types:
        print(f"\n{'='*80}")
        print(f"TASK TYPE: {task_type}")
        print(f"{'='*80}")

        samples = load_samples(args.data_dir, task_type, args.samples_per_type)
        if not samples:
            print(f"  No samples found, skipping.")
            all_results[task_type] = []
            continue

        print(f"  Loaded {len(samples)} samples from {len(set(s['_repo'] for s in samples))} repos")

        type_results = []
        for i, sample in enumerate(samples):
            print(f"\n  Reviewing sample {i+1}/{len(samples)}...", end=" ", flush=True)
            result = review_sample(sample, args.gateway_url, args.model)
            print(f"({result['latency_s']:.1f}s)", end="")

            if result["success"]:
                overall = result["review"].get("overall", {}).get("score", "?")
                print(f" score={overall}/5")
            else:
                print(f" ERROR: {result['error'][:50]}")

            type_results.append({"sample": sample, "result": result})
            print_sample_review(i, sample, result)

        all_results[task_type] = type_results

    # Aggregate report
    print_aggregate_report(all_results)

    print(f"\nDone! Reviewed {sum(len(v) for v in all_results.values())} samples total.")


if __name__ == "__main__":
    main()
