#!/usr/bin/env python3
"""Convert downloaded general corpora to OpenAI messages JSONL format.

Handles:
- CommitPackFT: commit-based code editing (old_contents → new_contents)
- Magicoder: problem → solution pairs
- CodeSearchNet: docstring → code pairs

Output: {"messages": [...], "metadata": {...}}
"""

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE = Path("/data_fast_v3/eremite/cogito_explore/AgentMidtrain/data/general_corpora")
OUT = Path("/data_fast_v3/eremite/cogito_explore/AgentMidtrain/data/training_messages_v6/general")


def convert_commitpackft():
    """CommitPackFT: given old code + commit message, produce new code."""
    out_dir = OUT / "commitpackft"
    out_dir.mkdir(parents=True, exist_ok=True)
    src_dir = BASE / "commitpackft"
    if not src_dir.exists():
        logger.warning("CommitPackFT not found, skipping")
        return 0

    total = 0
    for lang_file in sorted(src_dir.glob("*.jsonl")):
        lang = lang_file.stem
        out_path = out_dir / f"{lang}.jsonl"
        count = 0
        with open(lang_file) as fin, open(out_path, "w") as fout:
            for line in fin:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                old = obj.get("old_contents", "").strip()
                new = obj.get("new_contents", "").strip()
                subject = obj.get("subject", "").strip()
                message = obj.get("message", "").strip()
                filepath = obj.get("new_file", obj.get("old_file", ""))

                if not old or not new or not subject:
                    continue

                desc = message if message and message != subject else subject
                query = f"File: {filepath}\n\nChange request: {desc}\n\nCurrent code:\n```{lang.lower()}\n{old[:4000]}\n```"
                answer = f"```{lang.lower()}\n{new[:4000]}\n```"

                out_obj = {
                    "messages": [
                        {"role": "system", "content": "You are a software engineer. Given a file and a change request, produce the updated code."},
                        {"role": "user", "content": query},
                        {"role": "assistant", "content": answer},
                    ],
                    "metadata": {
                        "source": "commitpackft",
                        "lang": lang,
                        "commit": obj.get("commit", ""),
                        "license": obj.get("license", ""),
                    },
                }
                fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                count += 1

        logger.info(f"  CommitPackFT/{lang}: {count:,} samples")
        total += count
    return total


def convert_magicoder():
    """Magicoder: problem → solution instruction pairs."""
    out_dir = OUT / "magicoder"
    out_dir.mkdir(parents=True, exist_ok=True)
    src_dir = BASE / "magicoder"
    if not src_dir.exists():
        logger.warning("Magicoder not found, skipping")
        return 0

    total = 0
    for src_file in sorted(src_dir.glob("*.jsonl")):
        name = src_file.stem
        out_path = out_dir / f"{name}.jsonl"
        count = 0
        with open(src_file) as fin, open(out_path, "w") as fout:
            for line in fin:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                problem = (obj.get("problem") or obj.get("instruction") or "").strip()
                solution = (obj.get("solution") or obj.get("response") or "").strip()
                if not problem or not solution:
                    continue

                out_obj = {
                    "messages": [
                        {"role": "system", "content": "You are an expert programmer. Solve the given programming problem."},
                        {"role": "user", "content": problem},
                        {"role": "assistant", "content": solution},
                    ],
                    "metadata": {
                        "source": f"magicoder/{name}",
                        "lang": obj.get("lang", ""),
                    },
                }
                fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                count += 1

        logger.info(f"  Magicoder/{name}: {count:,} samples")
        total += count
    return total


def convert_codesearchnet():
    """CodeSearchNet: docstring → code understanding pairs."""
    out_dir = OUT / "codesearchnet"
    out_dir.mkdir(parents=True, exist_ok=True)
    src_dir = BASE / "codesearchnet"
    if not src_dir.exists():
        logger.warning("CodeSearchNet not found, skipping")
        return 0

    total = 0
    for lang_file in sorted(src_dir.glob("*.jsonl")):
        lang = lang_file.stem
        out_path = out_dir / f"{lang}.jsonl"
        count = 0
        with open(lang_file) as fin, open(out_path, "w") as fout:
            for line in fin:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                docstring = obj.get("func_documentation_string", "").strip()
                code = obj.get("whole_func_string", "").strip()
                func_name = obj.get("func_name", "")
                repo = obj.get("repository_name", "")

                if not docstring or not code or len(docstring) < 20:
                    continue

                query = f"Write a {lang} function that does the following:\n\n{docstring}"
                answer = f"```{lang}\n{code[:4000]}\n```"

                out_obj = {
                    "messages": [
                        {"role": "system", "content": f"You are an expert {lang} programmer."},
                        {"role": "user", "content": query},
                        {"role": "assistant", "content": answer},
                    ],
                    "metadata": {
                        "source": "codesearchnet",
                        "lang": lang,
                        "func_name": func_name,
                        "repo": repo,
                    },
                }
                fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                count += 1

        logger.info(f"  CodeSearchNet/{lang}: {count:,} samples")
        total += count
    return total


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    logger.info(f"Converting general corpora → {OUT}")

    t1 = convert_commitpackft()
    t2 = convert_magicoder()
    t3 = convert_codesearchnet()

    grand_total = t1 + t2 + t3
    logger.info("=" * 60)
    logger.info(f"Grand total: {grand_total:,} samples")
    logger.info(f"  CommitPackFT: {t1:,}")
    logger.info(f"  Magicoder: {t2:,}")
    logger.info(f"  CodeSearchNet: {t3:,}")
    logger.info(f"Output: {OUT}")


if __name__ == "__main__":
    main()
