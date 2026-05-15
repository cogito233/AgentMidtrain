#!/usr/bin/env python3
"""Deduplicate JSONL task files based on exact and fuzzy output matching.

Uses MinHash + LSH for fuzzy deduplication (O(n) instead of O(n^2)).
Falls back to a pure-Python MinHash implementation if datasketch is not installed.

Usage:
    python dedup_tasks.py --input-dir data/tasks/django_v2 --output-dir data/tasks/django_v2_dedup
    python dedup_tasks.py --input-dir data/tasks/django_v2 --output-dir data/tasks/django_v2_dedup --method exact
    python dedup_tasks.py --input-dir data/tasks/django_v2 --output-dir data/tasks/django_v2_dedup --cross-task
    python dedup_tasks.py --input-dir data/tasks/django_v2 --output-dir data/tasks/django_v2_dedup --threshold 0.85
"""

import argparse
import hashlib
import json
import logging
import struct
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try importing datasketch; fall back to pure-Python implementation
# ---------------------------------------------------------------------------
try:
    from datasketch import MinHash as DSMinHash
    from datasketch import MinHashLSH as DSMinHashLSH

    HAS_DATASKETCH = True
    logger.info("Using datasketch library for MinHash + LSH.")
except ImportError:
    HAS_DATASKETCH = False
    logger.info("datasketch not available; using pure-Python MinHash + LSH fallback.")


# ===========================================================================
# Pure-Python MinHash + LSH (stdlib only)
# ===========================================================================


def _char_ngrams(text: str, n: int = 5) -> set[str]:
    """Generate character-level n-gram shingles from text."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def minhash_signature(text: str, num_perm: int = 128, ngram_size: int = 5) -> list[int]:
    """Generate MinHash signature for text using char-level n-grams.

    Each of the *num_perm* hash functions is simulated by prepending a
    unique seed index to every shingle before hashing with MD5. We keep
    the minimum 32-bit hash value per permutation.
    """
    shingles = _char_ngrams(text, ngram_size)

    if not shingles:
        # Empty text gets max-value signature so it never matches anything.
        return [0xFFFFFFFF] * num_perm

    signature: list[int] = []
    for i in range(num_perm):
        min_hash = 0xFFFFFFFF
        for shingle in shingles:
            h = hashlib.md5(f"{i}:{shingle}".encode()).digest()
            hash_val = struct.unpack("<I", h[:4])[0]
            min_hash = min(min_hash, hash_val)
        signature.append(min_hash)

    return signature


def estimate_jaccard(sig1: list[int], sig2: list[int]) -> float:
    """Estimate Jaccard similarity from MinHash signatures."""
    if len(sig1) != len(sig2):
        raise ValueError("Signatures must have the same length.")
    if not sig1:
        return 0.0
    return sum(1 for a, b in zip(sig1, sig2) if a == b) / len(sig1)


class PurePythonLSH:
    """Locality-Sensitive Hashing index backed by banding of MinHash signatures.

    Given *num_perm* hash permutations and a Jaccard *threshold*, we
    automatically choose the number of bands (*b*) and rows per band (*r*)
    such that the probability of a candidate pair at threshold is approx 0.5.

    Insert items, then query for candidates. Only candidate pairs need
    a full signature comparison, keeping dedup close to O(n).
    """

    def __init__(self, threshold: float = 0.8, num_perm: int = 128):
        self.threshold = threshold
        self.num_perm = num_perm
        # Choose b, r such that (1/b)^(1/r) ~ threshold
        self.bands, self.rows = self._optimal_params(num_perm, threshold)
        logger.info(
            f"LSH params: num_perm={num_perm}, bands={self.bands}, "
            f"rows={self.rows}, threshold={threshold:.2f}"
        )
        # Each band has a dict mapping band_hash -> set of item keys
        self._buckets: list[dict[int, set[int]]] = [
            defaultdict(set) for _ in range(self.bands)
        ]
        self._signatures: dict[int, list[int]] = {}

    @staticmethod
    def _optimal_params(num_perm: int, threshold: float) -> tuple[int, int]:
        """Find (b, r) with b*r <= num_perm that gives the best approximation
        of the threshold for the S-curve P = 1 - (1 - t^r)^b at P=0.5.

        The inflection point of the S-curve is approximately (1/b)^(1/r).
        """
        best_b, best_r = 1, num_perm
        best_err = float("inf")
        for b in range(1, num_perm + 1):
            r = num_perm // b
            if r == 0:
                continue
            try:
                approx_thresh = (1.0 / b) ** (1.0 / r)
            except (ZeroDivisionError, OverflowError):
                continue
            err = abs(approx_thresh - threshold)
            if err < best_err:
                best_err = err
                best_b, best_r = b, r
        return best_b, best_r

    def _band_hash(self, sig: list[int], band_idx: int) -> int:
        """Hash a single band (a slice of the signature)."""
        start = band_idx * self.rows
        end = start + self.rows
        return hash(tuple(sig[start:end]))

    def insert(self, key: int, signature: list[int]) -> None:
        """Insert an item with *key* and its MinHash *signature*."""
        self._signatures[key] = signature
        for b in range(self.bands):
            bh = self._band_hash(signature, b)
            self._buckets[b][bh].add(key)

    def query(self, signature: list[int]) -> set[int]:
        """Return candidate keys that share at least one band with *signature*."""
        candidates: set[int] = set()
        for b in range(self.bands):
            bh = self._band_hash(signature, b)
            bucket = self._buckets[b].get(bh)
            if bucket:
                candidates.update(bucket)
        return candidates

    def get_signature(self, key: int) -> list[int] | None:
        return self._signatures.get(key)


# ===========================================================================
# datasketch-backed wrappers (thin layer over datasketch API)
# ===========================================================================

if HAS_DATASKETCH:

    class DatasketchLSH:
        """Thin wrapper so the main dedup logic can be agnostic about backend."""

        def __init__(self, threshold: float = 0.8, num_perm: int = 128):
            self.threshold = threshold
            self.num_perm = num_perm
            self._lsh = DSMinHashLSH(threshold=threshold, num_perm=num_perm)
            self._minhashes: dict[int, "DSMinHash"] = {}

        @staticmethod
        def make_signature(
            text: str, num_perm: int = 128, ngram_size: int = 5
        ) -> "DSMinHash":
            m = DSMinHash(num_perm=num_perm)
            shingles = _char_ngrams(text, ngram_size)
            for s in shingles:
                m.update(s.encode("utf-8", errors="replace"))
            return m

        def insert(self, key: int, mh: "DSMinHash") -> None:
            self._minhashes[key] = mh
            try:
                self._lsh.insert(str(key), mh)
            except ValueError:
                pass  # key already exists

        def query(self, mh: "DSMinHash") -> set[int]:
            result = self._lsh.query(mh)
            return {int(r) for r in result}

        def get_minhash(self, key: int):
            return self._minhashes.get(key)


# ===========================================================================
# IO helpers
# ===========================================================================


def load_jsonl(filepath: Path) -> list[dict[str, Any]]:
    """Load a JSONL file, returning a list of parsed JSON objects."""
    records: list[dict[str, Any]] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed JSON at {filepath}:{line_num}: {e}")
    return records


def write_jsonl(filepath: Path, records: list[dict[str, Any]]) -> None:
    """Write a list of records to a JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_output_field(record: dict[str, Any]) -> str:
    """Extract the output field from a record, handling missing values."""
    output = record.get("output", "")
    if output is None:
        return ""
    return str(output)


def compute_output_hash(output: str) -> str:
    """Compute MD5 hash of the output field for exact dedup."""
    return hashlib.md5(output.encode("utf-8")).hexdigest()


# ===========================================================================
# Dedup stages
# ===========================================================================


def dedup_exact(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Deduplicate records by exact output hash. First occurrence wins."""
    seen_hashes: set[str] = set()
    deduped: list[dict[str, Any]] = []
    removed = 0

    for record in records:
        output = get_output_field(record)
        h = compute_output_hash(output)
        if h in seen_hashes:
            removed += 1
            continue
        seen_hashes.add(h)
        deduped.append(record)

    return deduped, removed


def dedup_minhash(
    records: list[dict[str, Any]],
    threshold: float = 0.8,
    num_perm: int = 128,
    ngram_size: int = 5,
) -> tuple[list[dict[str, Any]], int]:
    """Fuzzy-deduplicate records using MinHash + LSH.

    For each record we compute a MinHash signature of the output field,
    insert it into an LSH index, and check for candidate near-duplicates
    among previously inserted records. Only candidate pairs (those sharing
    at least one LSH band) are compared, so the expected cost is O(n).
    """
    if not records:
        return records, 0

    kept: list[dict[str, Any]] = []
    removed = 0

    if HAS_DATASKETCH:
        # ----- datasketch path -----
        lsh = DatasketchLSH(threshold=threshold, num_perm=num_perm)
        for idx, record in enumerate(records):
            output = get_output_field(record)
            mh = DatasketchLSH.make_signature(
                output, num_perm=num_perm, ngram_size=ngram_size
            )

            # Query existing index for candidates
            candidates = lsh.query(mh)
            is_dup = False
            for cand_key in candidates:
                cand_mh = lsh.get_minhash(cand_key)
                if cand_mh is not None:
                    sim = mh.jaccard(cand_mh)
                    if sim >= threshold:
                        is_dup = True
                        break

            if is_dup:
                removed += 1
            else:
                lsh.insert(len(kept), mh)
                kept.append(record)
    else:
        # ----- pure-Python path -----
        lsh = PurePythonLSH(threshold=threshold, num_perm=num_perm)
        for idx, record in enumerate(records):
            output = get_output_field(record)
            sig = minhash_signature(output, num_perm=num_perm, ngram_size=ngram_size)

            # Query existing index for candidates
            candidates = lsh.query(sig)
            is_dup = False
            for cand_key in candidates:
                cand_sig = lsh.get_signature(cand_key)
                if cand_sig is not None:
                    sim = estimate_jaccard(sig, cand_sig)
                    if sim >= threshold:
                        is_dup = True
                        break

            if is_dup:
                removed += 1
            else:
                lsh.insert(len(kept), sig)
                kept.append(record)

    return kept, removed


# ===========================================================================
# Cross-task overlap detection
# ===========================================================================


def cross_task_check(
    task_records: dict[str, list[dict[str, Any]]],
    threshold: float,
    num_perm: int = 128,
    ngram_size: int = 5,
) -> None:
    """Detect cross-task duplicates using MinHash similarity.

    Groups records by commit identifier, then checks for high similarity
    across task types. Logs warnings but does not remove.
    """
    # Build index: commit_id -> [(task_type, output)]
    commit_index: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for task_type, records in task_records.items():
        for record in records:
            commit_id = (
                record.get("commit")
                or record.get("commit_hash")
                or record.get("commit_id")
                or record.get("instance_id")
            )
            if commit_id:
                output = get_output_field(record)
                commit_index[commit_id].append((task_type, output))

    cross_task_warnings = 0
    checked_pairs: set[tuple[str, str, str]] = set()

    for commit_id, entries in commit_index.items():
        if len(entries) < 2:
            continue

        # Pre-compute signatures for this commit's entries
        sigs = []
        for _, output in entries:
            if HAS_DATASKETCH:
                mh = DatasketchLSH.make_signature(
                    output, num_perm=num_perm, ngram_size=ngram_size
                )
                sigs.append(mh)
            else:
                sig = minhash_signature(output, num_perm=num_perm, ngram_size=ngram_size)
                sigs.append(sig)

        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                task_a = entries[i][0]
                task_b = entries[j][0]

                if task_a == task_b:
                    continue

                pair_key = (commit_id, min(task_a, task_b), max(task_a, task_b))
                if pair_key in checked_pairs:
                    continue
                checked_pairs.add(pair_key)

                if HAS_DATASKETCH:
                    sim = sigs[i].jaccard(sigs[j])
                else:
                    sim = estimate_jaccard(sigs[i], sigs[j])

                if sim >= threshold:
                    cross_task_warnings += 1
                    logger.warning(
                        f"Cross-task duplicate detected: commit={commit_id}, "
                        f"tasks=[{task_a}, {task_b}], "
                        f"similarity={sim:.2%} (>={threshold:.0%})"
                    )

    if cross_task_warnings > 0:
        logger.warning(f"Total cross-task duplicate warnings: {cross_task_warnings}")
    else:
        logger.info("No cross-task duplicates detected.")


# ===========================================================================
# File processing
# ===========================================================================


def process_file(
    input_path: Path,
    output_path: Path,
    method: str,
    threshold: float,
    num_perm: int,
    ngram_size: int,
) -> dict[str, int]:
    """Process a single JSONL file: exact dedup, then optional MinHash fuzzy dedup.

    Returns statistics dict.
    """
    records = load_jsonl(input_path)
    stats: dict[str, int] = {"total": len(records)}

    # --- Stage 1: exact dedup (always) ---
    records, exact_removed = dedup_exact(records)
    stats["exact_removed"] = exact_removed

    # --- Stage 2: fuzzy dedup (MinHash + LSH) ---
    if method == "minhash":
        records, fuzzy_removed = dedup_minhash(
            records,
            threshold=threshold,
            num_perm=num_perm,
            ngram_size=ngram_size,
        )
        stats["fuzzy_removed"] = fuzzy_removed
    else:
        stats["fuzzy_removed"] = 0

    stats["final"] = len(records)

    # Write output
    write_jsonl(output_path, records)
    return stats


# ===========================================================================
# Statistics
# ===========================================================================


def print_statistics(all_stats: dict[str, dict[str, int]], method: str) -> None:
    """Print formatted dedup statistics."""
    total_before = 0
    total_exact = 0
    total_fuzzy = 0
    total_after = 0

    print("\n" + "=" * 78)
    print(f"{'DEDUPLICATION RESULTS':^78}")
    print("=" * 78)
    print(
        f"{'Task File':<30} {'Before':>8} {'Exact-':>8} {'Fuzzy-':>8} "
        f"{'After':>8} {'Rate':>8}"
    )
    print("-" * 78)

    for filename, stats in sorted(all_stats.items()):
        before = stats["total"]
        exact = stats["exact_removed"]
        fuzzy = stats["fuzzy_removed"]
        after = stats["final"]
        rate = (1 - after / before) * 100 if before > 0 else 0.0

        total_before += before
        total_exact += exact
        total_fuzzy += fuzzy
        total_after += after

        print(
            f"{filename:<30} {before:>8} {exact:>8} {fuzzy:>8} "
            f"{after:>8} {rate:>7.1f}%"
        )

    print("-" * 78)
    overall_rate = (
        (1 - total_after / total_before) * 100 if total_before > 0 else 0.0
    )
    print(
        f"{'TOTAL':<30} {total_before:>8} {total_exact:>8} {total_fuzzy:>8} "
        f"{total_after:>8} {overall_rate:>7.1f}%"
    )
    print("=" * 78)
    method_label = {
        "exact": "exact only",
        "minhash": f"exact + MinHash/LSH ({'datasketch' if HAS_DATASKETCH else 'pure-Python'})",
    }
    print(f"Method: {method_label.get(method, method)}")
    print()


# ===========================================================================
# CLI
# ===========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deduplicate JSONL task files based on output content.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input-dir data/tasks/v2 --output-dir data/tasks/v2_dedup
  %(prog)s --input-dir data/tasks/v2 --output-dir data/tasks/v2_dedup --method exact
  %(prog)s --input-dir data/tasks/v2 --output-dir data/tasks/v2_dedup --threshold 0.85
  %(prog)s --input-dir data/tasks/v2 --output-dir data/tasks/v2_dedup --cross-task
        """,
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing input JSONL task files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write deduplicated JSONL files.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["exact", "minhash"],
        default="minhash",
        help="Dedup method: 'exact' (hash only) or 'minhash' (exact + MinHash/LSH fuzzy). Default: minhash.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Jaccard similarity threshold for fuzzy dedup (default: 0.8).",
    )
    parser.add_argument(
        "--num-perm",
        type=int,
        default=128,
        help="Number of MinHash permutations (default: 128). More = more accurate but slower.",
    )
    parser.add_argument(
        "--ngram-size",
        type=int,
        default=5,
        help="Character n-gram size for shingling (default: 5).",
    )
    parser.add_argument(
        "--cross-task",
        action="store_true",
        default=False,
        help="Enable cross-task duplicate detection (log warnings, no auto-removal).",
    )

    args = parser.parse_args()

    method = args.method
    threshold = args.threshold
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    num_perm = args.num_perm
    ngram_size = args.ngram_size

    if not input_dir.is_dir():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    if not (0 < threshold <= 1):
        logger.error("Threshold must be in (0, 1].")
        sys.exit(1)

    # Find all JSONL files
    jsonl_files = sorted(input_dir.glob("*.jsonl"))
    if not jsonl_files:
        logger.error(f"No .jsonl files found in {input_dir}")
        sys.exit(1)

    logger.info(f"Found {len(jsonl_files)} JSONL file(s) in {input_dir}")
    logger.info(
        f"Method: {method} | Threshold: {threshold:.2f} | Perms: {num_perm} | Ngram: {ngram_size}"
    )
    logger.info(f"Output directory: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each file
    all_stats: dict[str, dict[str, int]] = {}
    task_records: dict[str, list[dict[str, Any]]] = {}

    for jsonl_file in jsonl_files:
        filename = jsonl_file.name
        output_path = output_dir / filename

        logger.info(f"Processing: {filename}")
        stats = process_file(
            jsonl_file,
            output_path,
            method=method,
            threshold=threshold,
            num_perm=num_perm,
            ngram_size=ngram_size,
        )
        all_stats[filename] = stats

        # If cross-task checking is enabled, load the deduped records
        if args.cross_task:
            task_type = jsonl_file.stem
            task_records[task_type] = load_jsonl(output_path)

    # Cross-task dedup check
    if args.cross_task and len(task_records) > 1:
        logger.info("Running cross-task duplicate detection...")
        cross_task_check(
            task_records,
            threshold=threshold,
            num_perm=num_perm,
            ngram_size=ngram_size,
        )

    # Print statistics
    print_statistics(all_stats, method)


if __name__ == "__main__":
    main()
