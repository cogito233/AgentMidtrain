#!/usr/bin/env python3
"""
Detect consecutive/related commit chains in the Django repository
for synthesizing multi-step training data.

Chain types:
1. Follow-up Fix: consecutive commits modifying same files/functions
2. Revert->Fix: revert commit followed by a proper fix
3. Feature Development: 3+ commits referencing the same ticket

Output: JSONL file with detected chains.
"""

import subprocess
import re
import json
import os
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path

REPO_PATH = "/data_fast_v3/eremite/cogito_explore/AgentMidtrain/repos/django_django"
OUTPUT_PATH = "/data_fast_v3/eremite/cogito_explore/AgentMidtrain/data/commit_chains_django.jsonl"
NUM_COMMITS = 10000

# Regex patterns
TICKET_PATTERN = re.compile(r'(?:Fixed|Refs|Closes|Related to)\s*#(\d+)', re.IGNORECASE)
REVERT_PATTERN = re.compile(r'^Revert\s+"?(.+?)"?\s*$', re.MULTILINE)
FOLLOWUP_KEYWORDS = re.compile(
    r'follow[\-\s]?up|also\s+fix|edge\s+case|additional|missed|forgot|another\s+fix',
    re.IGNORECASE
)


def get_commits(repo_path, num_commits):
    """Get commit data using git log with custom format."""
    # Format: hash|author_date_iso|subject|body
    # Use ASCII separators to avoid conflicts with commit content
    sep = "\x1f"  # Unit Separator
    record_sep = "\x1e"  # Record Separator

    fmt = f"%H{sep}%aI{sep}%s{sep}%b{record_sep}"

    result = subprocess.run(
        ["git", "log", f"-{num_commits}", f"--format={fmt}", "--no-merges"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=120
    )

    commits = []
    raw_records = result.stdout.split(record_sep)

    for record in raw_records:
        record = record.strip()
        if not record:
            continue
        parts = record.split(sep)
        if len(parts) < 3:
            continue

        commit_hash = parts[0].strip()
        date_str = parts[1].strip()
        subject = parts[2].strip()
        body = parts[3].strip() if len(parts) > 3 else ""

        if not commit_hash or len(commit_hash) != 40:
            continue

        try:
            # Parse ISO date
            date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except (ValueError, IndexError):
            continue

        commits.append({
            "hash": commit_hash,
            "date": date,
            "date_str": date_str,
            "subject": subject,
            "body": body,
            "message": f"{subject}\n{body}".strip(),
        })

    return commits


def get_commit_files(repo_path, commit_hash):
    """Get files modified in a commit."""
    result = subprocess.run(
        ["git", "diff-tree", "--no-commit-id", "-r", "--name-only", commit_hash],
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=30
    )
    return [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]


def get_commit_files_batch(repo_path, commit_hashes):
    """Get files for multiple commits in batch using git log."""
    if not commit_hashes:
        return {}

    # Process in chunks to avoid command line too long
    chunk_size = 200
    all_results = {}

    for i in range(0, len(commit_hashes), chunk_size):
        chunk = commit_hashes[i:i+chunk_size]
        # Use git log with --name-only
        sep = "\x1e"
        fmt = f"{sep}%H"

        cmd = ["git", "log", f"--format={fmt}", "--name-only", "--no-merges"]
        cmd += chunk
        # We need to use git show for multiple specific commits
        # Instead, let's just use diff-tree for each
        for h in chunk:
            files = get_commit_files(repo_path, h)
            all_results[h] = files

    return all_results


def extract_tickets(message):
    """Extract ticket numbers from commit message."""
    return set(TICKET_PATTERN.findall(message))


def detect_followup_chains(commits, files_cache):
    """
    Detect follow-up fix chains: consecutive commits within 7 days
    that modify the same files or reference the same ticket.
    """
    chains = []
    used = set()

    for i in range(len(commits) - 1):
        if i in used:
            continue

        chain = [i]
        current_files = set(files_cache.get(commits[i]["hash"], []))
        current_tickets = extract_tickets(commits[i]["message"])

        for j in range(i + 1, min(i + 10, len(commits))):
            if j in used:
                continue

            c_i = commits[chain[-1]]
            c_j = commits[j]

            # Must be within 7 days
            time_diff = abs((c_i["date"] - c_j["date"]).total_seconds())
            if time_diff > 7 * 86400:
                break

            j_files = set(files_cache.get(c_j["hash"], []))
            j_tickets = extract_tickets(c_j["message"])

            # Check if related
            shared_files = current_files & j_files
            shared_tickets = current_tickets & j_tickets

            has_followup_keyword = bool(FOLLOWUP_KEYWORDS.search(c_j["message"]))
            has_shared_files = len(shared_files) > 0
            has_shared_tickets = len(shared_tickets) > 0

            # Require: shared ticket OR (shared files + keyword hint)
            if has_shared_tickets or (has_shared_files and has_followup_keyword):
                chain.append(j)
                current_files |= j_files
                current_tickets |= j_tickets
            elif has_shared_files and len(shared_files) >= 2:
                # Multiple shared files is a strong signal even without keywords
                chain.append(j)
                current_files |= j_files
                current_tickets |= j_tickets

        if len(chain) >= 2:
            for idx in chain:
                used.add(idx)

            chain_commits = [commits[idx] for idx in chain]
            all_files = set()
            for c in chain_commits:
                all_files |= set(files_cache.get(c["hash"], []))

            shared_f = set(files_cache.get(chain_commits[0]["hash"], []))
            for c in chain_commits[1:]:
                shared_f &= set(files_cache.get(c["hash"], []))

            all_tickets = set()
            for c in chain_commits:
                all_tickets |= extract_tickets(c["message"])

            chains.append({
                "chain_type": "follow_up",
                "commits": [{
                    "hash": c["hash"],
                    "message": c["subject"],
                    "date": c["date_str"],
                    "files": files_cache.get(c["hash"], [])
                } for c in chain_commits],
                "shared_ticket": ",".join(sorted(all_tickets)) if all_tickets else "",
                "shared_files": sorted(list(shared_f))
            })

    return chains


def detect_revert_fix_chains(commits, files_cache):
    """
    Detect Revert->Fix chains: a revert commit followed by a proper fix.
    """
    chains = []

    revert_indices = []
    for i, c in enumerate(commits):
        if "revert" in c["subject"].lower():
            revert_indices.append(i)

    for ri in revert_indices:
        revert_commit = commits[ri]
        revert_files = set(files_cache.get(revert_commit["hash"], []))
        revert_tickets = extract_tickets(revert_commit["message"])

        # Also try to extract the reverted commit's subject
        revert_match = REVERT_PATTERN.search(revert_commit["message"])
        reverted_subject = revert_match.group(1) if revert_match else ""

        # Look at commits BEFORE the revert (lower index = more recent in git log)
        # git log is reverse chronological, so index 0 is most recent
        # A fix after a revert means it has a LOWER index
        for j in range(ri - 1, max(ri - 20, -1), -1):
            fix_commit = commits[j]

            time_diff = abs((revert_commit["date"] - fix_commit["date"]).total_seconds())
            if time_diff > 14 * 86400:
                break

            fix_files = set(files_cache.get(fix_commit["hash"], []))
            fix_tickets = extract_tickets(fix_commit["message"])

            shared_files = revert_files & fix_files
            shared_tickets = revert_tickets & fix_tickets

            if shared_files or shared_tickets:
                # Also find the original commit that was reverted
                # Look after the revert (higher index = older)
                original_commit = None
                for k in range(ri + 1, min(ri + 50, len(commits))):
                    if reverted_subject and reverted_subject.lower() in commits[k]["subject"].lower():
                        original_commit = commits[k]
                        break

                chain_commits = []
                if original_commit:
                    chain_commits.append({
                        "hash": original_commit["hash"],
                        "message": original_commit["subject"],
                        "date": original_commit["date_str"],
                        "files": files_cache.get(original_commit["hash"], [])
                    })

                chain_commits.append({
                    "hash": revert_commit["hash"],
                    "message": revert_commit["subject"],
                    "date": revert_commit["date_str"],
                    "files": files_cache.get(revert_commit["hash"], [])
                })
                chain_commits.append({
                    "hash": fix_commit["hash"],
                    "message": fix_commit["subject"],
                    "date": fix_commit["date_str"],
                    "files": files_cache.get(fix_commit["hash"], [])
                })

                all_tickets = revert_tickets | fix_tickets

                chains.append({
                    "chain_type": "revert_fix",
                    "commits": chain_commits,
                    "shared_ticket": ",".join(sorted(all_tickets)) if all_tickets else "",
                    "shared_files": sorted(list(shared_files))
                })
                break  # Only take the first fix after each revert

    return chains


def detect_feature_chains(commits, files_cache):
    """
    Detect feature development chains: 3+ commits referencing the same ticket,
    modifying overlapping file sets, within 30 days.
    """
    # Group commits by ticket number
    ticket_commits = defaultdict(list)

    for i, c in enumerate(commits):
        tickets = extract_tickets(c["message"])
        for t in tickets:
            ticket_commits[t].append(i)

    chains = []
    used_tickets = set()

    for ticket, indices in ticket_commits.items():
        if len(indices) < 3:
            continue
        if ticket in used_tickets:
            continue

        # Sort by date (indices are already in reverse chronological order from git log)
        indices_sorted = sorted(indices, key=lambda i: commits[i]["date"])

        # Check time span
        time_span = abs((commits[indices_sorted[-1]]["date"] - commits[indices_sorted[0]]["date"]).total_seconds())
        if time_span > 30 * 86400:
            # Try to find a sub-chain within 30 days
            # Take the longest consecutive subsequence within 30 days
            best_chain = []
            for start in range(len(indices_sorted)):
                current_chain = [indices_sorted[start]]
                for end in range(start + 1, len(indices_sorted)):
                    td = abs((commits[indices_sorted[end]]["date"] - commits[indices_sorted[start]]["date"]).total_seconds())
                    if td <= 30 * 86400:
                        current_chain.append(indices_sorted[end])
                    else:
                        break
                if len(current_chain) > len(best_chain):
                    best_chain = current_chain

            if len(best_chain) < 3:
                continue
            indices_sorted = best_chain

        # Check overlapping files
        all_file_sets = []
        for idx in indices_sorted:
            files = set(files_cache.get(commits[idx]["hash"], []))
            all_file_sets.append(files)

        # Find files that appear in at least 2 commits
        file_count = defaultdict(int)
        for fs in all_file_sets:
            for f in fs:
                file_count[f] += 1

        overlapping_files = [f for f, count in file_count.items() if count >= 2]

        if not overlapping_files:
            continue

        used_tickets.add(ticket)

        chain_commits = [{
            "hash": commits[idx]["hash"],
            "message": commits[idx]["subject"],
            "date": commits[idx]["date_str"],
            "files": files_cache.get(commits[idx]["hash"], [])
        } for idx in indices_sorted]

        chains.append({
            "chain_type": "feature",
            "commits": chain_commits,
            "shared_ticket": ticket,
            "shared_files": sorted(overlapping_files[:20])  # Limit for readability
        })

    # Sort by chain length descending
    chains.sort(key=lambda x: len(x["commits"]), reverse=True)

    return chains


def main():
    print(f"Analyzing {NUM_COMMITS} commits from Django repository...")
    print(f"Repo: {REPO_PATH}")
    print()

    # Step 1: Get commits
    print("Step 1: Fetching commit history...")
    commits = get_commits(REPO_PATH, NUM_COMMITS)
    print(f"  Retrieved {len(commits)} commits")

    if not commits:
        print("ERROR: No commits found!")
        return

    print(f"  Date range: {commits[-1]['date_str'][:10]} to {commits[0]['date_str'][:10]}")
    print()

    # Step 2: Get files for all commits (this is the slow part)
    print("Step 2: Fetching file lists for commits...")
    all_hashes = [c["hash"] for c in commits]

    # Use git log with --name-only and a unique marker per commit
    print("  Using batch git log approach...")
    marker = "COMMIT_MARKER_XYZ_"
    result = subprocess.run(
        ["git", "log", f"-{NUM_COMMITS}", f"--format={marker}%H", "--name-only", "--no-merges"],
        cwd=REPO_PATH,
        capture_output=True,
        text=True,
        timeout=300
    )

    files_cache = {}
    current_hash = None
    current_files = []

    for line in result.stdout.split('\n'):
        if line.startswith(marker):
            if current_hash:
                files_cache[current_hash] = current_files
            current_hash = line[len(marker):].strip()
            current_files = []
        elif line.strip() and current_hash:
            current_files.append(line.strip())

    if current_hash:
        files_cache[current_hash] = current_files

    print(f"  Got file lists for {len(files_cache)} commits")
    print()

    # Step 3: Detect chains
    print("Step 3: Detecting follow-up fix chains...")
    followup_chains = detect_followup_chains(commits, files_cache)
    print(f"  Found {len(followup_chains)} follow-up chains")

    print("Step 4: Detecting revert->fix chains...")
    revert_chains = detect_revert_fix_chains(commits, files_cache)
    print(f"  Found {len(revert_chains)} revert->fix chains")

    print("Step 5: Detecting feature development chains...")
    feature_chains = detect_feature_chains(commits, files_cache)
    print(f"  Found {len(feature_chains)} feature development chains")
    print()

    # Step 4: Statistics
    all_chains = followup_chains + revert_chains + feature_chains

    print("=" * 70)
    print("STATISTICS")
    print("=" * 70)
    print(f"Total chains detected: {len(all_chains)}")
    print(f"  Follow-up fix chains: {len(followup_chains)}")
    print(f"  Revert->fix chains:   {len(revert_chains)}")
    print(f"  Feature dev chains:   {len(feature_chains)}")
    print()

    if followup_chains:
        avg_len = sum(len(c["commits"]) for c in followup_chains) / len(followup_chains)
        print(f"Average follow-up chain length: {avg_len:.1f} commits")
    if revert_chains:
        avg_len = sum(len(c["commits"]) for c in revert_chains) / len(revert_chains)
        print(f"Average revert->fix chain length: {avg_len:.1f} commits")
    if feature_chains:
        avg_len = sum(len(c["commits"]) for c in feature_chains) / len(feature_chains)
        print(f"Average feature chain length: {avg_len:.1f} commits")
    print()

    # Show examples
    print("=" * 70)
    print("EXAMPLE CHAINS")
    print("=" * 70)

    # Best follow-up examples (longest chains with shared tickets)
    print("\n--- Follow-up Fix Chains (top 5) ---")
    followup_sorted = sorted(followup_chains, key=lambda x: (len(x["shared_ticket"]) > 0, len(x["commits"])), reverse=True)
    for chain in followup_sorted[:5]:
        print(f"\n  Ticket: #{chain['shared_ticket'] or 'N/A'}")
        print(f"  Shared files: {chain['shared_files'][:3]}")
        for c in chain["commits"]:
            print(f"    [{c['date'][:10]}] {c['hash'][:8]} {c['message'][:80]}")

    # Best revert examples
    print("\n--- Revert->Fix Chains (top 5) ---")
    for chain in revert_chains[:5]:
        print(f"\n  Ticket: #{chain['shared_ticket'] or 'N/A'}")
        print(f"  Shared files: {chain['shared_files'][:3]}")
        for c in chain["commits"]:
            print(f"    [{c['date'][:10]}] {c['hash'][:8]} {c['message'][:80]}")

    # Best feature examples (longest chains)
    print("\n--- Feature Development Chains (top 5) ---")
    for chain in feature_chains[:5]:
        print(f"\n  Ticket: #{chain['shared_ticket']}")
        print(f"  Chain length: {len(chain['commits'])} commits")
        print(f"  Shared files: {chain['shared_files'][:5]}")
        for c in chain["commits"][:6]:
            print(f"    [{c['date'][:10]}] {c['hash'][:8]} {c['message'][:80]}")
        if len(chain["commits"]) > 6:
            print(f"    ... and {len(chain['commits']) - 6} more commits")

    # Step 5: Write output
    print("\n" + "=" * 70)
    print(f"Writing output to {OUTPUT_PATH}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, 'w') as f:
        for chain in all_chains:
            f.write(json.dumps(chain, ensure_ascii=False) + '\n')

    print(f"Written {len(all_chains)} chains to JSONL file")
    print(f"File size: {os.path.getsize(OUTPUT_PATH) / 1024:.1f} KB")
    print("\nDone!")


if __name__ == "__main__":
    main()
