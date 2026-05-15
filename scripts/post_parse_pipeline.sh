#!/bin/bash
# Post-parse pipeline: wait for parse_commits to finish, then filter + generate tasks
# Usage: bash post_parse_pipeline.sh

BASE="/data_fast_v3/eremite/cogito_explore/AgentMidtrain"
REPOS="certbot_certbot getsentry_sentry home-assistant_core saltstack_salt twisted_twisted"
LOG="$BASE/logs/post_parse_pipeline.log"
mkdir -p "$BASE/logs"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$LOG"; }

for repo in $REPOS; do
  raw="$BASE/data/filtered_commits/${repo}_raw.jsonl"
  
  # Wait for parse to finish
  log "Waiting for $repo parse to complete..."
  while true; do
    if ! ps aux | grep -v grep | grep "parse_commits.*$repo" > /dev/null 2>&1; then
      break
    fi
    sleep 30
  done
  
  if [ ! -f "$raw" ]; then
    log "ERROR: $repo parse finished but no raw file at $raw"
    continue
  fi
  
  raw_count=$(wc -l < "$raw")
  log "$repo: Raw parse complete - $raw_count commits"
  
  # Strict filter
  strict="$BASE/data/filtered_commits/${repo}.jsonl"
  log "$repo: Running strict filter..."
  python "$BASE/scripts/filter_commits.py" "$raw" -o "$strict" \
    --require-test --require-src --fix-keywords --python-only \
    --max-src-files 5 --max-edit-lines 200 --max-patch-length 10000 2>&1 | tail -3 | tee -a "$LOG"
  
  strict_count=$(wc -l < "$strict" 2>/dev/null || echo 0)
  log "$repo: Strict filter - $strict_count commits"
  
  # Relaxed filter
  relaxed="$BASE/data/filtered_commits/${repo}_relaxed.jsonl"
  log "$repo: Running relaxed filter..."
  python "$BASE/scripts/filter_commits.py" "$raw" -o "$relaxed" \
    --require-src --python-only --exclude-docs \
    --max-src-files 8 --max-edit-lines 300 --max-patch-length 15000 2>&1 | tail -3 | tee -a "$LOG"
  
  relaxed_count=$(wc -l < "$relaxed" 2>/dev/null || echo 0)
  log "$repo: Relaxed filter - $relaxed_count commits"
  
  # Generate strict tasks
  if [ "$strict_count" -gt 0 ]; then
    outdir="$BASE/data/tasks/$repo"
    mkdir -p "$outdir"
    log "$repo: Generating strict tasks..."
    python "$BASE/scripts/generate_tasks.py" \
      --input "$strict" \
      --output-dir "$outdir" \
      --repo-path "$BASE/repos/$repo" \
      --workers 4 2>&1 | tail -10 | tee -a "$LOG"
    log "$repo: Strict tasks done"
  fi
  
  # Generate relaxed tasks
  if [ "$relaxed_count" -gt 0 ]; then
    outdir="$BASE/data/tasks/${repo}_relaxed"
    mkdir -p "$outdir"
    log "$repo: Generating relaxed tasks..."
    python "$BASE/scripts/generate_tasks.py" \
      --input "$relaxed" \
      --output-dir "$outdir" \
      --repo-path "$BASE/repos/$repo" \
      --task-types localization,edit_generation,commit_message,code_review,bug_detection \
      --workers 4 2>&1 | tail -10 | tee -a "$LOG"
    log "$repo: Relaxed tasks done"
  fi
  
  log "$repo: COMPLETE"
  log "============================================================"
done

log "=== ALL REPOS COMPLETE ==="
