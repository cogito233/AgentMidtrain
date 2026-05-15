# Mid-Training Data Synthesis Budget Estimate

## Scope

- **25 repos**, ~530K total commits
- **Strict filter** (fix keyword + test + src, <=5 src files): ~42K candidates
- **Relaxed filter** (no fix keyword, larger patches): ~80-100K candidates
- Analysis below uses **42K (strict)** as base and **100K (relaxed)** as upper bound

---

## Part 1: Zero-Cost Tasks (Pure Git Extraction)

These tasks require NO LLM calls -- only git operations and template logic:

| Task Type | Output per Candidate | Total Samples (42K base) |
|-----------|---------------------|--------------------------|
| Localization QA | 1-3 per commit | ~80K |
| Edit Generation QA | 1 per commit | ~42K |
| Test Writing QA | 1 per commit | ~42K |
| Commit Message Generation QA | 1 per commit | ~42K |
| Code Review QA | 1 per commit | ~42K |
| Error Recovery (Level 1, template) | 2-3 per commit | ~100K |

**Subtotal free samples: ~350K QA pairs** (strict) / **~830K** (relaxed)

---

## Part 2: LLM-Required Tasks -- Token Estimates

### Task A: Issue Synthesis
- **Purpose**: Transform commit message into natural language issue description
- **Candidates**: All 42K (strict) or 100K (relaxed)
- **Tokens per sample**:
  - Input: ~600 tokens (commit msg + src patch summary + test patch summary)
  - Output: ~200 tokens

| Scenario | Candidates | Input Tokens | Output Tokens |
|----------|-----------|--------------|---------------|
| Strict (42K) | 42,000 | 25.2M | 8.4M |
| Relaxed (100K) | 100,000 | 60.0M | 20.0M |

### Task B: Bug Description Synthesis
- **Purpose**: Generate natural bug description from buggy code region
- **Candidates**: All 42K (strict) or 100K (relaxed)
- **Tokens per sample**:
  - Input: ~400 tokens (buggy code region + context)
  - Output: ~150 tokens

| Scenario | Candidates | Input Tokens | Output Tokens |
|----------|-----------|--------------|---------------|
| Strict (42K) | 42,000 | 16.8M | 6.3M |
| Relaxed (100K) | 100,000 | 40.0M | 15.0M |

### Task C: Negative Capability Samples
- **Purpose**: Generate "why not this file" reasoning for localization training
- **Candidates**: ~20K (subset of strict) or ~40K (relaxed)
- **Tokens per sample**:
  - Input: ~300 tokens
  - Output: ~100 tokens

| Scenario | Candidates | Input Tokens | Output Tokens |
|----------|-----------|--------------|---------------|
| Strict (20K) | 20,000 | 6.0M | 2.0M |
| Relaxed (40K) | 40,000 | 12.0M | 4.0M |

### Task D: Error Recovery (Level 2 -- Weak Model Attempts)
- **Purpose**: Generate plausible-but-wrong fix attempts for error recovery training
- **Candidates**: ~10K (strict) or ~25K (relaxed)
- **Tokens per sample**:
  - Input: ~800 tokens (issue + repo context + file content)
  - Output: ~500 tokens (attempted fix)

| Scenario | Candidates | Input Tokens | Output Tokens |
|----------|-----------|--------------|---------------|
| Strict (10K) | 10,000 | 8.0M | 5.0M |
| Relaxed (25K) | 25,000 | 20.0M | 12.5M |

---

## Part 3: Total Token Consumption

### Strict Scenario (42K candidates)

| Task | Input Tokens | Output Tokens | Total Tokens |
|------|-------------|---------------|--------------|
| Issue Synthesis | 25.2M | 8.4M | 33.6M |
| Bug Description | 16.8M | 6.3M | 23.1M |
| Negative Capability | 6.0M | 2.0M | 8.0M |
| Error Recovery L2 | 8.0M | 5.0M | 13.0M |
| **Total** | **56.0M** | **21.7M** | **77.7M** |

### Relaxed Scenario (100K candidates)

| Task | Input Tokens | Output Tokens | Total Tokens |
|------|-------------|---------------|--------------|
| Issue Synthesis | 60.0M | 20.0M | 80.0M |
| Bug Description | 40.0M | 15.0M | 55.0M |
| Negative Capability | 12.0M | 4.0M | 16.0M |
| Error Recovery L2 | 20.0M | 12.5M | 32.5M |
| **Total** | **132.0M** | **51.5M** | **183.5M** |

---

## Part 4: Cost by Model

### Pricing Assumptions

| Model | Input ($/M tokens) | Output ($/M tokens) | Quality | Speed |
|-------|--------------------|--------------------|---------|-------|
| Claude Haiku | $0.25 | $1.25 | Good for templated tasks | Fast |
| Claude Sonnet | $3.00 | $15.00 | High quality | Medium |
| Qwen3-32B (local) | ~$0 | ~$0 | Moderate | GPU-bound |

### Cost Matrix -- Strict Scenario (42K)

| Model | Input Cost | Output Cost | **Total** |
|-------|-----------|-------------|-----------|
| All Haiku | $14.00 | $27.13 | **$41.13** |
| All Sonnet | $168.00 | $325.50 | **$493.50** |
| All Local (Qwen3-32B) | $0 | $0 | **$0** (GPU time only) |

### Cost Matrix -- Relaxed Scenario (100K)

| Model | Input Cost | Output Cost | **Total** |
|-------|-----------|-------------|-----------|
| All Haiku | $33.00 | $64.38 | **$97.38** |
| All Sonnet | $396.00 | $772.50 | **$1,168.50** |
| All Local (Qwen3-32B) | $0 | $0 | **$0** (GPU time only) |

---

## Part 5: Recommended Strategy

### Model Assignment by Task

| Task | Recommended Model | Rationale |
|------|-------------------|-----------|
| Issue Synthesis | **Haiku** or **Qwen3-32B** | Structured transformation, doesn't need Sonnet-level reasoning |
| Bug Description | **Haiku** or **Qwen3-32B** | Descriptive text from code, moderate complexity |
| Negative Capability | **Qwen3-32B (local)** | Lower quality bar, high volume, save costs |
| Error Recovery L2 | **Qwen3-32B (local)** | Intentionally imperfect outputs -- weaker model is actually *preferable* |
| Quality Audit (5% sample) | **Sonnet** | Verify synthesis quality on random subset |

### Hybrid Strategy Cost Estimate

**Strict (42K) -- Recommended Plan:**

| Task | Model | Input Tokens | Output Tokens | Cost |
|------|-------|-------------|---------------|------|
| Issue Synthesis | Haiku | 25.2M | 8.4M | $16.80 |
| Bug Description | Haiku | 16.8M | 6.3M | $12.08 |
| Negative Capability | Local Qwen3-32B | 6.0M | 2.0M | $0 |
| Error Recovery L2 | Local Qwen3-32B | 8.0M | 5.0M | $0 |
| Quality Audit (5%) | Sonnet | 2.8M | 1.1M | $24.90 |
| **Total** | | | | **$53.78** |

**Relaxed (100K) -- Recommended Plan:**

| Task | Model | Input Tokens | Output Tokens | Cost |
|------|-------|-------------|---------------|------|
| Issue Synthesis | Haiku | 60.0M | 20.0M | $40.00 |
| Bug Description | Haiku | 40.0M | 15.0M | $28.75 |
| Negative Capability | Local Qwen3-32B | 12.0M | 4.0M | $0 |
| Error Recovery L2 | Local Qwen3-32B | 20.0M | 12.5M | $0 |
| Quality Audit (5%) | Sonnet | 6.6M | 2.6M | $58.80 |
| **Total** | | | | **$127.55** |

---

## Part 6: Compute & Infrastructure Costs

### Git Parsing (530K commits across 25 repos)

| Operation | Estimate |
|-----------|----------|
| Clone/fetch all 25 repos | ~5 min (repos already available) |
| Parse all 530K commits (git log + diff) | ~2-4 hours on single core |
| Parallelized (8 workers) | ~20-30 min |
| Filter to candidates | ~5 min (in-memory) |
| Extract patches + context | ~1-2 hours (disk I/O bound) |

**Total git ops time: ~1-3 hours** (with parallelization)

### Local Model GPU Hours (Qwen3-32B)

Assuming Qwen3-32B on available GPUs (~50 tokens/sec output for 32B model):

| Scenario | Output Tokens (local tasks) | GPU Hours |
|----------|----------------------------|-----------|
| Strict | 7.0M tokens | ~39 hours |
| Relaxed | 16.5M tokens | ~92 hours |

With batch processing and larger batch sizes (~150 tok/sec effective):

| Scenario | Effective GPU Hours |
|----------|-------------------|
| Strict | ~13 hours |
| Relaxed | ~31 hours |

**Note**: GPU is already provisioned and running for other tasks. Marginal cost is scheduling/opportunity cost only.

### Disk Space

| Item | Size Estimate |
|------|---------------|
| Raw commit data (530K commits, patches) | ~8-15 GB |
| Filtered candidates (42K-100K) | ~2-5 GB |
| Synthesized QA JSONL output | ~3-8 GB |
| Intermediate processing files | ~2 GB |
| **Total disk needed** | **~15-30 GB** |

Current CFS has ample space for this.

---

## Part 7: Timeline Estimate

| Phase | Duration | Blocking? |
|-------|----------|-----------|
| 1. Git parsing + filtering | 3 hours | No |
| 2. Free task extraction (350K-830K samples) | 4-8 hours | No |
| 3. LLM synthesis (Haiku calls) | 6-12 hours (rate limited) | No |
| 4. Local model synthesis (Qwen3-32B) | 13-31 GPU hours | Shares GPU |
| 5. Quality audit (Sonnet) | 2-4 hours | No |
| 6. Post-processing + formatting | 2 hours | No |

**Total wall-clock: ~2-4 days** (with parallelization and assuming GPU availability)

---

## Summary: Budget Table

| Scenario | API Cost | GPU Hours | Disk | Wall-clock | Total Samples |
|----------|----------|-----------|------|------------|---------------|
| **Strict (42K commits)** | **~$54** | ~13h | ~15 GB | ~2 days | ~350K QA pairs |
| **Relaxed (100K commits)** | **~$128** | ~31h | ~30 GB | ~4 days | ~830K QA pairs |
| All-Sonnet worst case | ~$1,169 | 0h | ~30 GB | ~4 days | ~830K QA pairs |
| All-Local (zero API) | **$0** | ~92h | ~30 GB | ~5 days | ~830K QA pairs |

---

## Recommended Plan

| Priority | Action | Cost |
|----------|--------|------|
| 1 | Start with strict filter (42K), Haiku for synthesis | ~$54 |
| 2 | Run free extraction tasks in parallel (zero cost) | $0 |
| 3 | Use Qwen3-32B for negative capability + error recovery | $0 (GPU) |
| 4 | Quality audit 5% with Sonnet | ~$25 |
| 5 | If quality good, expand to relaxed filter | +$74 |
| **Total recommended budget** | | **$55-130** |

### Key Insight

The vast majority of training data (~80% of samples) comes from **free git extraction** tasks. LLM calls are only needed for issue/bug description synthesis and can be done cheaply with Haiku. The "Error Recovery Level 2" task actually *benefits* from using a weaker local model since we want imperfect attempts.

**Bottom line: Full-scale synthesis across all 25 repos costs $55-130 in API fees, ~13-31 GPU hours (already available), and produces 350K-830K training QA pairs in 2-4 days.**
