# AgentMidtrain

SWE Agent Mid-training 数据合成：从开源 repo 的 commit 历史中合成 mid-training 数据，提升模型在 SWE-bench Verified 上的表现。

## 当前状态 (2025-05-15)

**已完成 5B+ token 目标**

| 指标 | 数值 |
|------|------|
| 总样本数 | 3,283,985 |
| 估计 tokens（tiktoken cl100k_base） | **5.13B** |
| 去重率 | 32% |
| 覆盖 repo 数 | 158 |
| 任务类型 | 6 |
| 数据总大小 | 41G (SQA format) |

### 任务类型分布

| Task Type | Samples | % | 平均答案长度 |
|-----------|---------|---|-------------|
| localization | 1,540,671 | 46.9% | 95 chars |
| commit_message | 735,761 | 22.4% | 40 chars |
| code_review | 695,259 | 21.2% | 261 chars |
| edit_generation | 243,099 | 7.4% | 1033 chars |
| bug_detection | 49,532 | 1.5% | 185 chars |
| test_writing | 19,663 | 0.6% | 1082 chars |

### Top Repos

sympy (6.7%), cpython (5.4%), sentry (4.5%), openstack/nova (4.3%), django (4.3%), ansible (3.2%), awx (2.4%), airflow (2.3%), astropy (2.1%), Theano (2.1%)

## 数据位置

```
data/
├── training_sqa_v3/          # 最终 SQA 格式训练数据 (最新)
│   ├── all_combined.jsonl    # 合并文件 (21G, 3.28M lines)
│   ├── localization.jsonl    # 3-step hierarchical localization
│   ├── edit_generation.jsonl # SEARCH/REPLACE format output
│   ├── test_writing.jsonl    # Complete test code output
│   ├── commit_message.jsonl  # Clean commit message
│   ├── bug_detection.jsonl   # Bug/Location/Problematic code
│   └── code_review.jsonl     # Summary/Files/Type/Risk/Suggestions
├── tasks_merged_v3/          # 去重后的中间格式
├── tasks/                    # 原始生成目录 (362 子目录)
├── filtered_commits/         # 过滤后的 commit JSONL
└── raw_commits/              # 解析后的原始 commit
```

## SQA 格式

```json
{
  "system": "You are a software engineer...",
  "query": "Issue description + code context...",
  "answer": "Expected output (edits/paths/message/review)...",
  "task_type": "edit_generation",
  "repo": "django_django",
  "commit": "abc123..."
}
```

## Pipeline

```
clone repos → parse_commits.py → filter_commits.py → generate_tasks.py → merge_all_tasks.py → convert_to_sqa.py
```

### 4-Layer Filtering Strategy

| Layer | Flags | Pass Rate |
|-------|-------|-----------|
| strict | --require-test --require-src --fix-keywords --python-only --max-src-files 5 --max-edit-lines 200 | ~3-7% |
| relaxed | --require-src --python-only --exclude-docs --max-src-files 8 --max-edit-lines 300 | ~15-30% |
| ultra | --python-only --max-src-files 12 --max-edit-lines 500 --max-patch-length 20000 | ~47% |
| max | --max-src-files 12 --max-edit-lines 500 --max-patch-length 20000 | ~95% |

### 6 Task Generators

1. **localization** (3 independent samples per commit):
   - Step 1: Directory localization (issue + dir listing → target dirs)
   - Step 2: File localization (issue + file listing in candidate dirs → target files)
   - Step 3: Function localization (issue + code skeletons → target classes/functions)
   - 10% of samples: NOT_FOUND (target excluded from candidates)

2. **edit_generation**: Issue + focused code section → SEARCH/REPLACE blocks
   - Quality gates: 3-25 changed lines, all hunks in window, focused 5-80 lines

3. **test_writing**: Issue + fix summary → complete test code with imports
   - Syntax-verified (compile check), balanced delimiters, assertion present

4. **commit_message**: File paths + stats → clean commit subject line
   - No code shown (prevents leakage), no ticket refs

5. **bug_detection**: Buggy code (60 lines context) → Bug/Location/Problematic code
   - Only bug_fix commits, single-file, logic bugs only

6. **code_review**: Diff → Summary/Files/Type/Risk/Suggestions
   - Structured output differentiated from commit_message

## 使用方式

### 重新生成（如果后台进程完成后需要更新）

```bash
cd /data_fast_v3/eremite/cogito_explore/AgentMidtrain

# Re-merge all task directories (dedup)
python scripts/merge_all_tasks.py --output-dir data/tasks_merged_v3

# Convert to SQA format
python scripts/convert_to_sqa.py --input-dir data/tasks_merged_v3 --output-dir data/training_sqa_v3
```

### 添加新 repo

```bash
# 1. Clone
python scripts/add_repos.py --repos "owner/repo_name"

# 2. Parse commits
python scripts/parse_commits.py --repo-path repos/owner_repo --output data/filtered_commits/owner_repo_raw.jsonl

# 3. Filter (max = most inclusive)
python scripts/filter_commits.py data/filtered_commits/owner_repo_raw.jsonl \
  -o data/filtered_commits/owner_repo_max.jsonl \
  --max-src-files 12 --max-edit-lines 500 --max-patch-length 20000

# 4. Generate tasks
python scripts/generate_tasks.py \
  --input data/filtered_commits/owner_repo_max.jsonl \
  --output-dir data/tasks/owner_repo_max \
  --repo-path repos/owner_repo \
  --task-types localization,edit_generation,commit_message,code_review,bug_detection \
  --workers 4
```

## Scripts

| Script | Purpose |
|--------|---------|
| `add_repos.py` | Clone repos from GitHub |
| `parse_commits.py` | Extract commit data (message, patches, file metadata) |
| `filter_commits.py` | Quality filter with configurable criteria |
| `generate_tasks.py` | Generate 6 task types from filtered commits |
| `refilter_relaxed.py` | Batch re-filter existing raw commits with relaxed criteria |
| `merge_all_tasks.py` | Merge all task dirs, deduplicate by MD5 hash |
| `convert_to_sqa.py` | Convert to {system, query, answer} training format |

## 参考工作

- **R2E-Gym** (Jain et al., 2025): SWE-GEN 从 commit 合成数据
- **SWE-Gym** (Pan et al., 2024): 2438 task instances, SFT + Verifier
- **Agentless** (Xia et al., 2024): localization → repair → test pipeline
- **SWE-bench**: https://github.com/princeton-nlp/SWE-bench
