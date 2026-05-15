# AgentMidtrain

从开源 repo 的 commit 历史中合成 SWE mid-training 数据，提升模型在 SWE-bench Verified 上的表现。

## 当前状态

**数据版本**: SQA v5b
**输出路径**: `/data_fast_v3/eremite/cogito_explore/AgentMidtrain/data/training_sqa_v5b/`

| 指标 | 数值 |
|------|------|
| 总样本数（去重后） | 6,835,907 |
| 估算 tokens（tiktoken cl100k_base） | **~13.1B** |
| 去重率 | 35.0% |
| 覆盖 repo 数 | 185 |
| 任务类型 | 6（推荐 4 种） |
| 语言 | Python |

---

## 推荐使用的数据 (4 类)

| 任务类型 | 样本数 | 说明 |
|----------|-------:|------|
| localization | 2,988,000 | 3 步分层文件定位 |
| commit_message | 1,666,112 | diff → 简短 subject line |
| edit_generation | 480,412 | issue + 代码 → SEARCH/REPLACE 编辑 |
| test_writing | 26,942 | issue + patch → 测试代码 |
| **合计** | **5,161,466** | **~9.9B tokens** |

### 不推荐的数据

| 任务类型 | 样本数 | 问题 |
|----------|-------:|------|
| bug_detection | 108,492 | 80.9% 的 answer 包含机械模板短语 |
| code_review | 1,565,881 | 99% 与 commit_message 重复 |

---

## Pipeline 架构

```
clone (bare) → parse_commits.py → filter_commits.py → generate_tasks.py → merge_all_tasks.py → convert_to_sqa.py
```

### 4 层过滤策略

| 级别 | 条件 | 通过率 | 用途 |
|------|------|--------|------|
| strict | require-test + require-src + fix-keywords + python-only + max 5 files + 200 lines | 3-7% | 高质量核心 |
| relaxed | require-src + python-only + exclude-docs + max 8 files + 300 lines | 15-30% | 扩量 |
| ultra | python-only + max 12 files + 500 lines + max-patch-length 20000 | ~47% | 进一步扩量 |
| max | max 12 files + 500 lines + max-patch-length 20000 | ~95% | 最大化覆盖 |

### 6 种任务生成器

1. **localization** (每个 commit 生成 2-3 个样本):
   - Step 1: Directory localization (issue + dir listing → target dirs)
   - Step 2: File localization (issue + file listing → target files)
   - Step 3: Function localization (issue + code skeletons → target classes/functions)
   - ~30% 样本为 NOT_FOUND（negative sample）

2. **edit_generation**: Issue + focused code section → SEARCH/REPLACE blocks
   - 质量门控: 3-25 changed lines, all hunks in window, focused 5-80 lines

3. **test_writing**: Issue + fix summary → 完整 test 代码
   - 语法验证 (compile check), 括号平衡, 断言存在

4. **commit_message**: File paths + stats → clean commit subject line
   - 不展示代码（防止泄露），清洗 ticket refs

5. **bug_detection** (不推荐): Buggy code → Bug/Location/Problematic code

6. **code_review** (不推荐): Diff → Summary/Files/Type/Risk/Suggestions

---

## 数据位置

```
data/
├── training_sqa_v5b/          # 最终 SQA 格式训练数据 (当前版本)
│   ├── localization.jsonl     # 37G, 2,988,000 lines
│   ├── commit_message.jsonl   # 1.4G, 1,666,112 lines
│   ├── edit_generation.jsonl  # 2.0G, 480,412 lines
│   ├── test_writing.jsonl     # 784M, 26,942 lines
│   ├── bug_detection.jsonl    # 不推荐
│   ├── code_review.jsonl      # 不推荐
│   └── all_combined.jsonl     # 全量合并
├── tasks_merged_v5b/          # 去重后的中间格式
├── tasks/                     # 原始生成目录 (488 子目录)
├── filtered_commits/          # 过滤后的 commit JSONL
└── raw_commits/               # 解析后的原始 commit
```

---

## SQA 格式

```json
{
  "system": "You are a code localization expert...",
  "query": "<task-specific input>",
  "answer": "<task-specific output>",
  "task_type": "localization|commit_message|edit_generation|test_writing",
  "repo": "org_name",
  "commit": "sha"
}
```

---

## Quick Start: 添加新 repo

```bash
cd /data_fast_v3/eremite/cogito_explore/AgentMidtrain

# 1. Clone
python scripts/add_repos.py --repos "owner/repo_name"

# 2. Parse commits
python scripts/parse_commits.py --repo-path repos/owner_repo --output data/raw_commits/owner_repo.jsonl

# 3. Filter (max = 最大覆盖)
python scripts/filter_commits.py data/raw_commits/owner_repo.jsonl \
  -o data/filtered_commits/owner_repo_max.jsonl \
  --max-src-files 12 --max-edit-lines 500 --max-patch-length 20000

# 4. Generate tasks
python scripts/generate_tasks.py \
  --input data/filtered_commits/owner_repo_max.jsonl \
  --output-dir data/tasks/owner_repo_max \
  --repo-path repos/owner_repo \
  --task-types localization,edit_generation,commit_message,test_writing \
  --workers 4

# 5. Re-merge all tasks (dedup)
python scripts/merge_all_tasks.py --output-dir data/tasks_merged_v5b

# 6. Convert to SQA format
python scripts/convert_to_sqa.py --input-dir data/tasks_merged_v5b --output-dir data/training_sqa_v5b
```

---

## Scripts

| Script | 功能 |
|--------|------|
| `add_repos.py` | 从 GitHub clone repos (bare) |
| `parse_commits.py` | 提取 commit 数据 (message, patches, file metadata) |
| `filter_commits.py` | 可配置条件的质量过滤 |
| `generate_tasks.py` | 从过滤后的 commits 生成 6 种任务 |
| `batch_process_repos.py` | 批量处理多个 repo 的完整流程 |
| `refilter_relaxed.py` | 批量用 relaxed 条件重新过滤已有 raw commits |
| `merge_all_tasks.py` | 合并所有 task 目录，MD5 去重 |
| `convert_to_sqa.py` | 转换为 {system, query, answer} 训练格式 |
| `dedup_tasks.py` | 独立去重工具 |
| `review_quality.py` | 数据质量审查/统计 |
| `synthesize_issues.py` | issue 描述合成 |
| `synthesize_bug_descriptions.py` | bug 描述合成 |
| `refine_bug_detection.py` | bug_detection 数据精炼 |
| `detect_commit_chains.py` | 检测 commit 链 |
| `prepare_haiku_input.py` | 准备 Haiku 批量推理输入 |

---

## 多语言支持

当前仅覆盖 **Python**。Go 和 TypeScript 的支持正在开发中：
- 需要适配各语言的 parse 机制（文件分类、测试检测逻辑）
- `filter_commits.py` 中的 `--python-only` 标志需替换为对应语言过滤
- Java 计划后续加入

---

## 参考工作

- **R2E-Gym** (Jain et al., 2025): SWE-GEN 从 commit 合成数据
- **SWE-Gym** (Pan et al., 2024): 2438 task instances, SFT + Verifier
- **Agentless** (Xia et al., 2024): localization → repair → test pipeline
- **SWE-bench**: https://github.com/princeton-nlp/SWE-bench
