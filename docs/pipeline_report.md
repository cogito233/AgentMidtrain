# AgentMidtrain 数据合成 Pipeline 报告

## 1. 项目概述

**目标**: 从 25 个 Python 开源 repo 的 commit 历史中自动合成 SWE agent mid-training 数据，覆盖 localization、edit generation、code review 等多种任务类型。

**数据规模**:
- 原始 commits: ~530K (25 repos)
- Strict filter 后: ~42K candidates (has fix keyword + test + src, <=5 src files)
- Relaxed filter: ~80-100K candidates
- 目标产出: ~350K-450K QA pairs (strict) / ~830K (relaxed)

**核心差异化**: 相比 R2E-Gym/SWE-Gym 的 golden-path-only 方法，本项目额外包含：
- 错误恢复 (Error Recovery) 训练数据
- 负样本训练 (Negative Capability / What NOT to change)
- 多种 issue synthesis 策略对比验证
- TDD simulation 任务格式

---

## 2. Pipeline 设计

### 2.1 三阶段流程

```
parse_commits → filter_commits → generate_tasks
```

| 阶段 | 输入 | 输出 | 关键参数 |
|------|------|------|----------|
| parse_commits | git repo | raw commit JSONL (hash, msg, files, diff) | -- |
| filter_commits | raw JSONL | filtered JSONL (有 test+src patch 的 commits) | `has_test=True, max_src_files=5, fix_keyword=True/False` |
| generate_tasks | filtered JSONL + repo checkout | 多个 task-type JSONL | `--task-types`, `--max-file-lines=500`, `--workers=8` |

### 2.2 v1 到 v2 的改进

| 问题 | v1 表现 | v2 修复 |
|------|---------|---------|
| Localization 重复 | 58% 样本重复（同一 commit 生成多次）| 加入 commit-level 去重 |
| code_review = commit_message | output 直接复用 commit message，无独立内容 | 使用 diff 作为 input，结构化 review 格式 |
| bug_detection output 质量 | 直接输出清洗后的 commit_message | 输出 bug 描述+位置+severity 的结构化格式 |
| commit_message 泄露 ticket ref | "Fixed #37095 --" 前缀保留 | `clean_commit_message()` 函数清除 ticket ref、CVE ref、hash ref |
| localization candidates 不含 target | 部分样本无正确答案 | `include_prob=0.7` 控制，metadata 标注 `targets_included` |
| edit_generation 格式 | raw unified diff | `diff_to_search_replace()` 转换为 SEARCH/REPLACE block 格式 |

### 2.3 关键辅助函数 (v2)

```python
clean_commit_message(msg)      # 去除 "Fixed #NNN --", CVE ref, hash ref
diff_to_search_replace(diff)   # unified diff → SEARCH/REPLACE blocks
sample_candidate_files(...)    # localization 候选文件采样 (n=30, include_prob=0.7)
classify_commit_type(msg)      # 分类 commit: bug_fix/feature/refactor/docs/test/enhancement
```

---

## 3. 任务类型详解（6 种）

### 3.1 Localization

**目的**: 给定 issue 描述 + 文件列表，预测需要修改的文件。

**格式**:
- Input: issue description + 30 个候选文件列表
- Output: 目标文件路径 (或 NOT_FOUND)
- Metadata: `targets_included` (bool), `n_targets`

**示例** (django v2):
```json
{
  "task_type": "localization",
  "prompt": "You are a software engineer tasked with locating the source files...",
  "input": "Issue: Checked maximum redirect lengths against percent-encoded URLs.\n\nCandidate files:\n  1. tests/staticfiles_tests/...\n  ...\n  13. django/http/response.py\n  ...",
  "output": "django/http/response.py",
  "metadata": {"targets_included": true, "n_candidates": 30, "n_targets": 1}
}
```

**预计数据量**: ~80K (1-3 per commit, with negative samples)

---

### 3.2 Edit Generation

**目的**: 给定 issue + 源文件内容，生成修复 patch (SEARCH/REPLACE 格式)。

**格式**:
- Input: issue description + 完整源文件 (带行号)
- Output: SEARCH/REPLACE blocks

**示例** (django v2):
```
<<<<<<< SEARCH
    ):\n        super().__init__(*args, **kwargs)\n        self["Location"] = iri_to_uri(redirect_to)\n        redirect_to_str = str(redirect_to)\n        if max_length is not None and len(redirect_to_str) > max_length:
=======
    ):\n        super().__init__(*args, **kwargs)\n        self["Location"] = iri_to_uri(redirect_to)\n        if max_length is not None and len(self["Location"]) > max_length:
>>>>>>> REPLACE
```

**预计数据量**: ~42K

---

### 3.3 Test Writing

**目的**: 给定 issue + fix patch，预测需要编写的测试。

**格式**:
- Input: issue description + source patch (diff)
- Output: test patch (diff)

**预计数据量**: ~42K

---

### 3.4 Commit Message Generation

**目的**: 给定完整 diff，生成 conventional commit message。

**格式**:
- Input: full diff (src + test changes)
- Output: cleaned commit message

**预计数据量**: ~42K

---

### 3.5 Bug Detection

**目的**: 给定 buggy 代码片段，识别 bug 并描述。

**格式**:
- Input: 源文件代码片段 (修复前)
- Output: 结构化 bug 描述 (Bug + Location + Severity)

**示例** (django v2):
```
Bug: Checked maximum redirect lengths against percent-encoded URLs.
Location: django/http/response.py:642-653
Severity: bug
```

**预计数据量**: ~42K (strict), 部分 commit 产出多个（多文件时）

---

### 3.6 Code Review

**目的**: 给定一个 diff，生成结构化 review。

**格式**:
- Input: unified diff
- Output: Summary + Files changed + Type

**示例** (django v2):
```
Summary: Checked maximum redirect lengths against percent-encoded URLs.
Files changed: django/http/response.py
Type: Bug Fix
```

**预计数据量**: ~42K

---

## 4. 探索性实验结果

### 4.1 Bug Description Synthesis -- 3 种策略对比

在 5 个 Django commit 上对比了 3 种 bug description 合成策略（使用 claude-sonnet-4.5）:

| 策略 | 方法 | 输入 | 输出特点 |
|------|------|------|----------|
| **Strategy 1: Root Cause** | 分析 buggy code 的根本原因 | 修复前源码 + commit msg | 聚焦代码逻辑缺陷，开发者视角 |
| **Strategy 2: User Report** | 模拟用户提交 bug report | 修复前源码 + commit msg | Bug Report 格式，Expected vs Actual，面向用户 |
| **Strategy 3: Code Review** | 模拟 code review 时发现 bug | 修复前源码 + commit msg | 聚焦特定代码行，review 视角 |

**对比示例** (commit 335c6d0129 -- redirect URL length validation):

| 策略 | 核心内容 |
|------|----------|
| Root Cause | "length check 在 percent-encoding 前执行，space 变 `%20` 后长度翻倍" |
| User Report | "我设置 max_length=100，含特殊字符的 90 字符 URL 通过了验证但编码后超限" |
| Code Review | "line 644-650 操作顺序错误：先 encode 再 check length 用了原始字符串" |

**结论和推荐**:
- **Root Cause** 最适合 mid-training (知识密集、深入)
- **User Report** 最适合模拟 SWE-bench issue 格式 (适配 evaluation)
- **Code Review** 最精确 (定位到行号) 但格式与 issue 不匹配
- **推荐**: Issue Synthesis 用 User Report 策略; Bug Detection 任务用 Root Cause 策略

---

### 4.2 Issue Synthesis -- 3 种策略对比

在 10 个 Django commit 上对比了 3 种 issue synthesis 策略（使用 claude-haiku-4.5）:

| 策略 | 方法 | 输入量 | 延迟 | 特点 |
|------|------|--------|------|------|
| **A: Full Context** | commit msg + src files + test files | 高 | ~3.5-4.6s | 最详细，含复现代码，但可能泄露实现细节 |
| **B: Minimal Context** | commit msg + src files (不含 test) | 中 | ~2.9-4.9s | 类似真实 issue report，Expected/Actual 明确 |
| **C: Bug Focused** | commit msg + src files (聚焦 bug 描述) | 中 | ~3.5-5.1s | 更技术化，聚焦 bug mechanism |

**质量对比** (同一 commit: AlterField attname-based to_field):

- **Strategy A**: 给出完整 model 定义 + migration 步骤，最接近 Django ticket 风格
- **Strategy B**: "Steps to reproduce" 格式清晰，无代码但有操作步骤
- **Strategy C**: 直接描述 bug 成因 + 代码示例

**特殊情况**: Strategy B 对 commit `9c655e9800` (无 test patch) 返回 `[SKIPPED]`，设计合理。

**结论和推荐**:
- **Strategy A (Full Context)** 质量最高但 input token 多、成本高
- **Strategy B (Minimal Context)** 最平衡：成本适中、格式标准化
- **推荐**: 量产使用 Strategy B (Haiku); 5% 质量审计用 Strategy A (Sonnet)

---

### 4.3 Commit Chains Detection

从 Django repo 中检测到 **975 条 commit chain** (follow-up 类型)。

**示例 chain** (ticket #36593 -- 5 commits):
```
1. "Deprecated QuerySet.select_related() with no arguments"
2. "Used explicit select_related() in admin history_view()"
3. "Used explicit select_related() calls in tests"
4. "Deprecated setting ModelAdmin.list_select_related to True"
5. "Made ModelAdmin.list_select_related = False more efficient"
```

**共享特征**: `shared_ticket: "36593"`, 相同 module 的渐进式修改。

**应用**:
- Multi-commit planning 任务 (T2 Progressive Refactoring)
- 测试 "is this fix complete?" 判断能力
- 训练模型理解 follow-up 工作模式

---

### 4.4 TDD Simulation

以 "test → fix" 的格式构造任务，模拟 Test-Driven Development。

**格式**:
```json
{
  "task_type": "tdd_simulation",
  "input": "The following test should pass but currently fails:\n```python\ndef test_redirect_url_max_length_checks_encoded_location(self):\n    ...\n```\nRelevant source code (before fix):\n```python\n# django/http/response.py\n...\n```\nFix the source code to make the test pass.",
  "output": "diff --git a/django/http/response.py ...\n-        redirect_to_str = str(redirect_to)\n-        if max_length is not None and len(redirect_to_str) > max_length:\n+        if max_length is not None and len(self[\"Location\"]) > max_length:"
}
```

**特点**:
- 最接近 SWE-bench evaluation 格式
- 提供失败测试 + source code before fix
- 期望输出为 unified diff
- 涵盖 single-file 和 multi-file 修复

**推荐**: 作为 post-training SFT 的核心任务类型，高 transfer value。

---

## 5. 数据质量分析

### 5.1 v1 的问题

| 问题 | 严重度 | 影响 |
|------|--------|------|
| Localization 58% 重复 | 高 | 有效样本量仅为标称的 42% |
| code_review = commit_message | 高 | 任务退化，无独立学习信号 |
| bug_detection 直接输出 commit msg | 中 | 格式不匹配，模型学不到 bug detection 能力 |
| ticket reference 泄露 | 中 | 模型可能学到 "Fixed #XXXXX" pattern 作为捷径 |
| edit_generation 用 raw diff | 低 | 格式不适合 SFT (模型难学 unified diff) |

### 5.2 v2 的改进效果

- **去重**: commit-level dedup，localization 唯一样本率从 42% 提升到接近 100%
- **code_review**: 独立 prompt + 结构化输出 (Summary/Files/Type)，不再等于 commit_message
- **bug_detection**: 结构化 "Bug/Location/Severity" 格式
- **edit_generation**: SEARCH/REPLACE block 格式，更易学习
- **ticket ref 清除**: `clean_commit_message()` 覆盖 Fixed/Refs/Closes/CVE 等模式

### 5.3 去重策略

1. **Commit hash 去重**: 同一 commit 不产出重复样本
2. **SWE-bench 排除**: 与 SWE-bench instance IDs 做 exact match 排除
3. **Localization 30% 负样本**: `targets_included=False` 确保模型学会输出 NOT_FOUND
4. **Output content hash**: 对 edit_generation 输出做 content-based dedup

---

## 6. 预算和规模估计

### 6.1 Token 量估计

| 任务 | Input Tokens (42K) | Output Tokens (42K) | Total |
|------|--------------------|--------------------|-------|
| Issue Synthesis | 25.2M | 8.4M | 33.6M |
| Bug Description | 16.8M | 6.3M | 23.1M |
| Negative Capability | 6.0M | 2.0M | 8.0M |
| Error Recovery L2 | 8.0M | 5.0M | 13.0M |
| **Total** | **56.0M** | **21.7M** | **77.7M** |

### 6.2 API 调用成本

| 方案 | 成本 | 说明 |
|------|------|------|
| **推荐混合方案 (strict 42K)** | **~$54** | Haiku + Local Qwen3-32B + 5% Sonnet audit |
| 推荐混合方案 (relaxed 100K) | ~$128 | 同上，规模翻倍 |
| All-Haiku worst case | $97 | 全量 API 调用 |
| All-Sonnet worst case | $1,169 | 仅用于参考 |
| All-Local (零 API) | $0 | 需 ~92 GPU hours |

### 6.3 存储需求

| 项目 | 大小 |
|------|------|
| Raw commit data (530K) | ~8-15 GB |
| Filtered candidates | ~2-5 GB |
| 合成 QA JSONL | ~3-8 GB |
| **Total** | **~15-30 GB** |

### 6.4 时间估计

| 阶段 | 时长 |
|------|------|
| Git parsing + filtering | ~3h |
| Free task extraction (350K+ samples) | ~4-8h |
| LLM synthesis (Haiku) | ~6-12h (rate limited) |
| Local model synthesis (Qwen3-32B) | ~13h GPU (batch) |
| Quality audit (Sonnet) | ~2-4h |
| **Total wall-clock** | **~2-4 days** |

---

## 7. 下一步计划

### 7.1 剩余 24 个 repo 的批量处理

当前已完成 Django 的全流程验证 (136 条 v2 样本 + 975 条 commit chains)。下一步:

1. 并行对 24 个 repo 执行 `parse_commits` + `filter_commits`
2. 按 repo 特性调整 filter 参数 (如 numpy 可能无 "Fixed" keyword pattern)
3. 统一执行 `generate_tasks` (8 workers per repo)
4. 预计产出 350K+ QA pairs (strict) / 830K+ (relaxed)

### 7.2 Agentless 风格的分层 Localization (Skeleton Step)

当前 localization 是 flat candidate list (30 files)。计划引入分层:

```
Level 1: File skeleton → 从 repo tree 中选择 top-5 relevant modules
Level 2: Function skeleton → 从选中文件中选择 relevant functions
Level 3: Line-level localization → 精确到代码行
```

这与 Agentless 论文的 hierarchical localization 对齐，为 agent 训练提供更细粒度信号。

### 7.3 LLM-based Issue Synthesis 全量部署

基于探索实验结论:
- **默认使用 Strategy B (Minimal Context)** + **Claude Haiku**
- 对 42K strict candidates 全量生成 issue
- 5% 随机抽样用 Sonnet 做质量审计
- 预计成本: ~$17 (Haiku) + ~$25 (Sonnet audit) = ~$42

### 7.4 TDD Simulation 扩展

当前 TDD simulation 仅在 Django 上有少量 sample。计划:
- 扩展到全部 25 repos
- 自动筛选 `has_test_entity_edit` 的 commits
- 作为最接近 SWE-bench 格式的任务，优先级最高

### 7.5 Error Recovery 合成

- **Level 1 (Template-based)**: 利用 commit chains 和 co-change graph 构造 wrong-localization 路径 (~100K)
- **Level 2 (Weak model attempts)**: 用 Qwen3-32B 生成错误修复，构造 fail → correct 序列 (~20K)
- 两层均为零/极低 API 成本

---

## 附录: 关键文件索引

| 文件 | 作用 |
|------|------|
| `scripts/generate_tasks.py` | v2 pipeline 核心，6 种任务类型生成 |
| `data/tasks/django_v2/*.jsonl` | Django v2 样本输出 (136 条) |
| `data/synthesis_exploration/strategy_*.jsonl` | Issue synthesis 3 策略探索 (10 commits each) |
| `data/synthesis_exploration/bug_descriptions/` | Bug description 3 策略探索 (5 commits each) |
| `data/commit_chains_django.jsonl` | Django commit chains (975 chains) |
| `data/tasks/tdd_simulation_django_sample.jsonl` | TDD simulation 样本 |
| `docs/budget_estimate.md` | 详细预算估计 |
| `docs/brainstorm.md` | 完整任务类型 catalog + 方法论 |
