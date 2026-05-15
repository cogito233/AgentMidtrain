# Agent Mid-training: Brainstorm -- Beyond Agentless Decomposition

> 目标: 从 25 个 Python 开源 repo (~530K commits, ~42K 高质量候选) 中最大化提取 mid-training 信号，超越现有 R2E-Gym/SWE-Gym 的范式。

---

## 1. Full Task Type Catalog

### 1.1 Baseline Tasks (Agentless-style, 已规划)

| ID | Task Type | Input | Output | Format |
|----|-----------|-------|--------|--------|
| B1 | Localization | issue + repo tree | file/function list | QA |
| B2 | Edit Generation | localized files + issue | unified diff patch | QA |
| B3 | Test Writing | issue + fix patch | test code | QA |
| B4 | Issue Synthesis | commit diff | natural language issue | QA |

### 1.2 Extended Task Types -- Code Understanding & Reasoning

| ID | Task Type | Input | Output | Format | Source Signal |
|----|-----------|-------|--------|--------|---------------|
| E1 | **Root Cause Analysis** | failing test + traceback + repo context | explanation of bug mechanism + location | Long-form QA | 从 commit message 中提取 "because..." 模式 |
| E2 | **Impact Analysis** | proposed change (diff) | list of affected modules/tests/APIs | Structured JSON | 通过分析 commit 实际影响的文件范围 vs 修改文件 |
| E3 | **Code Review** | diff + repo conventions | review comments (approve/request changes) | Multi-turn | 从 PR review comments 提取 (如 repo 有 PR 数据) |
| E4 | **Dependency Resolution** | import graph + error | correct import path / package version | QA | 从修复 import error 的 commits 提取 |
| E5 | **API Migration** | deprecated API usage + changelog | updated code | Code transform | 从 deprecation-handling commits 提取 |
| E6 | **Commit Message Generation** | diff | conventional commit message | QA | 直接从 commit 历史 |
| E7 | **Change Scope Prediction** | issue description | "this change should touch N files, M functions, ~L lines" | Structured | 从实际 commit 统计反推 |

### 1.3 Extended Task Types -- Agent Behavior & Planning

| ID | Task Type | Input | Output | Format | Source Signal |
|----|-----------|-------|--------|--------|---------------|
| A1 | **Multi-step Planning** | complex issue | ordered plan (steps 1-N with file targets) | Structured plan | 从 multi-file commits 逆向拆解 |
| A2 | **When to Stop** | partial fix + test results | decision: "done" vs "need more changes" + reasoning | Classification + reasoning | 构造 partial-fix 场景 |
| A3 | **What NOT to Change** | issue + full codebase context | list of tempting-but-wrong files to modify | Negative QA | 从 localization 的负样本构造 |
| A4 | **Verification Strategy** | fix patch | "how to verify this works" (test plan) | QA | 从 commit 中 test 部分反推 |
| A5 | **Rollback Decision** | failed attempt trace + error | "should I rollback and try different approach?" | Classification | 合成 (见 Section 3) |
| A6 | **Context Gathering** | vague issue | "what files/docs should I read first?" | Action sequence | 从成功 agent trajectories 提取 |
| A7 | **Edit Ordering** | multi-file change | correct order to apply edits (dependency-aware) | Ordered list | 从 commit 中文件依赖分析 |

### 1.4 Extended Task Types -- Repo-level Knowledge

| ID | Task Type | Input | Output | Format | Source Signal |
|----|-----------|-------|--------|--------|---------------|
| R1 | **Convention Inference** | N example files from repo | coding conventions (naming, structure, patterns) | Structured doc | 统计分析 repo 的 style 模式 |
| R2 | **Architecture Q&A** | "how does module X work?" | explanation with key files/classes | Long QA | 从 module docstrings + structure 合成 |
| R3 | **Test Pattern Matching** | source module | test file location + test style | QA | 从 repo 的 src→test 映射关系学习 |
| R4 | **Related Change Prediction** | "I changed file A" | "you probably also need to change files B, C" | Prediction | 从 co-change 统计 (files frequently modified together) |
| R5 | **Version-aware Coding** | task + repo version (tag/date) | code using APIs available at that version | Code gen | 利用 git history 的时序性 |

### 1.5 Extended Task Types -- Cross-commit & Temporal

| ID | Task Type | Input | Output | Format | Source Signal |
|----|-----------|-------|--------|--------|---------------|
| T1 | **Bug Reintroduction Detection** | code region + history | "this pattern caused bug X before" | Classification | 从 revert commits / fix-then-regress 模式 |
| T2 | **Progressive Refactoring** | large refactor goal | sequence of safe incremental steps | Multi-step plan | 从 PR 中多 commit 序列提取 |
| T3 | **Merge Conflict Resolution** | two conflicting diffs | resolved merge | Code | 从 merge commits 提取 |
| T4 | **Regression Root Cause** | "test X started failing at commit Y" | which change in Y caused it | QA | 从 bisect-like patterns |

---

## 2. Novel Approaches Not Covered by Existing Work

### 2.1 Contrastive Learning from Commit Pairs

**核心思路**: 不止学"正确答案"，同时学"为什么其他方案不对"。

**方法**:
- 对同一个 bug/issue，构造 (correct_fix, plausible_but_wrong_fix) 对
- 来源: (a) 同一区域的不同 commits 对比; (b) 模型生成的错误修复; (c) reverted commits
- **格式**: `{issue, candidate_fixes: [{patch, is_correct, reasoning}]}`

**与现有工作区别**: R2E-Gym 只用 verifier 做 pass/fail 判断，这里直接将对比信号编码进训练数据。

### 2.2 Negative Capability Training ("What NOT to do")

**核心思路**: SWE agents 的常见失败模式是 over-editing（改了不该改的）。从 commit 历史中提取 "scope control" 信号。

**方法**:
1. 对每个 commit，获取修改文件列表 F_modified
2. 获取"诱惑"文件列表 F_tempting: 与修改文件在同 module、被同测试覆盖、名称相似但实际不需改动的文件
3. 构造 negative samples: `{issue, correct_scope: F_modified, incorrect_scope: F_modified + F_tempting, explanation}`

**格式**:
```json
{
  "issue": "...",
  "repo_tree": "...",
  "candidates": [
    {"file": "models/query.py", "should_edit": true, "reason": "contains the QuerySet logic"},
    {"file": "models/base.py", "should_edit": false, "reason": "base class unchanged, bug is in subclass"}
  ]
}
```

### 2.3 Synthetic Error Recovery Trajectories

**核心思路**: 解决 commit-based 数据缺乏"失败→恢复"路径的根本缺陷。

**方法** (3层难度):

**Level 1 -- Mechanical Detours** (低成本, 高规模):
- 在正确 localization 前插入 1-2 个错误 localization step
- 模板化: `search wrong_file → read → realize not relevant → search correct_file`
- 可从 co-change 统计中选择"相关但非目标"的文件

**Level 2 -- Plausible Wrong Fixes** (中等成本):
- 用弱模型 (e.g. Qwen2.5-7B) 对 issue 生成修复
- 用 test suite 验证失败
- 构造: `attempt_1 (fail) → analyze_error → attempt_2 (correct)`
- 关键: 保留弱模型的"推理过程"作为反面教材

**Level 3 -- Full Rollback Scenarios** (高成本, 需执行环境):
- 在 Docker 中重放: model attempt → run tests → observe failure → backtrack → correct attempt
- 类似 R2E-Gym 但专注收集 failure trajectories 而非 success-only

**与现有工作区别**:
- SWE-Gym 的 rejection sampling 只保留成功 trajectory, 丢弃失败信息
- R2E-Gym 用 RL 隐式学习错误恢复, 不显式编码
- 我们的方法将错误恢复显式编码为训练数据

### 2.4 Repository State Machine (Repo-as-Context)

**核心思路**: 不单独看每个 commit，而是将 repo 建模为随时间演进的状态机，学习 "given the repo at state S, action A is appropriate"。

**方法**:
- 对每个 repo，构建 commit 序列的 "state transitions"
- 每个训练样本包含: repo state summary (key files, recent changes, open issues) + appropriate next action
- 利用 commit 时序: "在加入 feature X 之后, 通常需要 update Y"

**格式**:
```json
{
  "repo_state": {
    "recent_commits": ["added caching layer", "refactored DB models"],
    "open_issues_summary": "...",
    "last_modified_modules": ["cache/", "db/models/"]
  },
  "task": "fix cache invalidation when model updates",
  "context_files_to_read_first": ["cache/invalidation.py", "db/signals.py"],
  "reasoning": "given recent caching addition, invalidation logic likely in new cache module"
}
```

### 2.5 Hierarchical Commit Decomposition

**核心思路**: 大 commit (5+ files) 通常包含多个逻辑子任务。拆解它们可以训练 planning 能力。

**方法**:
1. 识别 multi-file commits with clear sub-components (via AST analysis of modified entities)
2. 用 LLM 将一个大 commit 分解为 ordered sub-tasks
3. 验证: 每个 sub-task 单独应用后 repo 仍能 pass 相关 tests (或至少 parse)
4. 训练数据: `{large_issue} → {plan: [subtask_1, subtask_2, ...], order_constraints: [...]}`

### 2.6 Cross-repo Pattern Transfer

**核心思路**: 不同 repo 中相似的 bug 模式（off-by-one, null check missing, race condition）可以泛化。

**方法**:
- 对 42K 候选 commits 进行 bug pattern clustering
- 每个 cluster 内跨 repo 构造 few-shot examples
- 格式: `{pattern_description, examples: [{repo_A_instance}, {repo_B_instance}], new_instance_to_solve}`

**Pattern categories**:
- Type errors / None handling
- Boundary conditions (off-by-one, empty collection)
- Concurrency issues
- API contract violations
- Configuration/default value bugs

### 2.7 Test-Driven Development Simulation

**核心思路**: 反转 "fix → test" 为 "test → fix"，模拟 TDD 工作流。

**方法**:
- 对 has_test_entity_edit commits: 将 test changes 作为 input specification
- 任务: given failing test (the new/modified test from commit), write the fix that makes it pass
- 这更接近 SWE-bench 的实际 evaluation format

**格式**:
```json
{
  "failing_test": "def test_queryset_filter_null(): ...",
  "test_error": "AssertionError: expected [...] got [...]",
  "repo_context": "relevant source files",
  "expected_fix": "the commit's source changes"
}
```

### 2.8 Agentic Tool Use Grounding

**核心思路**: Mid-training 不仅是 code generation，还需要 grounding 模型到 tool-use patterns (file search, shell commands, etc.)。

**方法**:
- 从 commit 反推: "要完成这个修改，agent 需要执行哪些 tool calls?"
- 构造 ground-truth tool-use traces:
  - `grep "pattern" → find relevant files`
  - `read file:line_range → understand context`
  - `edit file → apply change`
  - `bash: python -m pytest test_file → verify`
- 关键: 每个 tool call 的 observation 都是 deterministic (可以在 repo snapshot 中重放)

**格式** (multi-turn tool-use):
```json
{
  "messages": [
    {"role": "user", "content": "Fix issue: ..."},
    {"role": "assistant", "content": "I'll search for relevant code.", "tool_calls": [{"name": "grep", "args": {"pattern": "...", "path": "..."}}]},
    {"role": "tool", "content": "file.py:42: matching line..."},
    {"role": "assistant", "content": "Found the issue. Let me read the full function.", "tool_calls": [{"name": "read", "args": {"file": "...", "lines": "35-60"}}]},
    {"role": "tool", "content": "..."},
    {"role": "assistant", "content": "The bug is... I'll fix it.", "tool_calls": [{"name": "edit", "args": {"file": "...", "old": "...", "new": "..."}}]}
  ]
}
```

### 2.9 Incremental Context Disclosure

**核心思路**: 训练模型处理"信息逐步揭露"的场景——初始信息不足时，学会主动收集更多上下文。

**方法**:
- 将 issue 描述分层: Level 0 (vague) → Level 1 (with error msg) → Level 2 (with failing test) → Level 3 (with stack trace)
- 在每层训练模型: "given this level of info, what's the best next action?"
- 来源: 从 commit message 合成不同详细程度的 issue 描述

### 2.10 Multi-commit Bug Fix Chains

**核心思路**: 现实中复杂 bug 经常需要多次 commit 才能完全修复 (initial fix → edge case → follow-up)。

**方法**:
- 识别 commit chains: 连续 commits 修改相同文件/函数, message 中有 "follow-up", "also fix", "edge case"
- 构造 multi-turn task: 每次只展示部分问题, 逐步完善修复
- 训练模型预测 "is this fix complete or are there edge cases?"

---

## 3. Data Formats Beyond QA

### 3.1 Format Taxonomy

| Format | Description | Suitable For | Mid-train / Post-train |
|--------|-------------|--------------|------------------------|
| **QA** | Input→Output pair | Localization, Edit Gen | Both |
| **Multi-turn conversation** | User↔Assistant with observations | Agent trajectories | Post-train |
| **Tool-use traces** | Thought→Action→Observation chains | Agentic behavior | Post-train |
| **Preference pairs** | (chosen, rejected) for DPO | Scope control, style | Post-train |
| **Completion** | Long-form document continuation | Code understanding, repo conventions | Mid-train |
| **Fill-in-the-middle** | prefix + suffix → middle | Edit generation, API usage | Mid-train |
| **Classification + reasoning** | Input → label + explanation | Impact analysis, scope decisions | Both |
| **Structured prediction** | Input → JSON/structured output | Plans, file lists, dependency graphs | Both |
| **Contrastive** | (anchor, positive, negative) | Bug pattern similarity | Mid-train |

### 3.2 Mid-training Specific Formats

对 mid-training (continued pre-training) 阶段，格式应偏向 "知识注入" 而非 "指令遵循":

**Format M1 -- Annotated Code Walkthrough**:
```
# File: django/db/models/query.py
# Module: QuerySet filtering
# Pattern: Lazy evaluation with deferred SQL construction

class QuerySet:
    def filter(self, **kwargs):
        # KEY INSIGHT: filter() clones the queryset rather than mutating
        # This enables chaining: qs.filter(a=1).filter(b=2)
        clone = self._clone()
        clone.query.add_q(Q(**kwargs))
        return clone
```

**Format M2 -- Commit-as-Document**:
```
## Change Summary
Repository: django/django
Module: django.db.models.query
Type: Bug fix (boundary condition)

## Problem
When QuerySet.filter() receives an empty Q object, it incorrectly
returns an empty queryset instead of the unfiltered original.

## Root Cause
The `add_q()` method short-circuits on empty Q without checking
if the Q is negated (`~Q()`), which should match all objects.

## Fix
Check `q.negated` before short-circuiting in `add_q()`.

## Diff
[actual diff here]

## Verification
Test: tests/queries/test_q.py::test_empty_negated_q
```

**Format M3 -- Repository Knowledge Base** (per-repo pre-training):
```
# Django ORM Architecture

## Core Concepts
- QuerySet: lazy, chainable, cloning-based
- Q objects: composable query predicates
- Manager: interface between Model class and QuerySet

## File Layout
- django/db/models/query.py: QuerySet implementation
- django/db/models/sql/query.py: SQL compilation
- django/db/models/sql/compiler.py: DB-specific SQL generation

## Common Bug Patterns in This Module
1. Forgetting clone() → mutates shared state
2. Empty Q handling edge cases
3. Deferred field resolution timing
```

---

## 4. Addressing the Error Recovery Gap (Detailed)

### 4.1 Taxonomy of Error Types in SWE Tasks

| Error Type | Frequency | Can Synthesize from Commits? | Method |
|------------|-----------|------------------------------|--------|
| Wrong file localization | Very High | Yes | Negative sampling from co-change graph |
| Wrong function within correct file | High | Yes | Sibling function confusion |
| Correct location, wrong fix | High | Partially | Weak model attempts |
| Incomplete fix (missing edge case) | Medium | Yes | Partial commit application |
| Fix breaks other tests | Medium | Yes | Apply fix + run expanded test suite |
| Over-engineering (too many changes) | Medium | Yes | Superset-of-correct diffs |
| Syntax/parse errors | Low | No (too trivial) | -- |

### 4.2 Synthesis Strategy by Error Type

**Wrong Localization → Recovery**:
```json
{
  "trajectory": [
    {"thought": "Issue mentions 'filter', let me check views.py", "action": "read views.py:filter_view"},
    {"observation": "This handles HTTP filtering, not QuerySet filtering"},
    {"thought": "Wrong module. The issue is about ORM queries, should check models/query.py", "action": "read models/query.py:filter"},
    {"observation": "Found the relevant code..."}
  ],
  "label": "recovery_from_wrong_localization"
}
```

**Incomplete Fix → Expand**:
- Take a commit that fixes function A
- Apply only half the changes (e.g., main logic but not edge case handling)
- Run tests → observe which tests fail
- Construct trajectory: "initial fix passes basic tests, but test_edge_case fails → need to also handle X"

### 4.3 Cost-Efficient Error Recovery Data Pipeline

```
Phase 1 (Zero-cost): Template-based detours from commit metadata
  → ~100K samples, low quality, good for mid-train warmup

Phase 2 (Low-cost): Weak model attempts without execution
  → Qwen2.5-7B generates fixes, human/LLM judges correctness
  → ~20K samples, medium quality

Phase 3 (Medium-cost): Execution-verified trajectories
  → Docker environment, run tests, capture real error messages
  → ~5K samples, high quality, best for post-train SFT

Phase 4 (High-cost): RL exploration data
  → Full R2E-Gym style, multiple attempts per problem
  → ~2K problems × 10 attempts = 20K trajectories
```

---

## 5. Mid-training vs Post-training Considerations

### 5.1 Key Differences

| Dimension | Mid-training (Continued Pre-train) | Post-training (SFT/RLHF) |
|-----------|-------------------------------------|---------------------------|
| **Goal** | Inject domain knowledge, code understanding | Align to instruction format, tool use |
| **Format** | Free-form text, code completion, documents | Structured conversations, tool traces |
| **Scale** | Large (100K-1M samples OK) | Smaller (5K-50K high quality) |
| **Quality bar** | Moderate (bulk knowledge > precision) | High (format correctness critical) |
| **Focus** | "Know" (facts, patterns, conventions) | "Do" (execute tasks correctly) |

### 5.2 Task Type → Training Stage Mapping

**Best for Mid-training** (knowledge injection):
- R1 (Convention Inference) -- learn repo-specific patterns
- R2 (Architecture Q&A) -- understand code structure
- R5 (Version-aware Coding) -- temporal knowledge
- E6 (Commit Message Gen) -- code understanding
- Format M1/M2/M3 (annotated walkthroughs, commit-as-document)
- Cross-repo pattern clusters (bug pattern knowledge)
- T1 (Bug Reintroduction Detection) -- historical patterns

**Best for Post-training SFT** (behavior alignment):
- B1-B4 (Baseline Agentless tasks)
- A1-A7 (Agent behavior tasks)
- 2.3 (Error Recovery Trajectories)
- 2.8 (Agentic Tool Use Grounding)
- Format: multi-turn, tool-use traces

**Best for Post-training RL/DPO** (preference optimization):
- 2.1 (Contrastive Learning)
- 2.2 (Negative Capability)
- A2 (When to Stop) -- preference over stopping points
- Preference pairs from multi-attempt sampling

### 5.3 Curriculum Strategy

```
Stage 1: Mid-train on repo knowledge (Format M1-M3)
  → Model learns: code structure, patterns, conventions
  → Scale: 500K-1M tokens/repo × 25 repos

Stage 2: Mid-train on task knowledge (QA format)
  → Model learns: localization, edit patterns, test patterns
  → Scale: 42K commits × 4 task types = ~168K samples

Stage 3: Post-train SFT on agent trajectories
  → Model learns: multi-step planning, tool use, error recovery
  → Scale: 10K-30K high-quality trajectories

Stage 4: Post-train RL/DPO on preferences
  → Model learns: scope control, when to stop, quality
  → Scale: 5K-10K preference pairs
```

---

## 6. Leveraging Full Repo Context

### 6.1 Co-change Analysis

从 commit 历史中挖掘 "files that change together" 的统计规律:

```python
# 伪代码
for repo in repos:
    co_change_matrix = defaultdict(Counter)
    for commit in repo.commits:
        files = commit.modified_files
        for f1, f2 in combinations(files, 2):
            co_change_matrix[f1][f2] += 1

# 应用: 给定修改了 file A, 预测还需修改哪些 files
# 作为 Related Change Prediction (R4) 的 ground truth
```

### 6.2 Module-level Summarization

为每个 repo 的每个 top-level module 生成 "module card":
- Purpose (from docstrings + README)
- Key classes/functions
- Dependencies (import graph)
- Common modification patterns (from commit history)
- Test coverage mapping (source → test file)

### 6.3 Historical Bug Hotspots

统计每个文件/函数的 bug-fix commit 频率:
- High-frequency bug locations → extra attention in localization training
- 训练数据: "this function has been modified 47 times for bug fixes, common patterns include..."

### 6.4 API Evolution Tracking

追踪 API signatures 随时间的变化:
- Function parameter additions/removals
- Return type changes
- Deprecation → removal timeline
- 用于训练 Version-aware Coding (R5)

---

## 7. Cross-repo Patterns and Transfer Learning

### 7.1 Universal SWE Patterns (repo-agnostic)

| Pattern | Description | Training Signal |
|---------|-------------|-----------------|
| Defensive null check | Add `if x is None` guards | Commits adding None checks |
| Error propagation fix | Proper exception handling/re-raise | Commits modifying except blocks |
| Boundary condition | Off-by-one, empty list, first/last | Commits to loop/index logic |
| Import resolution | Fix circular/missing imports | Commits modifying import statements |
| Config default fix | Wrong default values | Commits changing default params |
| Thread safety | Add locks/atomic operations | Concurrency-related commits |

### 7.2 Framework-specific Transfer

```
Django patterns → applicable to Flask (web framework)
NumPy patterns → applicable to SciPy (array computing)
pytest patterns → applicable to all repos (testing)
```

### 7.3 Abstract Bug Templates

从具体 commits 抽象出 "bug templates":
```
Template: "Method M returns stale state because it doesn't invalidate cache after operation O"
Instances:
  - Django: QuerySet cache not invalidated after delete()
  - pandas: DataFrame cache not cleared after inplace operation
  - matplotlib: Figure renderer cache stale after axis modification
```

---

## 8. Teaching "When to Stop" / "What NOT to Change"

### 8.1 Scope Boundary Training

**Over-edit detection**: 构造"正确修复 + 额外不必要修改"的负例
```json
{
  "issue": "Fix: filter() should handle empty Q objects",
  "minimal_correct_patch": "2 lines changed in query.py",
  "over_engineered_patch": "2 lines in query.py + 15 lines refactoring + new utility function",
  "label": "over_engineered",
  "reasoning": "The refactoring is not necessary for this fix and increases risk"
}
```

### 8.2 Completion Signals

训练模型识别 "fix is complete" 的信号:
- All relevant tests pass
- No new warnings introduced
- Change is minimal and focused
- Edge cases covered (as evidenced by test additions in commit)

### 8.3 "Red Herring" Resistance

构造包含误导信息的 issue:
- Issue mentions module A but bug is in module B (A just surfaces the symptom)
- Stack trace points to framework code but fix is in user code
- Error message suggests type X but root cause is type Y

---

## 9. Practical Considerations

### 9.1 Cost Estimation

| Approach | Compute Cost | LLM API Cost | Human Effort | Scale |
|----------|-------------|--------------|--------------|-------|
| Baseline (B1-B4) | Low | $500-1000 (issue synthesis) | Minimal | 42K × 4 = 168K samples |
| Negative sampling (2.2) | Low | $0 (rule-based) | Minimal | 42K samples |
| Co-change analysis (6.1) | Low | $0 (git log parsing) | Minimal | All 530K commits |
| Repo knowledge (M1-M3) | Low | $200-500 | Minimal | 25 repos |
| Weak model attempts (4.3 Phase 2) | Medium | $1000-2000 | Low | 20K samples |
| Template detours (4.3 Phase 1) | Low | $0 (template-based) | Medium (template design) | 100K samples |
| Cross-repo clustering (7.3) | Medium | $500-1000 | Medium (validation) | 5K patterns |
| Execution-verified (4.3 Phase 3) | High (Docker) | $500 | Low | 5K samples |
| Full RL trajectories (4.3 Phase 4) | Very High | $5000+ | Low | 20K trajectories |

### 9.2 Difficulty & Risk Assessment

| Approach | Implementation Difficulty | Risk of Low Quality | Novelty vs Prior Work |
|----------|--------------------------|--------------------|-----------------------|
| Baseline | Low (well-established) | Low | None (R2E-Gym covered) |
| Negative Capability (2.2) | Low-Medium | Low | Medium |
| Error Recovery Synthesis (2.3) | Medium-High | Medium (synthetic may not reflect real errors) | High |
| Repo State Machine (2.4) | High | High (hard to validate) | High |
| Hierarchical Decomposition (2.5) | Medium | Medium | Medium |
| Tool-use Grounding (2.8) | Medium | Low (deterministic validation) | Medium-High |
| Cross-repo Transfer (2.6) | Medium | Medium | Medium |
| Contrastive (2.1) | Medium | Low | Medium |

### 9.3 Data Contamination Risks

- SWE-bench instances MUST be excluded from training (exact commit matching)
- Issues: some repos' test data is in public datasets
- Mitigation: strict commit hash deduplication against SWE-bench instance IDs
- Additional: time-based split (only use commits before SWE-bench creation date for validation)

---

## 10. Recommended Priority Ordering

### Tier 1: High Impact, Low Cost (Do First)

1. **B1-B4 Baseline** -- 必须做, 作为 foundation
2. **2.2 Negative Capability / Scope Control** -- 低成本, 解决真实痛点 (over-editing)
3. **6.1 Co-change Analysis + R4 Related Change Prediction** -- 纯 git analysis, 零 API 成本
4. **2.8 Tool-use Grounding** -- 从 commit 构造 tool-use traces, 可 deterministic 验证
5. **E6 Commit Message Generation** -- 直接可用, 提升 code understanding

### Tier 2: High Impact, Medium Cost (Do Second)

6. **2.3 Error Recovery (Level 1 + 2)** -- 最大差异化优势, template + weak model
7. **2.7 TDD Simulation** -- 格式最接近 SWE-bench evaluation, 高 transfer value
8. **2.1 Contrastive Learning** -- DPO-ready data, 提升 scope precision
9. **M1-M3 Repo Knowledge Formats** -- mid-training knowledge injection
10. **A1 Multi-step Planning** -- 从 multi-file commits 拆解

### Tier 3: Medium Impact, Higher Cost (If Resources Allow)

11. **2.5 Hierarchical Decomposition** -- 需要 AST 分析 + LLM validation
12. **2.6 Cross-repo Pattern Transfer** -- 需要 clustering + manual validation
13. **2.9 Incremental Context Disclosure** -- 需要 LLM 合成不同 detail levels
14. **T2 Progressive Refactoring** -- 需要 multi-commit chain identification
15. **E2 Impact Analysis** -- 需要 dependency graph construction

### Tier 4: High Novelty, High Cost (Research Exploration)

16. **2.4 Repo State Machine** -- 最新颖但最难验证效果
17. **2.3 Error Recovery (Level 3 + 4)** -- 需要 Docker execution environment
18. **T3 Merge Conflict Resolution** -- 需要 merge commit parsing
19. **2.10 Multi-commit Bug Fix Chains** -- 需要 commit chain identification + validation

---

## 11. Quick Wins: What to Implement This Week

Assuming access to 25 repos and basic LLM API:

1. **Run co-change analysis** on all 530K commits → immediate signal for R4
2. **Parse filtered 42K commits** into B1-B4 format → baseline data
3. **Construct negative localization samples** from co-change graph → 2.2
4. **Generate tool-use traces** from diffs (deterministic) → 2.8
5. **Cluster bug patterns** across repos (embedding-based) → 2.6 starter

---

## 12. Key Insight Summary

三个最重要的 insight:

1. **Error Recovery is the #1 gap**: 所有现有工作 (R2E-Gym, SWE-Gym, Agentless) 都主要从 golden-path 数据训练。显式编码错误恢复是最大差异化机会。

2. **Scope Control is undervalued**: SWE agent 的失败不仅是"改错"，更多是"改多了"。负样本训练 (what NOT to change) 是低成本高收益的方向。

3. **Mid-train for knowledge, Post-train for behavior**: 将 repo 理解 (architecture, conventions, patterns) 放在 mid-training，将 agent 行为 (tool use, planning, recovery) 放在 post-training，可以最大化利用两个阶段的特性。

---

## Appendix: Comparison with Existing Work

| Dimension | R2E-Gym | SWE-Gym | Agentless | Our Proposal |
|-----------|---------|---------|-----------|--------------|
| Data source | Commits | Task instances | -- | Commits + cross-commit |
| Task types | Fix gen + test gen | Full trajectory | Localize + repair + test | 25+ task types |
| Error recovery | RL (implicit) | Rejection sampling | None | Explicit synthesis (3 levels) |
| Negative examples | No | No | No | Yes (scope control) |
| Cross-repo patterns | No | No | No | Yes (clustering) |
| Repo knowledge | No | No | No | Yes (mid-train format) |
| Tool-use training | SWE-agent format | OpenHands format | N/A | Deterministic grounding |
| Multi-commit patterns | No | No | No | Yes (chains, refactoring) |
| Training stage | Post-train (RL) | Post-train (SFT) | N/A | Mid-train + Post-train curriculum |
