# Pipeline 实操指南

## 前置条件

### Python 环境

```bash
# 必须的包
pip install tiktoken gitpython tqdm

# 可选（质量统计）
pip install pandas matplotlib
```

### 系统依赖

- **Git** >= 2.25（需支持 `--format` 高级选项）
- **磁盘空间**: bare clone 约 500MB-5GB/repo，生成数据约 200MB-2GB/repo
- **内存**: 建议 >= 16GB（大 repo 如 cpython 的 commit 解析较吃内存）

### 目录结构

```
/data_fast_v3/eremite/cogito_explore/AgentMidtrain/
├── scripts/          # 所有脚本
├── repos/            # bare clone 的仓库
├── data/
│   ├── raw_commits/        # parse 输出
│   ├── filtered_commits/   # filter 输出
│   ├── tasks/              # generate 输出（每 repo 一个子目录）
│   ├── tasks_merged_v5b/   # merge 输出（去重后）
│   └── training_sqa_v5b/   # convert 输出（最终格式）
└── docs/
```

---

## 添加新 Python 仓库（完整流程）

### Step 1: Clone

```bash
cd /data_fast_v3/eremite/cogito_explore/AgentMidtrain

# 单个 repo
python scripts/add_repos.py --repos "django/django"

# 多个 repo（逗号分隔）
python scripts/add_repos.py --repos "pallets/flask,psf/requests,encode/httpx"
```

clone 为 bare 模式，存放在 `repos/` 目录下（`django/django` → `repos/django_django`）。

### Step 2: Parse commits

```bash
python scripts/parse_commits.py \
  --repo-path repos/django_django \
  --output data/raw_commits/django_django.jsonl
```

输出：每行一个 JSON，包含 commit hash、message、file list、patch 内容等。

### Step 3: Filter

根据需要选择过滤级别：

```bash
# strict: 高质量（需有 test + src + fix keyword）
python scripts/filter_commits.py data/raw_commits/django_django.jsonl \
  -o data/filtered_commits/django_django_strict.jsonl \
  --require-test --require-src --fix-keywords --python-only \
  --max-src-files 5 --max-edit-lines 200

# relaxed: 中等质量
python scripts/filter_commits.py data/raw_commits/django_django.jsonl \
  -o data/filtered_commits/django_django_relaxed.jsonl \
  --require-src --python-only --exclude-docs \
  --max-src-files 8 --max-edit-lines 300

# max: 最大覆盖
python scripts/filter_commits.py data/raw_commits/django_django.jsonl \
  -o data/filtered_commits/django_django_max.jsonl \
  --max-src-files 12 --max-edit-lines 500 --max-patch-length 20000
```

### Step 4: Generate tasks

```bash
python scripts/generate_tasks.py \
  --input data/filtered_commits/django_django_max.jsonl \
  --output-dir data/tasks/django_django_max \
  --repo-path repos/django_django \
  --task-types localization,edit_generation,commit_message,test_writing \
  --workers 4
```

参数说明：
- `--task-types`: 逗号分隔，推荐只用 4 种（不含 bug_detection 和 code_review）
- `--workers`: 并行 worker 数，建议 4-8（受 IO 和 Git 操作限制）
- `--max-file-lines`: 单文件最大行数限制（默认 500）

### Step 5: Merge + Dedup

```bash
python scripts/merge_all_tasks.py --output-dir data/tasks_merged_v5b
```

扫描 `data/tasks/` 下所有子目录，按 task type 合并，MD5 去重。

### Step 6: Convert to SQA

```bash
python scripts/convert_to_sqa.py \
  --input-dir data/tasks_merged_v5b \
  --output-dir data/training_sqa_v5b
```

输出 `{system, query, answer, task_type, repo, commit}` 格式的 JSONL。

---

## 批量处理多个 repo

```bash
# 使用 batch_process_repos.py 一键完成 parse → filter → generate
python scripts/batch_process_repos.py \
  --repo-list repos_to_process.txt \
  --filter-level max \
  --task-types localization,edit_generation,commit_message,test_writing \
  --workers 4
```

或用 `refilter_relaxed.py` 对已有 raw commits 批量重新过滤：

```bash
python scripts/refilter_relaxed.py \
  --input-dir data/raw_commits/ \
  --output-dir data/filtered_commits/ \
  --level max
```

---

## 添加新语言（开发中）

### 核心需适配的模块

1. **`parse_commits.py`**: 文件语言检测
   - 添加语言对应的文件后缀映射（如 `.go`, `.ts`, `.tsx`）
   - 添加项目配置文件识别（`go.mod`, `package.json`, `pom.xml`）

2. **`filter_commits.py`**: 文件分类规则
   - 定义该语言的 test 文件识别规则（Go: `*_test.go`; TS: `*.test.ts`, `*.spec.ts`）
   - 定义 docs/config 文件排除规则
   - 添加 `--go-only` / `--typescript-only` 等语言过滤标志

3. **`generate_tasks.py`**: localization 适配
   - Go: flat package 结构，目录层级较浅
   - TS: `src/` + `__tests__/` 或 colocated test 模式
   - 候选文件列表生成逻辑需按语言调整

### 操作步骤

```bash
# 1. Clone Go 仓库
python scripts/add_repos.py --repos "kubernetes/kubernetes"

# 2. Parse（去掉 --python-only 相关逻辑，或添加 --lang go）
python scripts/parse_commits.py \
  --repo-path repos/kubernetes_kubernetes \
  --output data/raw_commits/kubernetes_kubernetes.jsonl

# 3. Filter（用语言对应的参数）
python scripts/filter_commits.py data/raw_commits/kubernetes_kubernetes.jsonl \
  -o data/filtered_commits/kubernetes_kubernetes_max.jsonl \
  --go-only \
  --max-src-files 12 --max-edit-lines 500 --max-patch-length 20000

# 4. Generate tasks（同 Python 流程）
python scripts/generate_tasks.py \
  --input data/filtered_commits/kubernetes_kubernetes_max.jsonl \
  --output-dir data/tasks/kubernetes_kubernetes_max \
  --repo-path repos/kubernetes_kubernetes \
  --task-types localization,edit_generation,commit_message \
  --workers 4
```

注意：`--go-only` / `--typescript-only` 等标志尚未实现，当前仅支持 `--python-only`。

---

## 常用参数参考

### filter_commits.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--python-only` | 仅保留涉及 .py 文件的 commit | False |
| `--require-test` | 要求 commit 同时修改 test 文件 | False |
| `--require-src` | 要求 commit 修改 src 文件 | False |
| `--fix-keywords` | 要求 commit message 包含 fix/bug 等关键词 | False |
| `--exclude-docs` | 排除仅修改文档的 commit | False |
| `--max-src-files N` | 最多修改 N 个 src 文件 | 无限 |
| `--max-edit-lines N` | 最多修改 N 行 | 无限 |
| `--max-patch-length N` | patch 文本最大字符数 | 无限 |

### generate_tasks.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--task-types` | 生成的任务类型（逗号分隔） | 全部 6 种 |
| `--workers N` | 并行 worker 数 | 1 |
| `--max-file-lines N` | 单文件最大行数 | 500 |
| `--input` | 输入的 filtered commit JSONL | 必填 |
| `--output-dir` | 输出目录 | 必填 |
| `--repo-path` | 对应的 bare clone 路径 | 必填 |

### merge_all_tasks.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--output-dir` | 合并输出目录 | 必填 |
| `--input-dir` | 扫描的 tasks 目录 | `data/tasks/` |

---

## 常见问题排查

### Q: parse_commits.py 报 git 错误

**原因**: repo 没有正确 clone 或路径错误。
**解决**: 确认 `repos/<name>` 存在且是 bare clone：
```bash
ls repos/django_django/HEAD  # 应该存在
```
如缺失，重新 clone：
```bash
python scripts/add_repos.py --repos "django/django"
```

### Q: generate_tasks.py 卡住或极慢

**原因**: 大 repo（如 cpython）commit 多，单线程处理慢。
**解决**:
- 增加 `--workers`（建议 4-8）
- 先用 strict filter 生成小量验证，确认无误后再跑 max

### Q: merge 后数据量与预期不符

**原因**: 去重率高（commit:output 相同的样本被合并）。
**解决**: 这是正常行为。可用 `scripts/review_quality.py` 查看各 repo 的去重前后统计。

### Q: 磁盘空间不足

**各阶段空间占用估算**:
- `repos/`: 每个 bare clone 0.5-5GB（总计约 300GB）
- `data/raw_commits/`: 每 repo 10-500MB
- `data/tasks/`: 每 repo 100MB-2GB
- `data/training_sqa_v5b/`: 最终约 45GB

**建议**: 如磁盘紧张，可在 generate 完成后删除 `data/raw_commits/` 和 `data/filtered_commits/`（可从 repo 重新生成）。

### Q: test_writing 样本很少

**原因**: test_writing 要求 commit 同时修改 src 和 test 文件，严格过滤后自然少。
**解决**: 使用 relaxed/max filter 级别，或在 filter 时加 `--require-test`（会减少其他类型但增加 test_writing 的基数）。
