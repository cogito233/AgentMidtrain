# AgentMidtrain 最终报告

**日期**: 2026-05-15
**数据版本**: SQA v5b（含 ipython + jupyter 增量）
**输出路径**: `/data_fast_v3/eremite/cogito_explore/AgentMidtrain/data/training_sqa_v5b/`

---

## 一、总览

| 指标 | 值 |
|------|-----|
| 语言 | **Python only** (其他语言仅做初步调研，未纳入) |
| 数据来源 | 185 个 Python 开源仓库的 Git commit 历史 |
| 原始 commit 解析量 | ~150 万条（经 4 层过滤） |
| 去重前原始样本 | 10,516,392 |
| 去重后最终样本 | **6,835,907** |
| 去重率 | 35.0% (MD5 全行 + commit:output content key 双重去重) |
| 估算 token 总量 | **~13.1B** (tiktoken cl100k_base, avg 1,908 tok/sample) |

---

## 二、推荐使用的数据 (4 类)

| 文件 | 样本数 | 平均 query | 平均 answer | 路径 |
|------|-------:|----------:|----------:|------|
| localization | 2,988,000 | 12,549ch | 84ch | `data/training_sqa_v5b/localization.jsonl` |
| commit_message | 1,666,112 | 270ch | 47ch | `data/training_sqa_v5b/commit_message.jsonl` |
| edit_generation | 480,412 | 2,505ch | 1,026ch | `data/training_sqa_v5b/edit_generation.jsonl` |
| test_writing | 26,942 | 27,833ch | 1,096ch | `data/training_sqa_v5b/test_writing.jsonl` |
| **合计** | **5,161,466** | | | **~9.9B tokens** |

### 各类型说明

- **localization**: 3 步分层文件定位。给定 issue 描述 + 30 个候选文件列表，输出目标文件路径。约 30% 样本 output 为 NOT_FOUND（negative sample）。设计参考 Agentless 方案。
- **commit_message**: 给定 diff，生成简短 commit subject line。已清洗 ticket refs (#1234, Fixed/Refs/Closes 前缀)。
- **edit_generation**: 给定 issue 描述 + 文件内容，输出 SEARCH/REPLACE 格式的编辑操作。
- **test_writing**: 给定 issue + fix patch，输出对应的 test patch（SEARCH/REPLACE 格式）。语法已验证。量少但质量高。

---

## 三、不推荐使用的数据 (2 类)

| 文件 | 样本数 | 问题 |
|------|-------:|------|
| bug_detection | 108,492 | **80.9%** 的 answer 包含机械短语 "does not work correctly"，是 commit message 的拙劣改写 |
| code_review | 1,565,881 | **99.0%** 的 Summary 字段与 commit_message 完全重复，加了结构化壳但无独立审查内容 |

---

## 四、语言分布

| 语言 | 占比 | 说明 |
|------|------|------|
| **Python** | **100%** | 所有数据均来自 Python 仓库，使用 `--python-only` 过滤 |
| 其他语言 | 0% | 初步调研了 Go/Rust/TypeScript 等，但 parse 机制不同，未实际纳入 |

---

## 五、仓库列表 (185 个，按 commit_message 样本量排序)

### Tier 1: 主力仓库 (>20K samples, 共 18 个)

| # | 仓库 | 样本数 | 占比 | 领域 |
|---|------|-------:|-----:|------|
| 1 | odoo/odoo | 159,130 | 9.7% | ERP 系统 |
| 2 | python/cpython | 108,646 | 6.6% | CPython 解释器 |
| 3 | getsentry/sentry | 92,883 | 5.7% | 错误监控平台 |
| 4 | saltstack/salt | 56,165 | 3.4% | 配置管理 |
| 5 | home-assistant/core | 50,442 | 3.1% | 智能家居 |
| 6 | sympy/sympy | 49,968 | 3.1% | 符号计算 |
| 7 | ansible/ansible | 46,136 | 2.8% | 自动化运维 |
| 8 | matplotlib/matplotlib | 35,829 | 2.2% | 数据可视化 |
| 9 | django/django | 35,118 | 2.1% | Web 框架 |
| 10 | pandas-dev/pandas | 33,405 | 2.0% | 数据分析 |
| 11 | openstack/nova | 32,108 | 2.0% | 云计算 |
| 12 | pytorch/pytorch | 31,599 | 1.9% | 深度学习 |
| 13 | astropy/astropy | 31,445 | 1.9% | 天文计算 |
| 14 | numpy/numpy | 29,778 | 1.8% | 数值计算 |
| 15 | scikit-learn/scikit-learn | 28,565 | 1.7% | 机器学习 |
| 16 | ray-project/ray | 25,083 | 1.5% | 分布式计算 |
| 17 | ansible/awx | 23,123 | 1.4% | Ansible Tower |
| 18 | yt-dlp/yt-dlp | 21,810 | 1.3% | 视频下载 |

### Tier 2: 中型仓库 (5K-20K, 共 60 个)

wagtail, readthedocs, bokeh, mne-python, sphinx, openstack/neutron, hypothesis, Pillow, google-cloud-python, Theano, datalad, airflow, sqlalchemy, pytest, cryptography, mypy, statsmodels, aiohttp, twisted, celery, orange3, pip, django-cms, transformers, great-expectations, openstack/cinder, optuna, pylint, pyramid, pyinstaller, scrapy, dask, PaddlePaddle, openstack/horizon, aws-cli, openstack/keystone, dvc, conda, keras, coveragepy, django-oscar, networkx, buildbot, django-rest-framework, botocore, Zope, fastapi, scikit-image, openstack/swift, setuptools, xarray, beets, pygments, gevent, boto, discord.py, pydantic, manim, mitmproxy, requests

### Tier 3: 小型仓库 (1K-5K, 共 65 个)

thinc, boto3, cloud-custodian, mlflow, werkzeug, tornado, certbot, docker-compose, flask, GitPython, superset, pwntools, urllib3, poetry, faker, allauth, wandb, strawberry-graphql, pytorch-lightning, pipenv, marshmallow, paramiko, django-extensions, seaborn, flask-admin, docker-py, black, pyro, jinja, sanic, locust, click, ZODB, scipy, kombu, pgcli, piccolo, attrs, elasticsearch-py, isort, starlette, httpx, typer, fabric, django-oauth-toolkit, uvicorn, channels, rich, auto-sklearn, youtube-dl, trio, django-guardian, horovod, yfinance, quart, arrow, tox, dateutil, django-filter, django-redis, django-storages, pelican, django-taggit, django-model-utils, fairseq

### Tier 4: 尾部仓库 (<1K, 共 40 个)

xgboost, ansible-lint, tqdm, pre-commit, flake8, graphene-django, imbalanced-learn, pip-tools, easy-thumbnails, moviepy, salt-testing, cookiecutter, django-celery-beat, asgiref, Flask-SocketIO, bandit, httpie, w3lib, daphne, robotframework, waitress, python-lsp-server, packaging, wheel, yarl, flask-cors, mako, Flask-Migrate, schedule, spaCy, multidict, itsdangerous, virtualenv, django-debug-toolbar, click-plugins, alembic, ipython, jupyter/notebook

---

## 六、领域覆盖

| 领域 | 代表仓库 | 样本占比 |
|------|---------|---------|
| Web 框架 | Django, Flask, FastAPI, Tornado, Sanic, Starlette | ~8% |
| 数据科学/ML | pandas, numpy, scikit-learn, PyTorch, Keras, xarray | ~15% |
| 云/基础设施 | OpenStack(5), Ansible, Salt, Airflow | ~18% |
| DevOps/工具 | pip, setuptools, poetry, pytest, pylint, black, mypy | ~10% |
| 科学计算 | astropy, sympy, scipy, statsmodels, matplotlib | ~10% |
| 应用/业务 | Sentry, Odoo, Wagtail, django-cms, Home Assistant | ~20% |
| 网络/API | requests, aiohttp, httpx, urllib3, boto3 | ~8% |
| 其他 | 密码学, 多媒体, NLP, 游戏 | ~11% |

---

## 七、数据格式

每行 JSON (SQA format):
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

## 八、管线架构

```
clone (bare) → parse_commits.py → filter_commits.py → generate_tasks.py → merge_all_tasks.py → convert_to_sqa.py
```

### 4 层过滤策略

| 级别 | 条件 | 通过率 | 用途 |
|------|------|--------|------|
| strict | require-test + require-src + fix-keywords + python-only + max 5 files + 200 lines | 3-7% | 高质量核心 |
| relaxed | python-only + max 8 files + 300 lines | 15-30% | 扩量 |
| ultra | max 10 files + 400 lines | ~47% | 进一步扩量 |
| max | max 12 files + 500 lines | ~95% | 最大化 |

### 6 种任务生成器

| 生成器 | 设计 | 输出格式 |
|--------|------|---------|
| localization | 3 步分层定位 (候选采样 + 70% include prob) | 文件路径 / NOT_FOUND |
| commit_message | 清洗 ticket refs | 简短 subject line |
| edit_generation | diff → SEARCH/REPLACE 转换 | `<<<<<<< SEARCH ... >>>>>>> REPLACE` |
| test_writing | 语法验证 + 回退策略 | SEARCH/REPLACE 格式 |
| bug_detection | ~~结构化 Bug/Location/Severity~~ | **质量差，不推荐** |
| code_review | ~~结构化 Summary/Files/Type~~ | **与 commit_message 重复，不推荐** |

---

## 九、注意事项

1. **odoo 占比偏高** (8.7%)：如需更均匀分布，可在训练时 cap 每 repo ≤5%
2. **localization 占总量 58%**：这是设计如此（每个 commit 生成 2-3 个 localization 样本），如需调整比例可按类型 downsample
3. **test_writing 量少** (27K)：因为需要 commit 同时修改 src 和 test 文件，严格过滤后自然较少
4. **去重仅做精确去重**：MinHash 模糊去重未做，可能存在微小变体的近似重复

---

## 十、文件清单

```
推荐使用:
  /data_fast_v3/eremite/cogito_explore/AgentMidtrain/data/training_sqa_v5b/localization.jsonl       (37G, 2,988,000 lines)
  /data_fast_v3/eremite/cogito_explore/AgentMidtrain/data/training_sqa_v5b/commit_message.jsonl     (1.4G, 1,666,112 lines)
  /data_fast_v3/eremite/cogito_explore/AgentMidtrain/data/training_sqa_v5b/edit_generation.jsonl    (2.0G, 480,412 lines)
  /data_fast_v3/eremite/cogito_explore/AgentMidtrain/data/training_sqa_v5b/test_writing.jsonl       (784M, 26,942 lines)

不推荐:
  /data_fast_v3/eremite/cogito_explore/AgentMidtrain/data/training_sqa_v5b/bug_detection.jsonl      (80.9% 机械模板)
  /data_fast_v3/eremite/cogito_explore/AgentMidtrain/data/training_sqa_v5b/code_review.jsonl        (99% 与 commit_message 重复)
  /data_fast_v3/eremite/cogito_explore/AgentMidtrain/data/training_sqa_v5b/all_combined.jsonl       (含所有类型，不建议直接用)

历史版本:
  /data_fast_v3/eremite/cogito_explore/AgentMidtrain/data/training_sqa_v5/                          (v5, 不含 ipython/jupyter)

中间产物:
  /data_fast_v3/eremite/cogito_explore/AgentMidtrain/data/tasks_merged_v5b/                        (去重后的原始格式)
  /data_fast_v3/eremite/cogito_explore/AgentMidtrain/data/tasks/                                   (488 个 task 目录)
  /data_fast_v3/eremite/cogito_explore/AgentMidtrain/data/filtered_commits/                        (过滤后的 commit JSONL)
  /data_fast_v3/eremite/cogito_explore/AgentMidtrain/repos/                                        (186+ 个 bare clone)
```
