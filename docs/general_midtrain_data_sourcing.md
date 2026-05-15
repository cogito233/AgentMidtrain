# General Mid-Training Data Sourcing Report

> **Date**: 2025-05-15
> **Context**: We have ~10B tokens of task-specific SWE training data (localization, edit_generation, commit_message, test_writing) from 185 Python repos. This report surveys general-purpose open-source corpora to mix with our task-specific data for mid-training a coding-focused LLM.

---

## Table of Contents

1. [Code Completion / Code Understanding](#1-code-completion--code-understanding)
2. [Code Instruction / SFT Datasets](#2-code-instruction--sft-datasets)
3. [General Instruction Following](#3-general-instruction-following)
4. [Reasoning / Math](#4-reasoning--math)
5. [Documentation / Technical Writing](#5-documentation--technical-writing)
6. [Data Mixing Strategy Reference](#6-data-mixing-strategy-reference)
7. [Recommendations](#7-recommendations)

---

## 1. Code Completion / Code Understanding

### 1.1 The Stack v2 (BigCode)

| Field | Details |
|-------|---------|
| **HuggingFace Path** | `bigcode/the-stack-v2-dedup` (deduped), `bigcode/the-stack-v2` (full) |
| **Size** | ~67.5 TB raw; ~900B+ tokens (deduped); 619 programming languages |
| **License** | Permissive-only subset available; original code retains source licenses (MIT, Apache-2.0, etc.). Access requires agreeing to ToS. |
| **Format** | Raw source code files, organized by language. Includes metadata (repo, path, license). |
| **Quality** | State-of-the-art. Sourced from Software Heritage + GitHub. Near-dedup applied. Used to train StarCoder2. |
| **Relevance** | **Critical** - Primary source for code understanding. Python subset directly complements our SWE tasks. |
| **Download** | ```python
from datasets import load_dataset
ds = load_dataset("bigcode/the-stack-v2-dedup", data_dir="data/Python", split="train", streaming=True)
``` |

**Recommendation**: Use the Python subset (estimated ~100-150B tokens) as our primary code completion corpus. Filter for permissive licenses only.

---

### 1.2 StarCoder2 Training Data

| Field | Details |
|-------|---------|
| **HuggingFace Path** | `bigcode/starcoderdata` (v1), The Stack v2 variants for v2 |
| **Size** | StarCoder2-15B trained on 4T+ tokens; StarCoder2-3B on 3.3T tokens |
| **License** | Same as The Stack v2 (permissive subset available) |
| **Format** | Raw source code with FIM (Fill-in-Middle) formatting applied during training |
| **Quality** | Highest quality open code dataset. Deduplicated, quality-filtered. |
| **Relevance** | **High** - The processed/filtered version of The Stack v2 used for actual model training. |
| **Download** | ```python
from datasets import load_dataset
ds = load_dataset("bigcode/starcoderdata", split="train", streaming=True)
``` |

---

### 1.3 CodeParrot

| Field | Details |
|-------|---------|
| **HuggingFace Path** | `codeparrot/codeparrot-clean` |
| **Size** | ~50 GB of deduplicated Python code (~22M files) |
| **License** | Apache-2.0 (tooling); underlying code has mixed licenses |
| **Format** | Raw Python source code files |
| **Quality** | Good for Python-specific tasks. Basic deduplication applied. Older dataset (2022). |
| **Relevance** | **Medium** - Python-only, but overlaps significantly with The Stack v2. Use only if The Stack v2 access is problematic. |
| **Download** | ```python
from datasets import load_dataset
ds = load_dataset("codeparrot/codeparrot-clean", split="train")
``` |

---

### 1.4 CodeSearchNet

| Field | Details |
|-------|---------|
| **HuggingFace Path** | `code_search_net` |
| **Size** | ~6 million functions across 6 languages (Go, Java, JS, PHP, Python, Ruby); ~2M with docstrings |
| **License** | MIT License (dataset); source code has original licenses |
| **Format** | Function-level code with associated natural language docstrings/comments |
| **Quality** | Well-curated, academic-grade. Originally from GitHub/Microsoft. Good for code-NL alignment. |
| **Relevance** | **High** - Function-docstring pairs are excellent for code understanding. Directly relevant to localization tasks. |
| **Download** | ```python
from datasets import load_dataset
ds = load_dataset("code_search_net", "python", split="train")
``` |

---

## 2. Code Instruction / SFT Datasets

### 2.1 CommitPack / CommitPackFT

| Field | Details |
|-------|---------|
| **HuggingFace Path** | `bigcode/commitpack` (full), `bigcode/commitpackft` (filtered) |
| **Size** | CommitPack: ~4TB, ~3B commits across 350+ languages. CommitPackFT: ~2GB filtered subset |
| **License** | Permissive research license; inherits source repo licenses |
| **Format** | (commit_message, old_code, new_code) triples - instruction-style for code editing |
| **Quality** | **Excellent** for code editing tasks. Used to train OctoCoder/OctoGeeX. Paper: "OctoPack" (2023). |
| **Relevance** | **Critical** - Directly aligned with our edit_generation and commit_message tasks. CommitPackFT is the high-quality filtered subset. |
| **Download** | ```python
from datasets import load_dataset
ds = load_dataset("bigcode/commitpackft", split="train")
# For full: ds = load_dataset("bigcode/commitpack", "python", split="train", streaming=True)
``` |

**Recommendation**: CommitPackFT is a must-include. It directly complements our task-specific data format (edit generation, commit messages).

---

### 2.2 Magicoder OSS-Instruct

| Field | Details |
|-------|---------|
| **HuggingFace Path** | `ise-uiuc/Magicoder-OSS-Instruct-75K` |
| **Size** | ~75,000 instruction-response pairs |
| **License** | Apache-2.0 |
| **Format** | Instruction-response pairs: coding problems inspired by real OSS code snippets |
| **Quality** | High quality. Novel "OSS-Instruct" method generates instructions from real code. Paper: "Magicoder: Source Code Is All You Need" (2023). |
| **Relevance** | **High** - Teaches code generation grounded in real-world patterns. Good SFT complement. |
| **Download** | ```python
from datasets import load_dataset
ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K")
``` |

---

### 2.3 Magicoder Evol-Instruct

| Field | Details |
|-------|---------|
| **HuggingFace Path** | `ise-uiuc/Magicoder-Evol-Instruct-110K` |
| **Size** | ~110,000 instruction-response pairs |
| **License** | Apache-2.0 |
| **Format** | Evolved code instructions (progressively more complex) |
| **Quality** | High. Combines WizardCoder's Evol-Instruct technique with Magicoder's grounding approach. |
| **Relevance** | **High** - Complements OSS-Instruct with harder, multi-step coding problems. |
| **Download** | ```python
from datasets import load_dataset
ds = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K")
``` |

---

### 2.4 WizardCoder Evol-Instruct-Code-80k

| Field | Details |
|-------|---------|
| **HuggingFace Path** | `nickrosh/Evol-Instruct-Code-80k-v1` |
| **Size** | ~80,000 instruction-response pairs |
| **License** | Apache-2.0 (but uses GPT-generated outputs - check OpenAI ToS) |
| **Format** | Instruction-response pairs for coding tasks |
| **Quality** | Good. Used to train WizardCoder. The Evol-Instruct method increases problem complexity. |
| **Relevance** | **Medium** - Good for general code generation, less specific to SWE tasks. |
| **Download** | ```python
from datasets import load_dataset
ds = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1")
``` |

---

### 2.5 OpenCodeInterpreter

| Field | Details |
|-------|---------|
| **HuggingFace Path** | `m-a-p/OpenCodeInterpreter-DS` |
| **Size** | ~68,000 multi-turn samples with execution feedback |
| **License** | Apache-2.0 |
| **Format** | Multi-turn conversations with code generation, execution, and iterative refinement |
| **Quality** | High. Unique execution-feedback loop format mimics real development workflows. |
| **Relevance** | **High** - Multi-turn code refinement is very aligned with SWE editing workflows. |
| **Download** | ```python
from datasets import load_dataset
ds = load_dataset("m-a-p/OpenCodeInterpreter-DS")
``` |

---

### 2.6 Code Alpaca

| Field | Details |
|-------|---------|
| **HuggingFace Path** | `sahil2801/CodeAlpaca-20k` |
| **Size** | ~20,000 instruction-response pairs |
| **License** | CC BY-NC 4.0 (follows Stanford Alpaca; GPT-generated) |
| **Format** | (instruction, input, output) triples for coding tasks |
| **Quality** | Moderate. Older dataset (2023). Basic Self-Instruct method. |
| **Relevance** | **Low** - Small, basic quality. Superseded by Magicoder and others. |
| **Download** | ```python
from datasets import load_dataset
ds = load_dataset("sahil2801/CodeAlpaca-20k")
``` |

**Note**: CC BY-NC 4.0 license limits commercial use.

---

### 2.7 Glaive Code Assistant

| Field | Details |
|-------|---------|
| **HuggingFace Path** | `glaiveai/glaive-code-assistant` / `glaiveai/glaive-code-assistant-v2` |
| **Size** | v1: ~136K samples; v2: ~950K+ samples |
| **License** | Apache-2.0 |
| **Format** | Multi-turn conversations (user asks coding question, assistant provides solution) |
| **Quality** | Good. Synthetic but covers multiple languages. Large-scale. |
| **Relevance** | **Medium-High** - Good diversity of coding patterns. v2 has enough volume to be useful. |
| **Download** | ```python
from datasets import load_dataset
ds = load_dataset("glaiveai/glaive-code-assistant-v2")
``` |

---

### 2.8 SWE-bench (Princeton NLP)

| Field | Details |
|-------|---------|
| **HuggingFace Path** | `princeton-nlp/SWE-bench` |
| **Size** | Train: ~19,000 instances; Test: 2,294; Dev: 225 |
| **License** | MIT License |
| **Format** | (issue_description, repository, ground_truth_patch, test_cases) |
| **Quality** | Gold standard for SWE evaluation. Real GitHub issues with verified patches. |
| **Relevance** | **Critical** - Directly aligned with our SWE mid-training goal. The train split can supplement task data. |
| **Download** | ```python
from datasets import load_dataset
ds = load_dataset("princeton-nlp/SWE-bench", split="train")
``` |

---

## 3. General Instruction Following

### 3.1 OpenHermes 2.5

| Field | Details |
|-------|---------|
| **HuggingFace Path** | `teknium/OpenHermes-2.5` |
| **Size** | ~1,001,551 samples (~300-500M tokens estimated) |
| **License** | MIT License (but includes GPT-4 generated content) |
| **Format** | System prompt + user message + assistant response (ShareGPT format) |
| **Quality** | Excellent. One of the most popular SFT datasets. Diverse tasks including coding, reasoning, creative writing. |
| **Relevance** | **Medium** - Good for maintaining general capabilities during mid-training. Contains some coding data. |
| **Download** | ```python
from datasets import load_dataset
ds = load_dataset("teknium/OpenHermes-2.5")
``` |

---

### 3.2 WildChat-1M

| Field | Details |
|-------|---------|
| **HuggingFace Path** | `allenai/WildChat-1M` |
| **Size** | ~1 million real-world conversations |
| **License** | AI2 ImpACT License (Low-risk) - allows research and commercial use |
| **Format** | Multi-turn ShareGPT-style conversations with metadata |
| **Quality** | High. Real user interactions (not synthetic). Diverse topics. Multilingual. |
| **Relevance** | **Medium** - Good for instruction-following diversity. Includes some coding conversations. |
| **Download** | ```python
from datasets import load_dataset
ds = load_dataset("allenai/WildChat-1M")
``` |

---

### 3.3 OpenOrca

| Field | Details |
|-------|---------|
| **HuggingFace Path** | `Open-Orca/OpenOrca` |
| **Size** | ~4.2 million entries |
| **License** | MIT License (but includes GPT-generated outputs) |
| **Format** | System prompt + question + response (augmented FLAN data) |
| **Quality** | Good. Large-scale augmented FLAN collection. Strong for reasoning and following complex instructions. |
| **Relevance** | **Medium** - Useful for maintaining general reasoning during mid-training. |
| **Download** | ```python
from datasets import load_dataset
ds = load_dataset("Open-Orca/OpenOrca")
``` |

---

### 3.4 UltraChat 200k

| Field | Details |
|-------|---------|
| **HuggingFace Path** | `HuggingFaceH4/ultrachat_200k` (filtered); `stingning/ultrachat` (full 1.5M) |
| **Size** | 200K conversations (filtered) / 1.5M conversations (full); ~774M tokens |
| **License** | MIT License |
| **Format** | Multi-turn dialogue |
| **Quality** | Good. Filtered version used to train Zephyr-7B-beta. Covers diverse topics. |
| **Relevance** | **Low-Medium** - General instruction following. Less code-specific. |
| **Download** | ```python
from datasets import load_dataset
ds = load_dataset("HuggingFaceH4/ultrachat_200k")
``` |

---

### 3.5 LMSYS-Chat-1M

| Field | Details |
|-------|---------|
| **HuggingFace Path** | `lmsys/lmsys-chat-1m` |
| **Size** | ~1 million conversations (~3.5GB) |
| **License** | CC-BY-NC-4.0 (Non-Commercial) |
| **Format** | Real conversations with 25+ different LLMs |
| **Quality** | Excellent. Real user interactions from Chatbot Arena. Diverse and challenging. |
| **Relevance** | **Low** - Non-commercial license is restrictive. Real-world distribution is valuable but license limits use. |
| **Download** | Requires ToS agreement on HuggingFace |

**Warning**: CC-BY-NC-4.0 - not suitable for commercial training without explicit permission.

---

## 4. Reasoning / Math

### 4.1 OpenMathInstruct-2 (NVIDIA)

| Field | Details |
|-------|---------|
| **HuggingFace Path** | `nvidia/OpenMathInstruct-2` |
| **Size** | ~14 million problem-solution pairs |
| **License** | Permissive (NVIDIA open license, commercial-friendly) |
| **Format** | Math problem + step-by-step solution |
| **Quality** | Excellent. Generated using strong models with verification. Large-scale. |
| **Relevance** | **Medium** - Mathematical reasoning transfers to code logic. Helps with algorithmic thinking. |
| **Download** | ```python
from datasets import load_dataset
ds = load_dataset("nvidia/OpenMathInstruct-2")
``` |

---

### 4.2 MetaMathQA

| Field | Details |
|-------|---------|
| **HuggingFace Path** | `meta-math/MetaMathQA` |
| **Size** | ~395,000 question-answer pairs |
| **License** | MIT License |
| **Format** | Math question + detailed solution with reasoning steps |
| **Quality** | High. Augmented from GSM8K and MATH via rephrasing, self-verification, backward reasoning. |
| **Relevance** | **Medium** - Good for logical reasoning. Relatively small but high quality. |
| **Download** | ```python
from datasets import load_dataset
ds = load_dataset("meta-math/MetaMathQA")
``` |

---

### 4.3 OpenMathInstruct-1 (NVIDIA)

| Field | Details |
|-------|---------|
| **HuggingFace Path** | `nvidia/OpenMathInstruct-1` |
| **Size** | ~1.8 million problem-solution pairs |
| **License** | Permissive (CC-BY-4.0 or similar NVIDIA license) |
| **Format** | Math problem + solution (generated using Mixtral) |
| **Quality** | Good. Predecessor to v2. Generated with Mixtral models. |
| **Relevance** | **Low-Medium** - Superseded by v2 but can supplement if needed. |
| **Download** | ```python
from datasets import load_dataset
ds = load_dataset("nvidia/OpenMathInstruct-1")
``` |

---

## 5. Documentation / Technical Writing

### 5.1 FineWeb-Edu (HuggingFace)

| Field | Details |
|-------|---------|
| **HuggingFace Path** | `HuggingFaceFW/fineweb-edu` |
| **Size** | ~1.3 trillion tokens (filtered educational content from 15T token FineWeb) |
| **License** | ODC-BY-1.0 (Open Data Commons Attribution) |
| **Format** | Raw text documents scored for educational value (score >= 3 out of 5) |
| **Quality** | Excellent. State-of-the-art web text filtering. Significant benchmark improvements reported. |
| **Relevance** | **High** - Technical/educational content improves reasoning. Can extract CS/programming subsets. |
| **Download** | ```python
from datasets import load_dataset
ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
``` |

**Recommendation**: Use a sampled subset (~5-10B tokens) focusing on CS/programming/math educational content.

---

### 5.2 RedPajama-Data-V2

| Field | Details |
|-------|---------|
| **HuggingFace Path** | `togethercomputer/RedPajama-Data-V2` |
| **Size** | 30+ trillion tokens from 84 CommonCrawl snapshots; 5 languages |
| **License** | Open (Apache-2.0 for tooling; CC-BY for annotations) |
| **Format** | Raw text with 40+ quality signals per document. Includes documentation subsets. |
| **Quality** | Good base. Requires filtering using provided quality signals. |
| **Relevance** | **Medium** - Too large to use raw. Extract docs/tech-writing subsets using quality filters. |
| **Download** | ```python
from datasets import load_dataset
ds = load_dataset("togethercomputer/RedPajama-Data-V2", name="default", split="train", streaming=True)
``` |

---

### 5.3 ArXiv CS Papers (via Dolma / peS2o)

| Field | Details |
|-------|---------|
| **HuggingFace Path** | `allenai/peS2o` (Semantic Scholar filtered); also in `allenai/dolma` |
| **Size** | peS2o: ~39.8B tokens of academic papers; ArXiv subset: ~30B tokens |
| **License** | ODC-BY (Open Data Commons Attribution) for peS2o; Dolma is AI2 ImpACT |
| **Format** | Full paper text (title, abstract, body) |
| **Quality** | High academic quality. Filtered from Semantic Scholar. |
| **Relevance** | **Medium** - CS papers teach reasoning patterns, but verbose. Select CS subfields (PL, SE, AI). |
| **Download** | ```python
from datasets import load_dataset
ds = load_dataset("allenai/peS2o", split="train", streaming=True)
``` |

---

### 5.4 Wikipedia (Technical Articles)

| Field | Details |
|-------|---------|
| **HuggingFace Path** | `wikimedia/wikipedia` (official) |
| **Size** | English Wikipedia: ~4B tokens; Tech/CS articles: ~200-500M tokens |
| **License** | CC-BY-SA-3.0 |
| **Format** | Raw article text with metadata |
| **Quality** | Gold-standard factual content. Well-structured. |
| **Relevance** | **Low-Medium** - Factual knowledge helps, but not directly code-relevant. Small contribution. |
| **Download** | ```python
from datasets import load_dataset
ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
``` |

---

## 6. Data Mixing Strategy Reference

### Industry Precedents

| Model | Code % | NL (code-related) % | General NL % | Total Tokens |
|-------|--------|---------------------|--------------|--------------|
| Code Llama | ~85% | ~8% | ~7% | 500B |
| DeepSeek-Coder v1 | 87% | 10% (EN) | 3% (CN) | 2T |
| DeepSeek-Coder v2 | 60% | 30% (EN) | 10% (CN) | 10.2T |
| StarCoder2-15B | ~85-90% | ~10-15% | <5% | 4T+ |

### Key Principles for Mid-Training Data Mixing

1. **Prevent Catastrophic Forgetting**: Keep 10-30% natural language data to preserve general capabilities.
2. **Quality over Quantity**: High-quality filtered data outperforms raw volume (see FineWeb-Edu results).
3. **Task Proximity**: Data closest to target tasks (code editing, commits) should have highest weight.
4. **Curriculum Learning**: Some evidence suggests starting with broader data and progressively narrowing to task-specific data.
5. **Multi-epoch on Quality Data**: High-quality task-specific data can be seen 2-4x without degradation.

---

## 7. Recommendations

### 7.1 Recommended Data Mix Ratio

For mid-training with ~10B task-specific tokens as the core:

| Category | Ratio | Estimated Tokens | Source |
|----------|-------|------------------|--------|
| **Task-specific SWE data** (our data) | 25-30% | 10B | Internal (localization, edit_gen, commit_msg, test_writing) |
| **Code completion (Python-focused)** | 35-40% | 12-15B | The Stack v2 Python subset |
| **Code instruction/SFT** | 15-20% | 5-7B | CommitPackFT + Magicoder + OpenCodeInterpreter |
| **General instruction + reasoning** | 5-10% | 2-4B | OpenHermes 2.5 + MetaMathQA |
| **Documentation/Educational** | 5% | 1.5-2B | FineWeb-Edu (CS subset) |

**Total estimated: 30-38B tokens**

---

### 7.2 Top 5 Datasets to Prioritize

| Priority | Dataset | Why | Estimated Useful Tokens |
|----------|---------|-----|------------------------|
| 1 | **The Stack v2 (Python dedup)** | Core code understanding; directly complements our repo-level tasks | 10-15B (sampled) |
| 2 | **CommitPackFT** | Identical format to our edit_generation/commit_message tasks | 1-2B |
| 3 | **Magicoder OSS-Instruct + Evol-Instruct** | High-quality code instruction grounded in real code | 0.5-1B |
| 4 | **OpenHermes 2.5** | General instruction following to prevent capability loss | 1-2B |
| 5 | **FineWeb-Edu (CS/programming subset)** | Educational technical content for reasoning | 2-5B |

**Honorable Mentions**:
- OpenCodeInterpreter (multi-turn code refinement)
- SWE-bench train split (directly task-aligned)
- CodeSearchNet (function-docstring alignment)

---

### 7.3 Estimated Total Token Count After Mixing

| Scenario | Total Tokens | Task-specific % | Training Time (est.) |
|----------|-------------|-----------------|---------------------|
| **Conservative** | 25B | 40% | ~2-3 days on 8xA100 |
| **Recommended** | 35B | 28% | ~4-5 days on 8xA100 |
| **Aggressive** | 50B | 20% | ~6-8 days on 8xA100 |

**Recommended**: ~35B total tokens with 28-30% task-specific ratio. This allows 2-3 epochs over our task data while mixing with 1 epoch of general code.

---

### 7.4 Licensing Concerns

| Dataset | License | Commercial OK? | Risk Level |
|---------|---------|---------------|------------|
| The Stack v2 (permissive subset) | Source licenses (MIT/Apache) | Yes | Low |
| CommitPackFT | Research license | Check terms | Medium |
| Magicoder datasets | Apache-2.0 | Yes | Low |
| OpenHermes 2.5 | MIT (but GPT-4 generated) | Unclear | Medium |
| FineWeb-Edu | ODC-BY-1.0 | Yes | Low |
| OpenOrca | MIT (but GPT generated) | Unclear | Medium |
| Code Alpaca | CC BY-NC 4.0 | **No** | High |
| LMSYS-Chat-1M | CC-BY-NC-4.0 | **No** | High |
| MetaMathQA | MIT | Yes | Low |
| CodeSearchNet | MIT | Yes | Low |
| SWE-bench | MIT | Yes | Low |

**Key licensing risks**:
1. **GPT-generated data** (OpenHermes, WizardCoder, Code Alpaca) - OpenAI's ToS may restrict use for training competing models. Use with caution.
2. **NC-licensed datasets** (Code Alpaca, LMSYS-Chat-1M) - Exclude from commercial training.
3. **The Stack v2** - Must use permissively-licensed subset and respect opt-out requests.

**Safe choices**: The Stack v2 (permissive), Magicoder, MetaMathQA, FineWeb-Edu, CodeSearchNet, SWE-bench.

---

### 7.5 Practical Next Steps

1. **Download priority datasets**:
   ```bash
   # Install HF CLI
   pip install huggingface_hub datasets

   # Login
   huggingface-cli login

   # Download CommitPackFT (small, high-value)
   python -c "from datasets import load_dataset; ds = load_dataset('bigcode/commitpackft'); ds.save_to_disk('/data_fast_v3/eremite/cogito_explore/AgentMidtrain/data/commitpackft')"

   # Download Magicoder datasets
   python -c "from datasets import load_dataset; ds = load_dataset('ise-uiuc/Magicoder-OSS-Instruct-75K'); ds.save_to_disk('/data_fast_v3/eremite/cogito_explore/AgentMidtrain/data/magicoder_oss')"

   # Stream The Stack v2 Python (too large for full download)
   python -c "from datasets import load_dataset; ds = load_dataset('bigcode/the-stack-v2-dedup', data_dir='data/Python', split='train', streaming=True)"
   ```

2. **Format standardization**: Convert all datasets to a unified format (e.g., JSONL with `text` field for pretraining, or `instruction/input/output` for SFT).

3. **Quality filtering pipeline**:
   - Deduplication against our existing 185 repos
   - Length filtering (remove very short/very long samples)
   - Language detection (ensure Python focus)
   - Perplexity filtering using a reference model

4. **Implement data mixing**:
   - Use temperature-based sampling across domains
   - Implement curriculum: start broad (epoch 1-2), narrow to task-specific (epoch 3+)
   - Monitor loss per domain during training

---

## Appendix: Quick Reference Table

| Dataset | Path | Tokens (est.) | License | Priority |
|---------|------|---------------|---------|----------|
| The Stack v2 (Python) | `bigcode/the-stack-v2-dedup` | 100-150B | Permissive | P0 |
| CommitPackFT | `bigcode/commitpackft` | 1-2B | Research | P0 |
| Magicoder OSS-Instruct | `ise-uiuc/Magicoder-OSS-Instruct-75K` | ~200M | Apache-2.0 | P1 |
| Magicoder Evol-Instruct | `ise-uiuc/Magicoder-Evol-Instruct-110K` | ~300M | Apache-2.0 | P1 |
| OpenHermes 2.5 | `teknium/OpenHermes-2.5` | ~500M | MIT* | P1 |
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | 1.3T | ODC-BY | P1 |
| OpenCodeInterpreter | `m-a-p/OpenCodeInterpreter-DS` | ~100M | Apache-2.0 | P2 |
| SWE-bench (train) | `princeton-nlp/SWE-bench` | ~50M | MIT | P2 |
| CodeSearchNet | `code_search_net` | ~500M | MIT | P2 |
| MetaMathQA | `meta-math/MetaMathQA` | ~100M | MIT | P2 |
| OpenMathInstruct-2 | `nvidia/OpenMathInstruct-2` | ~2B | NVIDIA Open | P2 |
| Glaive Code Assistant v2 | `glaiveai/glaive-code-assistant-v2` | ~1B | Apache-2.0 | P3 |
| OpenOrca | `Open-Orca/OpenOrca` | ~1.5B | MIT* | P3 |
| WildChat-1M | `allenai/WildChat-1M` | ~1B | AI2 ImpACT | P3 |
| UltraChat 200k | `HuggingFaceH4/ultrachat_200k` | ~774M | MIT | P3 |
| RedPajama-V2 | `togethercomputer/RedPajama-Data-V2` | 30T+ | Apache-2.0 | P3 |

*MIT license on dataset, but content generated by GPT-4/3.5 may have additional restrictions.

---

*Report generated 2025-05-15. Data sizes and availability should be verified before downloading as HuggingFace datasets are frequently updated.*
