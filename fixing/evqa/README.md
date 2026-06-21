# EVQA-fix 使用说明（代码功能梳理 + 推荐修复流程）

本目录包含一套用于修复 E-VQA（EVQA）标注质量的问题审计与修复脚本，主要覆盖：

- `question` 清晰度与改写
- `evidence` 是否支持答案
- QA 对齐（question-answer alignment）
- answer leakage（问题中泄露答案）
- 最终问题/答案/证据的整合检查
- 修复后挑战查询（challenging queries）与拼图图像构建（可选）

本文档基于 `EVQA-fix` 目录脚本的实际实现整理，重点说明每个脚本的核心功能、输入输出列依赖，以及推荐的使用顺序。

## 1. 目录里哪些脚本是“核心”

推荐主修复链（当前更实用）：

- `evqa_add_data_id.py`
- `evqa_question_fix_audit.py`
- `evqa_fix_short_evidence_newlines.py`（可选后处理）
- `evqa_evidence_supporting_audit.py`（证据充分性审计）
- `evqa_qa_alignment_audit.py`（可选诊断）
- `evqa_answer_leak_audit.py`（可选诊断）
- `evqa_final_check.py`（最终整合）
- `evqa_generate_challenging_queries.py`（可选下游）
- `evqa_build_composite_images.py`（可选下游）

早期/实验性审计链（仍可用，但更像前期版本）：

- `evqa_judge.py`：先判断 evidence 是否蕴含 QA
- `evqa_annotation_audit.py`：审计答案/证据标注质量并可覆盖写回
- `evqa_reaudit_improvable.py`：对 `improvable` 的样本复审
- `EVQA_question_clarity.py`：仅做 question clarity 分类（JSONL 输出）

辅助工具：

- `evqa_kb_lookup.py`：按 URL 查 KB 条目/section
- `evqa_context_length_analysis.py`：统计 KB section/context 长度分布
- `evqa_fix_short_evidence_newlines.py`：清洗 `short_evidence` 中换行

不建议作为主线使用的文件：

- `*copy.py`：通常是备份/旧版本副本
- `results_old/`, `results_Jan20/`, `results_Jan21/`：历史结果与分析

## 2. 运行前准备

### 2.1 Python 依赖

大多数脚本只用标准库；以下脚本有额外依赖：

- `evqa_final_check.py` 需要 `openai` SDK
- `evqa_build_composite_images.py` 需要 `Pillow`

建议安装：

```bash
pip install openai pillow
```

### 2.2 API 环境变量（LLM 审计脚本）

大部分审计脚本支持 `openai_compat / qwen / qwen3` 风格接口，常用环境变量：

```bash
export OPENAI_API_BASE="http://your-endpoint"
export OPENAI_API_KEY="your-key"
```

也支持脚本内自动回退读取：

- `QWEN_API_BASE`, `QWEN_API_KEY`
- `DEEPSEEK_API_BASE`, `DEEPSEEK_API_KEY`（`evqa_final_check.py`）

### 2.3 KB 文件（重要）

默认 KB 文件是 `EVQA-fix/encyclopedic_kb_wiki.json`（当前目录下文件非常大）。

注意：

- 多个脚本会直接 `json.load()` 整个 KB 文件到内存
- 这会占用很高内存（明显高于文件大小本身）
- 如果机器内存不够，建议分批跑、关闭 KB (`--no-kb`)，或先做小样本验证

## 3. 输入 CSV 约定（多个脚本共用）

常见必需列（按脚本不同会有差异）：

- `question`
- `answer`
- `evidence`
- `wikipedia_url`
- `evidence_section_id`
- `evidence_section_title`
- `question_type`
- `data_id`（很多脚本强依赖，没有就先用 `evqa_add_data_id.py` 添加）

格式约定（脚本中有显式处理）：

- `answer` 中用 `|` 表示多个候选答案
- 单个候选答案内部用 `&&` 表示多部分答案（multi-part）
- `evidence == "0"` 会被当作缺失证据处理
- `evidence_section_id` 常用 `|` 分隔多个 section id

## 4. 推荐修复流程（主线）

以下流程假设你在仓库根目录运行：

```bash
cd /data2/QianMa/FixKBVQA
```

### Step 1. 给原始 EVQA CSV 补 `data_id`

脚本：`evqa_add_data_id.py`

功能：

- 新增/填充 `data_id`（默认前缀 `E-VQA_`）
- 默认只填空值；加 `--overwrite` 会全部重写

示例：

```bash
python EVQA-fix/evqa_add_data_id.py \
  --input-csv EVQA-fix/test_evqa.csv \
  --output-csv EVQA-fix/test_evqa_with_id.csv
```

### Step 2. 审计 question 清晰度并建议修复（主入口）

脚本：`evqa_question_fix_audit.py`

核心功能：

- 用 LLM 判断问题是否清晰（含具体缺失类型）
- 输出 `suggested_question`
- 若 `Q_clear` 且原始 `evidence` 缺失，可从 KB 抽一段 `short_evidence`
- 自动避免把答案泄露进 `suggested_question`

主要新增列：

- `question_clarity_tag`
- `clarity_reason`
- `suggested_question`
- `short_evidence`
- `short_evidence_source`
- `short_evidence_section_id`
- `short_evidence_section_title`

标签（代码内定义）：

- `Q_clear`
- `Missing_Attribute_Constraint`
- `Missing_Temporal_Scope`
- `Missing_Spatial_Reference`

示例（推荐先小批量试跑）：

```bash
python EVQA-fix/evqa_question_fix_audit.py \
  --input-csv EVQA-fix/test_evqa_with_id.csv \
  --question-column auto \
  --output-jsonl EVQA-fix/results_question_fix/evqa_question_fix.jsonl \
  --output-csv EVQA-fix/results_question_fix/evqa_question_fix.csv \
  --provider openai_compat \
  --model qwen3 \
  --resume \
  --limit 100
```

说明：

- `--question-column auto` 会优先用 `question_original`（若存在），否则退回 `question`
- 支持 `--resume`
- 可用 `--dump-raw` 保存原始 LLM 回复，便于排查 prompt/解析问题

### Step 3. 清理 `short_evidence` 中意外换行（可选但常用）

脚本：`evqa_fix_short_evidence_newlines.py`

功能：

- 清洗目标列中的换行（默认列名 `short_evidence`）
- 只改目标列，其它列原样保留

示例：

```bash
python EVQA-fix/evqa_fix_short_evidence_newlines.py \
  EVQA-fix/results_question_fix/evqa_question_fix.csv \
  -o EVQA-fix/results_question_fix/evqa_question_fix.fixed.csv
```

### Step 4. 审计证据是否支持答案（建议跑）

脚本：`evqa_evidence_supporting_audit.py`

核心功能：

- 检查当前 `evidence` 是否足以支持 annotated answer
- 从 evidence 中提取答案并判断与标注答案是否匹配
- 若 CSV evidence 不支持，会尝试去 KB 对应 section 的段落里找 supporting evidence
- 写回实际使用的证据文本（`evidence_used*` 列）

主要新增列：

- `evidence_sufficiency_tag`
- `matches_annotated_answer`
- `extraction_answer`
- `evidence_supporting_explanation`
- `evidence_used`
- `evidence_used_source`
- `evidence_used_section_id`
- `evidence_used_section_title`
- `kb_missing_url`
- `kb_missing_ids`

标签（代码内定义）：

- `evidence_sufficiency_tag`: `E_supporting` / `E_unsupporting`
- `matches_annotated_answer`: `Match` / `NoMatch` / `NoExtraction`

示例：

```bash
python EVQA-fix/evqa_evidence_supporting_audit.py \
  --input-csv EVQA-fix/results_question_fix/evqa_question_fix.fixed.csv \
  --question-column auto \
  --output-jsonl EVQA-fix/results_evidence_supporting/evqa_evidence_supporting.jsonl \
  --output-csv EVQA-fix/results_evidence_supporting/evqa_evidence_supporting.csv \
  --provider openai_compat \
  --model qwen3 \
  --resume
```

注意：

- 该脚本默认 `--skip-question-types 2_hop`
- 若要全量处理，可显式传空字符串：`--skip-question-types ""`

### Step 5. QA 对齐 / 答案泄露审计（可选诊断）

这两步不会被 `evqa_final_check.py` 自动消费其输出列（仅作诊断或人工复核辅助）。

#### 5.1 QA 对齐审计

脚本：`evqa_qa_alignment_audit.py`

功能：

- 检查 question 与 answer 是否对齐（是否问到了答案所回答的内容）
- 输入时会优先用 `suggested_question`（如果存在）
- 输出 `revised_question` 供人工采用

新增列：

- `qa_alignment_tag`（`Aligned` / `Misaligned`）
- `revised_question`
- `qa_alignment_reason`

示例：

```bash
python EVQA-fix/evqa_qa_alignment_audit.py \
  --input-csv EVQA-fix/results_question_fix/evqa_question_fix.fixed.csv \
  --question-column auto \
  --resume
```

#### 5.2 Answer leakage 审计

脚本：`evqa_answer_leak_audit.py`

功能：

- 检查问题是否直接泄露答案或高度可推断
- 输入时会优先用 `suggested_question`（如果存在）

新增列：

- `answer_leak_tag`（`A_leaks` / `A_inferrable` / `A_ok`）
- `answer_leak_reason`

示例：

```bash
python EVQA-fix/evqa_answer_leak_audit.py \
  --input-csv EVQA-fix/results_question_fix/evqa_question_fix.fixed.csv \
  --question-column auto \
  --resume
```

说明：

- 如果你决定采用 `qa_alignment` 生成的 `revised_question`，需要先手动合并/覆盖到 `suggested_question` 或 `question` 列；`evqa_final_check.py` 不会自动读取 `revised_question`。

### Step 6. 最终整合检查并生成最终修复版 CSV（主出口）

脚本：`evqa_final_check.py`

核心功能（代码中实际做了这些）：

- 选择最终 question（优先 `suggested_question`，否则原 question）
- 从 `answer` 的多候选（`|`）里挑选更匹配证据的候选答案（multi-part `&&` 保留）
- 选择用于 final check 的 evidence（优先 `evidence`，其次 `short_evidence`，再尝试 KB）
- 用 LLM 做最终检查并给出 `final_check_tag`
- 可将 `answer` 列覆盖为最终选择的答案（默认会覆盖）

实践建议：

- `final_check` 本身不依赖 `evidence_sufficiency_tag` 等 Step 4 产物，但如果以 Step 4 的 CSV 作为输入，最终导出会保留这些审计列，便于后续筛查。

主要新增列：

- `final_check_tag`
- `final_check_reason`
- `final_revised_question`
- `final_question_used`
- `final_question_source`
- `final_answer`
- `final_answer_source`
- `final_evidence_used`
- `final_evidence_source`
- `final_answer_leak`
- `final_model`
- `kb_missing_url`
- `kb_missing_ids`
- `answer_original`（若原表没有该列会补上）

标签（代码内定义）：

- `OK`
- `Needs_revision`
- `Answer_leak`
- `Not_answerable`

额外说明（实现细节）：

- 默认 `--skip-q-clear` 为开启状态：`question_clarity_tag == Q_clear` 的样本会跳过 LLM final check，并在输出中用 `final_check_tag = Q_clear` 标记（这是脚本里的“跳过标记”，不是 `FINAL_TAGS` 之一）
- 默认会覆盖 `answer` 列（`--overwrite-answer` 为开启状态）；若不想覆盖，使用 `--no-overwrite-answer`
- 该脚本使用 `openai` SDK（即使 provider 是 `openai_compat`）

示例（推荐）：

```bash
python EVQA-fix/evqa_final_check.py \
  --input-csv EVQA-fix/results_evidence_supporting/evqa_evidence_supporting.csv \
  --question-column auto \
  --output-jsonl EVQA_results_final_check/evqa_final_check.jsonl \
  --output-csv EVQA_results_final_check/evqa_final_check.csv \
  --provider openai_compat \
  --model deepseek-chat \
  --resume
```

如果你没有跑 Step 4，也可以直接输入 `EVQA-fix/results_question_fix/evqa_question_fix.fixed.csv`。

如果你希望连 `Q_clear` 也全部重新检查：

```bash
python EVQA-fix/evqa_final_check.py \
  --input-csv EVQA-fix/results_evidence_supporting/evqa_evidence_supporting.csv \
  --question-column auto \
  --no-skip-q-clear
```

## 5. 修复后生成 challenging queries（可选下游）

### 5.1 生成挑战查询

脚本：`evqa_generate_challenging_queries.py`

功能：

- 从 EVQA CSV 中抽取 anchor 样本
- 按实体类型做平衡采样
- 生成两种配对（Method1 / Method2）
- 为 Method2 生成显式代词解析版本与左右位置版本 query

常见输入来源：

- `EVQA_results_final_check/evqa_final_check.csv`

输出会包含（示例列）：

- `anchor_data_id`, `anchor_entity`, `anchor_question`, `anchor_answer`
- `target_side`
- `method1_pair_data_id`, `method1_query`, `method1_expected_answer`
- `method2_pair_data_id`, `method2_query_with_position`, `method2_query_without_position`
- `method2_pronoun_clarity`, `distractor_entity`

示例：

```bash
python EVQA-fix/evqa_generate_challenging_queries.py \
  --input-csv EVQA_results_final_check/evqa_final_check.csv \
  --output-csv EVQA_results_final_check/evqa_challenging_queries_seed3185.csv \
  --num-samples 100 \
  --seed 3185
```

### 5.2 生成拼接对比图（composite images）

脚本：`evqa_build_composite_images.py`

功能：

- 读取 challenging query CSV（含 anchor / pair 的 `data_id`）
- 从 source CSV（需含 `data_id`, `dataset_image_ids`, `dataset_name`）解析图像路径
- 生成 Method1/Method2 的左右拼图 JPEG
- 把生成路径和状态写回 CSV

新增列（核心）：

- `anchor_image_path`
- `method1_pair_image_path`, `method2_pair_image_path`
- `method1_composite_image_path`, `method2_composite_image_path`
- `method1_image_status`, `method2_image_status`
- `image_error`

示例：

```bash
python EVQA-fix/evqa_build_composite_images.py \
  --challenging-csv EVQA_results_final_check/evqa_challenging_queries_seed3185.csv \
  --source-csv EVQA_results_final_check/evqa_final_check.csv \
  --output-csv EVQA_results_final_check/evqa_challenging_queries_seed3185_with_images.csv \
  --output-image-dir EVQA_results_final_check/composite_images_seed3185
```

## 6. 旧版/补充脚本功能说明（如需复现实验）

### `evqa_judge.py`

作用：

- 用 LLM 判断 QA 与 evidence 的关系：`entailed / contradicted / not_supported`
- 支持从 KB 读取证据、长证据切窗、多答案/多部分答案处理
- 输出 JSONL（不写回 CSV）

适用场景：

- 前期统计 evidence 支持率、筛样本、做误差分析

### `evqa_annotation_audit.py`

作用：

- 基于问题、答案、证据，审计 annotation 质量
- 标签：`good / improvable / incorrect / missing_evidence`
- 可给出 `suggested_answer`、`suggested_evidence`
- 可用 `--apply-suggestions` 直接覆盖写回 CSV

输出：

- JSONL + CSV
- 新增列：`annotation_quality_label`, `annotation_quality_reason`, `annotation_quality_improve_type`, `suggested_answer`, `suggested_evidence`, `evidence_source`, `csv_evidence_in_kb`

### `evqa_reaudit_improvable.py`

作用：

- 读取 `evqa_annotation_audit.py` 的 JSONL 结果
- 针对特定标签（默认 `improvable`）二次复审
- 输出 `reaudit_*` 列（JSONL + CSV）

关键输入依赖：

- `--audit-jsonl` 必须指向上一轮审计 JSONL

### `EVQA_question_clarity.py`

作用：

- 独立版问题清晰度判断（仅 JSONL）
- 标签：`Q_clear / Q_redundant / Q_under-specified`

说明：

- 如果你已经使用 `evqa_question_fix_audit.py`，通常不需要再单独跑这个脚本

## 7. 脚本之间的列依赖（快速查）

- `evqa_add_data_id.py`
: 生成 `data_id`（很多脚本必需）

- `evqa_question_fix_audit.py`
: 生成 `question_clarity_tag`, `suggested_question`, `short_evidence*`

- `evqa_qa_alignment_audit.py`
: 读取 `suggested_question`（若存在），输出 `revised_question`（不会自动被 final_check 使用）

- `evqa_answer_leak_audit.py`
: 读取 `suggested_question`（若存在），输出 `answer_leak_*`（诊断为主）

- `evqa_final_check.py`
: 读取 `question_clarity_tag`, `suggested_question`, `short_evidence`，并生成最终 `final_*`

## 8. 常见问题（按代码行为整理）

### 8.1 报错提示缺少 `data_id`

很多脚本会直接退出并提示先运行 `evqa_add_data_id.py`。先补 ID 再继续。

### 8.2 `Missing --api-base` / `Missing --api-key`

说明对应脚本需要 LLM API。设置环境变量或显式传参：

- `--api-base`
- `--api-key`

### 8.3 `OpenAI SDK not installed`（多见于 `evqa_final_check.py`）

安装：

```bash
pip install openai
```

### 8.4 `Pillow is required`（`evqa_build_composite_images.py`）

安装：

```bash
pip install pillow
```

### 8.5 KB 太大导致内存不足

解决思路：

- 先加 `--limit` 做小样本
- 临时关闭 KB（`--no-kb`），先验证流程
- 分批处理（`--start/--end`）

## 9. 推荐最小可跑命令序列（从原始 EVQA 到 final check）

```bash
cd /data2/QianMa/FixKBVQA

# 1) 加 data_id
python EVQA-fix/evqa_add_data_id.py \
  --input-csv EVQA-fix/test_evqa.csv \
  --output-csv EVQA-fix/test_evqa_with_id.csv

# 2) 问题修复审计（生成 suggested_question / short_evidence）
python EVQA-fix/evqa_question_fix_audit.py \
  --input-csv EVQA-fix/test_evqa_with_id.csv \
  --question-column auto \
  --resume

# 3) 清理 short_evidence 换行（可选）
python EVQA-fix/evqa_fix_short_evidence_newlines.py \
  EVQA-fix/results_question_fix/evqa_question_fix.csv \
  -o EVQA-fix/results_question_fix/evqa_question_fix.fixed.csv

# 4) 证据支持性审计（建议）
python EVQA-fix/evqa_evidence_supporting_audit.py \
  --input-csv EVQA-fix/results_question_fix/evqa_question_fix.fixed.csv \
  --question-column auto \
  --resume

# 5) 最终检查（生成最终修复版）
python EVQA-fix/evqa_final_check.py \
  --input-csv EVQA-fix/results_evidence_supporting/evqa_evidence_supporting.csv \
  --question-column auto \
  --resume
```

如果你需要把 `qa_alignment` 的 `revised_question` 纳入主线，建议在 Step 5 前先人工审核并合并到 `suggested_question` 列。
