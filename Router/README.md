# Reachy BaseBert - LoRA 微调工具

基于 PEFT LoRA 的中文文本分类微调工具，使用 `chinese-bert-wwm-ext` 作为基础模型。

## 目录

- [项目结构](#项目结构)
- [核心功能](#核心功能)
  - [1. 模型训练 (train_lora.py)](#1-模型训练)
  - [2. 模型测试 (test_model.py)](#2-模型测试)
  - [3. 数据生成 (generate_train_dataset.py)](#3-数据生成)
  - [4. 数据增强 (augment_data.py)](#4-数据增强)
  - [5. 数据标注 (llm_label.py)](#5-数据标注)
  - [6. 数据审核 (review_labels.py)](#6-数据审核)
  - [7. 数据转换](#7-数据转换)
    - [7.1 20 Newsgroups 转换 (convert_20ng.py)](#71-20-newsgroups-转换)
    - [7.2 对话数据转换 (convert_interaction_dialogue.py)](#72-对话数据转换)
    - [7.3 Privacy-Care 转换 (convert_privacy_care.py)](#73-privacy-care-转换)
  - [8. LLM 客户端 (llm_client.py)](#8-llm-客户端)
  - [9. 训练辅助脚本 (run_train.ps1)](#9-训练辅助脚本)
  - [10. TensorBoard 可视化](#10-tensorboard-可视化)
- [使用方法](#使用方法)
  - [完整训练流程](#完整训练流程)
  - [单独使用各脚本](#单独使用各脚本)
- [命令行参数](#命令行参数)
- [输出目录结构](#输出目录结构)
- [依赖安装](#依赖安装)

---

## 项目结构

```
Reachy/BaseBert/
├── scripts/
│   ├── train_lora.py              # 主训练脚本（LoRA 微调）
│   ├── test_model.py               # 模型测试脚本
│   ├── generate_train_dataset.py   # 训练数据生成脚本
│   ├── augment_data.py             # 数据增强脚本
│   ├── llm_label.py               # LLM 数据标注脚本
│   ├── review_labels.py            # 交互式标签审核脚本
│   ├── convert_20ng.py            # 20NG 数据集转换
│   ├── convert_interaction_dialogue.py  # 对话数据集转换
│   ├── convert_privacy_care.py    # Privacy-Care 数据集转换
│   ├── llm_client.py              # LLM 客户端（公共库）
│   └── run_train.ps1              # PowerShell 训练辅助脚本
├── data/                           # 训练数据目录
├── origin_model/                   # 原始预训练模型
├── output_lora/                    # LoRA 输出目录
├── output_model/                   # 合并模型输出目录
├── .env                            # 环境变量配置（API Key 等）
└── README.md
```

---

## 核心功能

### 1. 模型训练

**脚本**: `scripts/train_lora.py`

使用 PEFT (Parameter-Efficient Fine-Tuning) 库实现 LoRA (Low-Rank Adaptation) 微调，仅训练少量新增参数，大幅降低显存需求。

#### 功能特性

| 功能 | 说明 |
|------|------|
| LoRA 微调 | 支持 `SEQ_CLS` 任务，默认 `r=8`, `lora_alpha=32` |
| 分类头训练 | 默认解冻 `classifier` 和 `pooler` 层 |
| 类别不平衡处理 | 自动计算逆频率权重，应用于 `CrossEntropyLoss` |
| 定期保存检查点 | 每 N 个 epoch 保存合并模型到 `epoch_{N}` |
| 定期评估 | 每 N 个 epoch 在验证集评估，追踪最佳模型 |
| 最佳模型追踪 | F1 创新高时保存最佳模型信息到 `best/` |
| TensorBoard 可视化 | 自动记录 loss、学习率、评估指标，支持交互式查看训练曲线 |

#### 使用示例

```bash
python scripts/train_lora.py --do_train \
  --train_csv data/train.csv \
  --val_csv data/val.csv \
  --label_list "no_privacy,privacy"
```

---

### 2. 模型测试

**脚本**: `scripts/test_model.py`

支持两种模式：**评估模式**（CSV 文件 + 指标计算 + HTML 报告）和**推理模式**（单句/文本文件，向后兼容）。

#### 模式一：评估模式（推荐）

对多个 CSV 测试文件进行评估，计算准确率、错误率、F1 分数，并生成 HTML 报告。

**CSV 格式**：需包含文本列和标签列（列名可通过参数配置）。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--test_csv` | 无 | 测试 CSV 文件路径（可指定多个） |
| `--text_column` | `text` | 文本列名 |
| `--label_column` | `label` | 标签列名 |
| `--recall_label` | 无 | 指定召回目标标签（如 `privacy`），在报告中高亮显示 |
| `--output_html` | `test_report.html` | HTML 报告输出路径 |

**报告内容**：
- 每个测试文件单独一个 section
- 指标表格：准确率、错误率、F1（macro）、各类精确率/召回率/F1
- 错误样本表格：文本、真实标签、预测标签、各标签概率
- 所有文件汇总表格

```bash
# 评估单个测试文件
python scripts/test_model.py \
  --eval_model_dir output_lora \
  --test_csv data/test.csv \
  --recall_label privacy \
  --output_html test_report.html

# 评估多个测试文件
python scripts/test_model.py \
  --eval_model_dir output_lora \
  --test_csv data/test1.csv data/test2.csv data/test3.csv \
  --recall_label privacy

# 自定义列名
python scripts/test_model.py \
  --eval_model_dir output_lora \
  --test_csv data/test.csv \
  --text_column sentence \
  --label_column category \
  --recall_label privacy
```

#### 模式二：推理模式（向后兼容）

加载模型对输入句子进行推理，输出每个类别的概率。

| 参数 | 说明 |
|------|------|
| `--sentence` | 单条文本（可重复使用） |
| `--input_file` | 文本文件（每行一句） |
| `--input_json` | JSON 文件（字符串数组） |

```bash
# 测试单句
python scripts/test_model.py \
  --eval_model_dir output_lora \
  --sentence "这是隐私信息" \
  --sentence "今天天气很好"

# 批量推理
python scripts/test_model.py \
  --eval_model_dir output_lora \
  --input_file sentences.txt
```

---

### 3. 数据生成

**脚本**: `scripts/generate_train_dataset.py`

根据 GB/T 45574-2025《数据安全技术 敏感个人信息处理安全要求》生成训练数据。

#### 功能特性

| 功能 | 说明 |
|------|------|
| 隐私样本生成 | 生成包含敏感个人信息的句子（10 种类别） |
| 非隐私样本生成 | 生成不包含隐私的句子（5 种类别） |
| 数据增强 | 使用 LLM 对生成样本进行释义增强 |
| 类别平衡 | 可配置每个类别生成的样本数 |

#### 敏感个人信息类别（GB/T 45574-2025）

| 类别 | 说明 |
|------|------|
| `biometric` | 生物识别信息（指纹、人脸、虹膜、声纹等） |
| `religious` | 宗教信仰信息 |
| `specific_identity` | 特定身份信息（身份证、护照、驾照、社保卡等） |
| `medical` | 医疗健康信息（病历、诊断、用药记录等） |
| `financial` | 金融账户信息（银行卡号、支付密码、交易记录等） |
| `location` | 精确位置信息（家庭地址、实时位置、行踪轨迹等） |
| `communication` | 通信内容（邮件、短信、聊天记录等） |
| `personal_identifiable` | 个人身份信息（姓名、电话、邮箱等） |
| `minor` | 未成年人个人信息 |
| `health_tracking` | 健康监测信息（运动数据、睡眠数据、心率等） |

#### 使用示例

```bash
python scripts/generate_train_dataset.py \
  --llm_backend ollama \
  --num_per_category 50 \
  --output_path data/generated_train.csv \
  --augment \
  --augment_times 2
```

---

### 4. 数据增强

**脚本**: `scripts/augment_data.py`

对现有训练数据进行增强，生成更多训练样本。

#### 功能特性

| 增强方法 | 说明 | 是否需要 LLM |
|----------|------|---------------|
| EDA (Easy Data Augmentation) | 同义词替换、随机插入、随机交换、随机删除 | 否 |
| 回译 (Back-Translation) | 中文→英文→中文，生成不同表达 | 是 |
| 释义 (Paraphrase) | 使用 LLM 生成句子的多种表述 | 是 |

#### 使用示例

```bash
# 仅使用 EDA 增强（无需 LLM）
python scripts/augment_data.py \
  --input data/train.csv \
  --output data/train_augmented.csv \
  --augment_times 2 \
  --use_eda

# 使用 LLM 释义增强
python scripts/augment_data.py \
  --input data/train.csv \
  --output data/train_augmented.csv \
  --augment_times 2 \
  --use_paraphrase \
  --llm_backend ollama
```

---

### 5. 数据标注

**脚本**: `scripts/llm_label.py`

使用 LLM 对现有训练数据进行重新标注（或辅助标注）。

#### 功能特性

| 功能 | 说明 |
|------|------|
| LLM 自动标注 | 将每条文本发送给 LLM，根据提示词判断标签 |
| 多后端支持 | 支持 Ollama、Qwen、Spark 三种 LLM 后端 |
| 断点续传 | 支持从已有输出文件恢复，跳过已处理行 |
| 标签解析 | 自动解析 LLM 输出为 `privacy`/`no_privacy`/`unknown` |

#### 标注提示词

脚本内置了详细的隐私判断提示词，定义了以下敏感信息类型：
- 宗教信仰信息
- 特定身份信息
- 医疗健康信息
- 金融账户信息
- 行踪轨迹信息
- 其他敏感个人信息（PII）
- 生物识别信息

#### 使用示例

```bash
python scripts/llm_label.py \
  --input data/train.csv \
  --llm_backend ollama
```

输出文件：`data/train_llm_labeled.csv`

---

### 6. 数据审核

**脚本**: `scripts/review_labels.py`

交互式命令行工具，用于审核和修正 CSV 文件中的标签。

#### 功能特性

| 功能 | 说明 |
|------|------|
| 交互式审核 | 逐条显示文本和标签，单键操作 |
| 标签翻转 | 按 `n` 键翻转标签（privacy ↔ no_privacy） |
| 导航 | `↑`/`p` 上一条，`↓`/`n` 下一条 |
| 自动备份 | 修改前自动创建 `.bak` 备份文件 |
| 两轮审核 | 第一轮审核主要标签，第二轮审核次要标签 |

#### 键盘快捷键

| 按键 | 功能 |
|------|------|
| `↑` / `p` | 上一条 |
| `↓` / `n` | 下一条（或翻转标签） |
| `Space` / `Enter` | 确认当前标签，下一条 |
| `q` | 退出并保存 |

#### 使用示例

```bash
python scripts/review_labels.py --input data/train.csv
```

---

### 7. 数据转换

将第三方数据集转换为统一的训练 CSV 格式（text, label）。

#### 7.1 20 Newsgroups 转换

**脚本**: `scripts/convert_20ng.py`

将 20 Newsgroups PII-Augmented 数据集转换为训练 CSV。

#### 转换流程

1. 读取 `20NG_5topics_PII_anotated.jsonl`
2. 使用 NLTK 将文本分割为句子
3. 滑动窗口采样（可配置窗口大小）
4. 根据实体偏移量判断标签（privacy/no_privacy）
5. 清洗文本（去除引用头、签名、邮件头等）
6. 去重
7. 使用 LLM 将英文翻译为中文

#### 使用示例

```bash
python scripts/convert_20ng.py \
  --window_sizes 1,2,3 \
  --llm_backend ollama
```

#### 7.2 对话数据转换

**脚本**: `scripts/convert_interaction_dialogue.py`

将 Interaction_Dialogue_with_Privacy 数据集转换为训练 CSV。

#### 转换流程

1. 读取 `privacy_annotation_train_zh.json` 和 `privacy_annotation_test_zh.json`
2. 提取每条 user/assistant 话语作为单独样本
3. 根据隐私短语位置判断标签
4. 去重
5. 输出 CSV（中文，无需翻译）

#### 使用示例

```bash
python scripts/convert_interaction_dialogue.py \
  --include_assistant \
  --min_length 5
```

#### 7.3 Privacy-Care 转换

**脚本**: `scripts/convert_privacy_care.py`

将 privacy-care-interactions 数据集转换为训练 CSV。

#### 转换流程

1. 读取 `unsplit-train-en.jsonl`
2. 按 `CW:`/`CR:` 分割说话人回合
3. 去重
4. 标签：`category==2` → privacy，否则 → no_privacy
5. 使用 LLM 将英文翻译为中文

#### 使用示例

```bash
python scripts/convert_privacy_care.py \
  --llm_backend ollama \
  --min_length 5
```

---

### 8. LLM 客户端

**脚本**: `scripts/llm_client.py`

公共 LLM 客户端库，供其他脚本调用。

#### 支持的后端

| 后端 | 说明 | 环境变量 |
|------|------|----------|
| `ollama` | 本地 Ollama 服务 | `OLLAMA_API_KEY`, `OLLAMA_MODEL`, `OLLAMA_BASE_URL` |
| `qwen` | 阿里云百炼（DashScope） | `QWEN_API_KEY`, `QWEN_MODEL`, `QWEN_BASE_URL` |
| `spark` | 讯飞星火 | `SPARK_API_PASSWORD`, `SPARK_MODEL`, `SPARK_BASE_URL` |

#### 环境变量配置（.env 文件）

```env
# Ollama (默认后端)
OLLAMA_API_KEY=ollama
OLLAMA_MODEL=qwen2.5-1.5b-instruct
OLLAMA_BASE_URL=http://127.0.0.1:11434/v1

# Qwen (阿里云百炼)
QWEN_API_KEY=your_api_key_here
QWEN_MODEL=qwen-plus
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# Spark (讯飞星火)
SPARK_API_PASSWORD=your_api_password_here
SPARK_MODEL=lite
SPARK_BASE_URL=https://spark-api-open.xf-yun.com/v1
```

#### 代码示例

```python
from scripts.llm_client import create_client

client = create_client("ollama")
response = client.generate(
    prompt="将以下英文翻译为中文：Hello World",
    system_prompt="你是一个翻译助手。",
    temperature=0.7,
    max_tokens=500
)
print(response)
```

---

### 9. 训练辅助脚本

**脚本**: `scripts/run_train.ps1`

PowerShell 脚本，简化本地训练流程。

#### 功能特性

| 功能 | 说明 |
|------|------|
| 虚拟环境激活 | 自动激活 `.venv` 虚拟环境 |
| 依赖安装 | 通过 `-InstallDependencies` 自动安装依赖 |
| 参数传递 | 支持传递训练参数（CSV 路径、epoch 数等） |
| 日志保存 | 自动保存训练日志到 `scripts/logs/` |

#### 使用示例

```powershell
# 快速运行（使用默认参数）
.\scripts\run_train.ps1

# 指定训练参数
.\scripts\run_train.ps1 `
  -TrainCsv data\train.csv `
  -ValCsv data\val.csv `
  -OutputDir output_lora `
  -Epochs 3 `
  -BatchSize 8

# 安装依赖
.\scripts\run_train.ps1 -InstallDependencies
```

---

### 10. TensorBoard 可视化

训练过程自动记录 TensorBoard 日志，可视化以下指标：

| 指标 | 说明 |
|------|------|
| `train/loss` | 每个 logging step 的训练 loss |
| `eval/loss` | 每个 eval step 的验证 loss |
| `eval/accuracy` | 验证集准确率 |
| `eval/f1_macro` | 验证集宏平均 F1 |
| `train/learning_rate` | 当前学习率 |

#### 启动 TensorBoard

```bash
# 默认日志目录：output_lora/runs/
tensorboard --logdir output_lora/runs/

# 自定义日志目录
tensorboard --logdir your_logging_dir/runs/
```

启动后在浏览器打开 `http://localhost:6006` 查看训练曲线。

#### 自定义日志目录

通过 `--logging_dir` 参数指定 TensorBoard 日志目录（默认：`output_dir/runs/`）：

```bash
python scripts/train_lora.py --do_train \
  --train_csv data/train.csv \
  --logging_dir custom_logs/runs/
```

#### 注意事项

- TensorBoard 功能需要安装 `tensorboard` 包（见[依赖安装](#依赖安装)）
- 日志目录与模型输出目录分离，删除日志不影响模型
- 多次训练使用同一日志目录时，TensorBoard 会自动区分不同 run

---

## 使用方法

### 完整训练流程

#### Step 1: 准备数据

**选项 A：使用现有数据集转换**

```bash
# 转换 20 Newsgroups 数据集
python scripts/convert_20ng.py --llm_backend ollama

# 转换对话数据集
python scripts/convert_interaction_dialogue.py

# 转换 Privacy-Care 数据集
python scripts/convert_privacy_care.py --llm_backend ollama
```

**选项 B：生成合成数据**

```bash
python scripts/generate_train_dataset.py \
  --llm_backend ollama \
  --num_per_category 50 \
  --output_path data/train.csv
```

#### Step 2: 数据增强（可选）

```bash
python scripts/augment_data.py \
  --input data/train.csv \
  --output data/train_augmented.csv \
  --augment_times 2 \
  --use_eda \
  --use_paraphrase \
  --llm_backend ollama
```

#### Step 3: LLM 辅助标注（可选）

```bash
python scripts/llm_label.py \
  --input data/train.csv \
  --llm_backend ollama
```

#### Step 4: 人工审核标签

```bash
python scripts/review_labels.py --input data/train.csv
```

#### Step 5: 训练模型

```bash
python scripts/train_lora.py --do_train \
  --train_csv data/train.csv \
  --val_csv data/val.csv \
  --model_dir origin_model/chinese-bert-wwm-ext \
  --output_dir output_lora \
  --label_list "no_privacy,privacy" \
  --epochs 3 \
  --batch_size 8
```

#### Step 6: 测试模型

```bash
python scripts/test_model.py \
  --model_dir output_lora/latest \
  --input_file data/test.txt
```

---

### 单独使用各脚本

#### 仅训练

```bash
python scripts/train_lora.py --do_train \
  --train_csv data/train.csv \
  --label_list "no_privacy,privacy"
```

#### 仅评估

```bash
python scripts/train_lora.py --do_eval \
  --model_dir output_lora/latest \
  --val_csv data/test.csv \
  --label_list "no_privacy,privacy"
```

#### 数据转换

```bash
# 20 Newsgroups
python scripts/convert_20ng.py --skip_translate  # 跳过翻译（输出英文）

# 对话数据
python scripts/convert_interaction_dialogue.py --include_assistant

# Privacy-Care
python scripts/convert_privacy_care.py --skip_translate
```

---

## 命令行参数

### train_lora.py 参数

#### 数据参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--train_csv` | `str` (多选) | 无 | 训练 CSV 文件路径（支持多个） |
| `--val_csv` | `str` (多选) | `[]` | 验证 CSV 文件路径（支持多个） |
| `--text_column` | `str` | `"text"` | CSV 中文本列的列名 |
| `--label_column` | `str` | `"label"` | CSV 中标签列的列名 |
| `--label_list` | `str` | `""` | 固定标签列表（逗号分隔） |
| `--max_length` | `int` | `256` | 文本最大长度 |

#### 模型参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_dir` | `str` | `"chinese-bert-wwm-ext"` | 预训练模型目录 |
| `--output_dir` | `str` | `"output_lora"` | 输出目录 |
| `--no_train_head` | `flag` | 禁用 | 禁用分类头训练 |
| `--no_class_weight` | `flag` | 禁用 | 禁用类别权重 |

#### 训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--epochs` | `int` | `3` | 训练轮数 |
| `--batch_size` | `int` | `8` | 批次大小 |
| `--learning_rate` | `float` | `5e-5` | 学习率 |
| `--gradient_accumulation_steps` | `int` | `1` | 梯度累积步数 |
| `--seed` | `int` | `42` | 随机种子 |

#### 保存与评估参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--save_every_epochs` | `int` | `10` | 每 N 个 epoch 保存（0=禁用） |
| `--eval_every_epochs` | `int` | `1` | 每 N 个 epoch 评估（0=禁用） |

#### 模式控制参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--do_train` | `flag` | 禁用 | 启用训练模式 |
| `--do_eval` | `flag` | 禁用 | 启用评估模式 |

---

### test_model.py 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_dir` | `str` | `"output_lora"` | 模型目录 |
| `--sentence` | `str` (多选) | 无 | 测试句子（可多次使用） |
| `--input_file` | `str` | 无 | 输入文本文件（每行一句） |
| `--input_json` | `str` | 无 | 输入 JSON 文件（字符串数组） |
| `--max_length` | `int` | `256` | 最大长度 |
| `--batch_size` | `int` | `1` | 批次大小 |

---

### generate_train_dataset.py 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--llm_backend` | `str` | `"ollama"` | LLM 后端 |
| `--num_per_category` | `int` | `50` | 每个类别生成的样本数 |
| `--output_path` | `str` | `"data/privacy/generated_train.csv"` | 输出路径 |
| `--augment` | `flag` | 禁用 | 是否进行数据增强 |
| `--augment_times` | `int` | `2` | 每个样本增强的次数 |

---

### augment_data.py 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | `str` | 必填 | 输入 CSV 文件路径 |
| `--output` | `str` | 必填 | 输出 CSV 文件路径 |
| `--augment_times` | `int` | `2` | 每个样本增强的次数 |
| `--use_eda` | `flag` | 启用 | 是否使用 EDA 增强 |
| `--use_back_translation` | `flag` | 禁用 | 是否使用回译增强 |
| `--use_paraphrase` | `flag` | 禁用 | 是否使用释义增强 |
| `--llm_backend` | `str` | 无 | LLM 后端 |

---

### llm_label.py 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | `str` (多选) | 必填 | 输入 CSV 文件（可多个） |
| `--llm_backend` | `str` | `"ollama"` | LLM 后端 |
| `--label_prompt` | `str` | 无 | 自定义标注提示词 |
| `-v` / `--verbose` | `flag` | 禁用 | 详细日志 |

---

### review_labels.py 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | `str` | 必填 | 输入 CSV 文件（原地修改） |
| `-v` / `--verbose` | `flag` | 禁用 | 详细日志 |

---

### convert_20ng.py 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | `str` | 自动检测 | 输入 JSONL 文件路径 |
| `--output` | `str` | 自动检测 | 输出 CSV 文件路径 |
| `--window_sizes` | `str` | `"1,2,3"` | 滑动窗口大小（逗号分隔） |
| `--llm_backend` | `str` | `"ollama"` | LLM 后端 |
| `--skip_translate` | `flag` | 禁用 | 跳过翻译（输出英文） |
| `-v` / `--verbose` | `flag` | 禁用 | 详细日志 |

---

### convert_interaction_dialogue.py 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | `str` (多选) | 自动检测 | 输入 JSON 文件（可多个） |
| `--output` | `str` | 自动检测 | 输出 CSV 文件路径 |
| `--min_length` | `int` | `5` | 最小字符长度 |
| `--include_assistant` | `flag` | 禁用 | 包含 assistant 话语 |
| `-v` / `--verbose` | `flag` | 禁用 | 详细日志 |

---

### convert_privacy_care.py 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | `str` | 自动检测 | 输入 JSONL 文件路径 |
| `--output` | `str` | 自动检测 | 输出 CSV 文件路径 |
| `--llm_backend` | `str` | `"ollama"` | LLM 后端 |
| `--skip_translate` | `flag` | 禁用 | 跳过翻译（输出英文） |
| `--min_length` | `int` | `5` | 最小字符长度 |
| `-v` / `--verbose` | `flag` | 禁用 | 详细日志 |

---

## 输出目录结构

```
output_lora/
├── origin/                      # 原始模型副本
│   ├── config.json
│   ├── pytorch_model.bin
│   └── ...
├── epoch_10/                    # 第 10 个 epoch 的检查点
│   ├── config.json
│   ├── pytorch_model.bin
│   └── ...
├── epoch_20/                    # 第 20 个 epoch 的检查点
│   └── ...
├── latest/                      # 最新检查点
│   ├── config.json
│   ├── pytorch_model.bin
│   └── ...
├── best/                        # 最佳模型信息
│   ├── score.txt                # 人类可读格式
│   └── score.json               # JSON 格式
└── trainer_state.json           # Trainer 状态（可选）
```

---

## 依赖安装

### 使用 pip

```bash
pip install torch transformers datasets peft accelerate evaluate
pip install openai requests python-dotenv
pip install nltk pandas
pip install tensorboard
```

### 使用 uv（推荐）

```bash
uv sync
uv add tensorboard
```

### 使用 PowerShell 脚本

```powershell
.\scripts\run_train.ps1 -InstallDependencies
```

> **注意**：TensorBoard 功能需要安装 `tensorboard` 包，用于可视化训练曲线。

---

## 故障排查

### 问题 1：标签映射错误

**错误信息：**
```
ValueError: Label 'xxx' not found in fixed label list: [...]
```

**解决方法：**
- 确保 `--label_list` 包含所有可能出现的标签
- 确保 CSV 文件中的标签与 `--label_list` 完全一致（包括大小写）

### 问题 2：模型加载失败

**错误信息：**
```
OSError: Can't load tokenizer for 'xxx'
```

**解决方法：**
- 确保 `--model_dir` 指向包含 `config.json` 和模型权重的目录
- 如果使用本地模型，确保目录路径正确

### 问题 3：NLTK 数据下载失败

**错误信息：**
```
LookupError: Resource punkt_tab not found
```

**解决方法：**
- 脚本会自动尝试手动下载，如仍失败，请手动下载 NLTK 数据：
```python
import nltk
nltk.download('punkt_tab')
```

### 问题 4：LLM 后端连接失败

**错误信息：**
```
ValueError: QWEN_API_KEY is required for qwen backend
```

**解决方法：**
- 确保 `.env` 文件中配置了正确的 API Key
- 确保后端服务正在运行（如 Ollama）

---

## 许可证

（待补充）

---

## 贡献

（待补充）

---

## 更新日志

- **2026-05-28**: 初始版本，支持 LoRA 微调、分类头训练、类别不平衡处理、定期保存与评估
- **2026-05-28**: 添加数据生成、增强、标注、审核、转换等全套数据处理脚本
