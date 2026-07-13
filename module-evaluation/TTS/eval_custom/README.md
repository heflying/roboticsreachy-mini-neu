# TTS 测试脚本使用说明

## 文件说明

- `streaming_tts.py`: 基于 sherpa-onnx 的流式 TTS 类
- `test_streaming_tts.py`: 测试脚本，输入句子列表，输出 wav 音频文件
- `.env.example`: 环境变量配置示例文件
- `.env`: 环境变量配置文件（需要用户创建）

## 使用步骤

### 1. 安装依赖

```bash
pip install sherpa-onnx python-dotenv numpy
```

### 2. 下载模型文件

从 [sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models) 下载模型文件。

支持的模型类型：
- **VITS** (`vits`): 中文语音数据训练，推荐用于中文场景
- **Piper** (`piper`): 英语等多语言
- **Coqui TTS** (`coqui` 或 `xtts`): 支持多语言克隆
- **MeloTTS** (`melo` 或 `melotts`): 中文、英文等

### 3. 创建 .env 配置文件

复制 `.env.example` 为 `.env`，并填写正确的模型路径：

```bash
# Windows
copy .env.example .env

# Linux/Mac
cp .env.example .env
```

然后编辑 `.env` 文件，填写正确的模型路径。

#### VITS 模型配置示例（推荐用于中文）

```env
# 模型类型
TTS_MODEL_TYPE=vits

# 模型文件路径
TTS_MODEL_PATH=E:/programs/Robot/Reachy/models/vits-chinese/model.onnx

# 词典文件路径 (VITS 中文模型需要)
TTS_LEXICON_PATH=E:/programs/Robot/Reachy/models/vits-chinese/lexicon.txt

# tokens 文件路径 (VITS 需要)
TTS_TOKENS_PATH=E:/programs/Robot/Reachy/models/vits-chinese/tokens.txt

# 规则文件路径 (VITS 中文需要)
TTS_RULE_FSTS_PATH=E:/programs/Robot/Reachy/models/vits-chinese/rule.fst

# 说话人 ID (可选，默认: 0)
TTS_SPEAKER_ID=0

# 语速 (可选，默认: 1.0)
TTS_SPEED=1.0

# 输出目录 (可选，默认: result)
TTS_OUTPUT_DIR=result
```

#### Piper 模型配置示例（英文）

```env
TTS_MODEL_TYPE=piper
TTS_MODEL_PATH=path/to/piper-en/model.onnx
TTS_TOKENS_PATH=path/to/piper-en/tokens.txt
TTS_SPEAKER_ID=0
```

#### MeloTTS 模型配置示例（中英文）

```env
TTS_MODEL_TYPE=melo
TTS_MODEL_PATH=path/to/melo/model.onnx
TTS_SPEAKER_ID=0
```

### 4. 运行测试脚本

```bash
python test_streaming_tts.py
```

可选参数：
- `--output-dir`: 输出目录（默认：`result` 或 `.env` 中的 `TTS_OUTPUT_DIR`）
- `--speed`: 语速（默认：1.0 或 `.env` 中的 `TTS_SPEED`）
- `--sentences-file`: 句子列表文件（每行一个句子）

示例：
```bash
# 指定输出目录
python test_streaming_tts.py --output-dir output

# 调整语速
python test_streaming_tts.py --speed 1.2

# 使用自定义句子列表
python test_streaming_tts.py --sentences-file my_sentences.txt

# 同时使用多个参数
python test_streaming_tts.py --output-dir output --speed 1.2 --sentences-file my_sentences.txt
```

### 5. 查看结果

测试完成后，音频文件将保存到 `result` 文件夹（或指定的输出目录）。

## 自定义测试句子

### 方法 1：修改脚本中的句子列表

编辑 `test_streaming_tts.py` 文件，修改 `get_test_sentences()` 函数中的句子列表。

### 方法 2：使用句子列表文件

创建一个文本文件，每行一个句子，然后使用 `--sentences-file` 参数指定该文件：

```bash
python test_streaming_tts.py --sentences-file my_sentences.txt
```

## 常见问题

### 1. 报错：请先安装 sherpa-onnx

运行以下命令安装：
```bash
pip install sherpa-onnx
```

### 2. 报错：缺少必要的环境变量

请根据 `.env.example` 中的说明，创建并编辑 `.env` 文件。

### 3. 生成的音频文件无法正常播放

请检查模型文件路径是否正确，以及模型文件是否完整。

## 示例句子列表

测试脚本默认使用以下句子列表（基于表3的测试维度）：

1. 速度指标 - 短句（测试首包延迟）
2. 效果指标 - 自然度
3. 效果指标 - 情感表现力（安慰、高兴、道歉、提醒、紧急告知）
4. 效果指标 - 停顿处理（长句）
5. 效果指标 - 中文口语适配性（生硬停顿、语调平板、中英文混读、数字读法、日期读法、称谓读法）
6. 效果指标 - 多音字测试

## 输出文件命名规则

输出文件命名规则：`序号_句子内容前20个字符.wav`

例如：`001_你好。.wav`、`003_今天天气真不错，我们一起去公园散步吧。.wav`

## 配置优先级

配置参数的优先级（从高到低）：
1. 命令行参数（如 `--output-dir`、`--speed`、`--sentences-file`）
2. `.env` 文件中的环境变量
3. 脚本中的默认值

例如：
- 如果同时指定了 `--speed 1.5` 和 `TTS_SPEED=1.2`，则使用 1.5
- 如果未指定 `--output-dir`，但 `.env` 中有 `TTS_OUTPUT_DIR=output`，则使用 `output`
- 如果未指定 `--output-dir` 且 `.env` 中也没有 `TTS_OUTPUT_DIR`，则使用默认值 `result`
