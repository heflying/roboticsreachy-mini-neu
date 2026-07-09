# Zipformer Sherpa-ONNX 本地流式 ASR Provider 设计规格

> 日期：2026-05-06
> 状态：待批准
> 范围：新增本地流式中文 ASR provider，基于 sherpa-onnx + Zipformer

---

## 一、背景与目标

### 现状

项目已有 8 个 ASR provider，分为云端（Qwen Realtime、Deepgram、OpenAI Realtime）和本地（Parakeet MLX/NeMo、Voxtral、Nemotron）。本地 provider 依赖 Apple Silicon 或 CUDA，无纯 CPU 的本地中文流式方案。

### 目标

引入基于 sherpa-onnx 的 Zipformer 本地流式 ASR provider：
- **纯 CPU 运行**，无特殊硬件要求
- **中文专用**，使用 INT8 量化模型（~160MB，RTF 0.15）
- **流式识别**，与现有 VAD + TurnController 打断机制无缝配合
- **即开即用**，首次运行自动下载模型

### 排除

- "边听边想"（基于 partial transcript 提前启动 LLM）不在本次 scope，作为后续独立优化
- GPU/CUDA 支持不在本次 scope，后续可扩展

---

## 二、技术选型

### 为什么选 sherpa-onnx

| 维度 | sherpa-onnx | NeMo 流式 | icefall |
|---|---|---|---|
| 部署 | `pip install sherpa-onnx`，捆绑 ONNX Runtime | 重量级，依赖 PyTorch | 仅训练框架 |
| CPU 延迟 | RTF 0.15 (INT8 large) | 显著更高 | N/A |
| 内存 | ~160MB (large INT8) | >2GB | N/A |
| 中文模型 | 多个预训练流式模型 (2025) | 有限 | 训练源 |
| 流式 API | 成熟的 Python OnlineRecognizer | 复杂 | 仅导出 |
| 端点检测 | 内置 3 条可配置规则 | 需手动实现 | N/A |
| 跨平台 | Linux/macOS/Windows | Linux/macOS/Windows | Linux |

### 选定模型

`csukuangfj/sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30`
- 编码器 154MB，INT8 量化
- RTF 0.15（约 6.7x 实时），单线程 CPU
- 运行时 RAM 约 300-500MB

---

## 三、架构设计

### 方案选择

**采用方案 A：直接继承 StreamingASRProvider**

sherpa-onnx 的 `OnlineRecognizer` + `OnlineStream` 是有状态的增量解码器，与 `StreamingASRProvider` 的 4 方法生命周期天然映射。ProgressiveASRBase 的滑动窗口层会与 OnlineStream 内部状态管理冲突，是冗余的。

### 类结构

```
StreamingASRProvider (base_streaming.py)
└── ZipformerSherpaASR (zipformer_sherpa.py)
    ├── _recognizer: OnlineRecognizer    # 类级单例，跨流共享
    ├── _stream: OnlineStream | None     # 每流实例，start_stream 创建
    ├── _partial_text: str               # 当前部分结果
    └── _model_loaded: bool              # 模型是否已加载
```

### 文件位置

- 实现：`src/reachy_mini_conversation_app/cascade/asr/zipformer_sherpa.py`
- 下载脚本：`scripts/download_zipformer_zh.py`（可选，离线预下载）

### VAD + ASR 配合

通过 `StreamingASRProvider` 接口与 `ContinuousVADRecorder` 的 `StreamingASRCallbacks` 配合：

```
VAD SPEECH_STARTED → callbacks.on_start()
  → asr.start_stream()
  → 发送 pre-roll 音频 (callbacks.on_chunk)

VAD RECORDING → 每个 audio chunk
  → callbacks.on_chunk() → asr.send_audio_chunk()
  → asr.get_partial_transcript() → UI 实时显示

VAD SPEECH_ENDED → asr.end_stream()
  → final_transcript → 启动 LLM pipeline
```

### 打断 (Barge-in) 支持

打断在 ASR 之上的 TurnController + InterruptCoordinator 层处理：
1. TTS 播放期间 VAD 检测到新语音 → `TurnController.handle_barge_in()`
2. 取消当前 turn 的 LLM/TTS 任务（TurnCancellationToken）
3. 中断音频播放（audio_playback.interrupt()）
4. 创建新 turn → ASR 开始新的 `start_stream()`

ZipformerSherpaASR 实现了 `StreamingASRProvider` 接口，天然适配此流程，无需额外代码。

---

## 四、核心方法设计

### 构造函数 — 即时加载

```python
def __init__(self, model_id, model_dir, num_threads, sample_rate,
             decoding_method, enable_endpoint, rule1_min_trailing_silence,
             rule2_min_trailing_silence):
    self._ensure_model()  # 立即加载，不做懒加载
```

`_ensure_model()` 做两件事：
1. 检查 `model_dir` 下 4 个文件是否完整，缺失则 `snapshot_download()`
2. 创建 `OnlineRecognizer.from_transducer(...)`
3. `INFO` 日志记录加载耗时

**不做懒加载**：确保对话一开始即可获得最佳性能。

### start_stream()

```python
async def start_stream(self) -> None:
    tracker.mark("asr_local_ready")           # B1: 本地模型已就绪
    self._stream = self._recognizer.create_stream()
    self._partial_text = ""
    tracker.mark("asr_local_stream_start")    # B2
```

### send_audio_chunk(audio_chunk: bytes)

```python
async def send_audio_chunk(self, audio_chunk: bytes) -> None:
    audio = wav_to_float32(audio_chunk, self.sample_rate)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, self._sync_feed, audio)

def _sync_feed(self, audio: np.ndarray) -> None:
    self._stream.accept_waveform(self.sample_rate, audio.tolist())
    if self._recognizer.is_ready(self._stream):
        self._recognizer.decode_stream(self._stream)
```

### get_partial_transcript()

```python
async def get_partial_transcript(self) -> str | None:
    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(
        None, self._recognizer.get_result, self._stream
    )
    if text.strip():
        self._partial_text = text.strip()
        tracker.mark("asr_local_chunk_decode")  # B3
    return self._partial_text or None
```

### end_stream()

```python
async def end_stream(self) -> str:
    tracker.mark("asr_local_final_decode")     # B4 start
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, self._stream.input_finished)
    await loop.run_in_executor(None, self._recognizer.decode_stream, self._stream)
    text = await loop.run_in_executor(
        None, self._recognizer.get_result, self._stream
    )
    result = text.strip() or self._partial_text.strip()
    tracker.mark("asr_result_delivered", {"transcript_len": len(result)})  # B4 end
    self._stream = None
    return result
```

### 线程模型

sherpa-onnx 的 C++ 内部是同步的。所有调用通过 `run_in_executor` 放入默认线程池。
- `OnlineRecognizer` 是线程安全的（只读，跨流共享）
- `OnlineStream` 是单线程使用的（每次流只在一个 executor 任务中操作）

---

## 五、模型管理

### 模型文件

```
models/zipformer-zh/
├── encoder.int8.onnx      (~154 MB)
├── decoder.onnx           (~4.9 MB)
├── joiner.int8.onnx       (~1.0 MB)
└── tokens.txt             (~20 KB)
```

### 自动下载策略

采用与 `YoloHeadTracker` 一致的模式：`__init__` 中 `_ensure_model()` 检查并下载。

1. 检查 `model_dir` 下 4 个文件是否都存在且 >0 字节
2. 缺失则 `huggingface_hub.snapshot_download(repo_id=model_id, local_dir=model_dir)`
3. 下载后验证文件完整性

### cascade.yaml 配置

```yaml
zipformer_sherpa:
  module: zipformer_sherpa
  class: ZipformerSherpaASR
  streaming: true
  location: local
  requires: []
  hardware: null
  import_check: sherpa_onnx
  install_extra: cascade_zipformer
  description: "Sherpa-ONNX Zipformer - 本地流式中文 ASR (CPU, ~160MB)"
  model_id: csukuangfj/sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30
  model_dir: models/zipformer-zh
  num_threads: 1
  sample_rate: 16000
  decoding_method: greedy_search
  enable_endpoint: true
  rule1_min_trailing_silence: 2.4
  rule2_min_trailing_silence: 1.2
```

### 依赖

`pyproject.toml` 新增 extra：

```toml
cascade_zipformer = ["sherpa-onnx>=1.10.0", "huggingface_hub"]
```

---

## 六、性能指标适配

### 问题

现有 B1-B4 指标体系为云端 WebSocket ASR 设计。本地 ASR 无连接/会话/上传概念。

### 适配方案

参考本地 TTS (Piper) 的自动检测模式，新增本地 ASR 事件集。

#### 新增事件映射（timing.py EVENT_ALIASES）

```python
# 本地 ASR 事件 (Zipformer sherpa-onnx)
"asr_local_ready":        "asr_reuse",         # B1: 本地模型已就绪 (0ms)
"asr_local_stream_start": "asr_b2_end",        # B2: 流创建完成
"asr_local_chunk_decode": "asr_b3_end",        # B3: chunk 解码 (监控)
"asr_local_final_decode": "asr_b4_start",      # B4 开始: end_stream
# asr_result_delivered 已存在，作为 B4 结束
```

#### B4 计算逻辑自动适配

`calculate_l2_asr_metrics()` 中 B4 增加：
- 云端路径：`asr_commit_sent → asr_result_delivered`（不变）
- 本地路径：`asr_local_final_decode → asr_result_delivered`（新增回退分支）

#### B4 本地 ASR 阈值

| 指标 | ✅ Excellent | 👍 Good | ⚠️ Acceptable | 热点判定 |
|---|---|---|---|---|
| B4 本地 ASR | ≤30ms | ≤50ms | ≤100ms | >100ms |

新增 `ASR_LOCAL_THRESHOLDS` 配置，`calculate_l2_asr_metrics()` 根据事件类型自动选择阈值表。

#### G1 / TTFB 公式

不变。本地 ASR 产出 `asr_result_delivered` 事件，G1 = `asr_result_delivered → llm_start` 自动适配。

TTFB = B4 + G1 + C1 + C2 + D2 + D3

本地 ASR 下 B4 预计 10-50ms（远低于云端 200-400ms），是 TTFB 改善的主要来源。

### timing.py 改动清单

1. `EVENT_ALIASES` 新增 4 个本地 ASR 事件映射
2. 新增 `ASR_LOCAL_THRESHOLDS` 阈值配置
3. `calculate_l2_asr_metrics()` 中 B1/B2/B3 检测本地 ASR 事件时标记 reuse/0ms
4. `calculate_l2_asr_metrics()` 中 B4 增加本地 ASR 分支
5. `validate_trace_formula()` 适配本地 ASR 的 B4 计算路径

---

## 七、可观测性

### 日志

| 级别 | 时机 | 内容 |
|---|---|---|
| INFO | 模型加载 | 加载耗时、模型路径 |
| INFO | end_stream | 最终识别文本 |
| DEBUG | send_audio_chunk | chunk 音频时长、部分结果 |
| DEBUG | 端点检测 | is_endpoint() 触发事件 |

### Timing marks

| 事件 | 阶段 | 说明 |
|---|---|---|
| `asr_local_ready` | B1 | 本地模型已就绪 |
| `asr_local_stream_start` | B2 | 流创建完成 |
| `asr_local_chunk_decode` | B3 | chunk 解码完成 |
| `asr_local_final_decode` | B4 start | end_stream 开始 |
| `asr_result_delivered` | B4 end | 最终结果交付 |

---

## 八、错误处理

| 场景 | 处理 |
|---|---|
| `sherpa_onnx` 未安装 | `ImportError`，提示 `pip install '.[cascade_zipformer]'` |
| 模型下载失败 | `RuntimeError`，附带 HuggingFace URL 和手动下载指引 |
| 音频解码失败 | `WARNING` 日志 + 跳过该 chunk，不中断流 |
| `end_stream` 无结果 | 返回最后的 `_partial_text` 或空字符串 |
| 模型文件损坏 | `_ensure_model()` 校验文件大小，损坏则重新下载 |

---

## 九、测试策略

### 单元测试

| 测试 | 方法 | 说明 |
|---|---|---|
| 模型下载逻辑 | mock `snapshot_download` | 验证文件检查、下载触发、完整性校验 |
| 流式生命周期 | mock `OnlineRecognizer` | start → send chunks → get partials → end |
| 音频格式转换 | 真实 `wav_to_float32` | WAV/PCM 各种采样率/位深 |
| 空音频处理 | 传入空 bytes | 不崩溃，返回空 |
| 时序事件 | 验证 tracker.mark 调用顺序 | B1-B4 事件链完整 |

### 集成测试

| 测试 | 说明 |
|---|---|
| 真实模型推理 | 使用小段中文音频，验证识别结果合理 |
| VAD + ASR 配合 | 模拟 VAD 生命周期，验证 start/send/end 调用正确 |
| 打断流程 | 模拟 barge-in，验证流可被正确重置 |
| 性能指标输出 | 验证 timing report 中 B4 本地指标正确显示 |

---

## 十、变更范围

### 新增文件

- `src/reachy_mini_conversation_app/cascade/asr/zipformer_sherpa.py` — ASR provider 实现
- `scripts/download_zipformer_zh.py` — 可选下载脚本
- `tests/cascade/test_zipformer_sherpa.py` — 测试

### 修改文件

- `cascade.yaml` — 新增 `zipformer_sherpa` provider 配置
- `pyproject.toml` — 新增 `cascade_zipformer` extra
- `src/reachy_mini_conversation_app/cascade/timing.py` — 新增本地 ASR 事件和阈值
- `docs/级联架构性能指标设计.md` — 新增本地 ASR B 指标章节

### 不修改

- `handler.py` — 无需修改，通过 StreamingASRProvider 接口自动适配
- `pipeline.py` — 无需修改
- `interrupt_coordinator.py` / `turn_controller.py` — 无需修改，打断逻辑不变
