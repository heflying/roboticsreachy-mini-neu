# Reachy Mini ChatBox 与 Reachy Mini Conversation App 相同点和差异分析

日期：2026-04-25

对比对象：
- ChatBox 工程：`D:\wsl\peiliao\T1-reference-apps\roboticsreachy-mini-chatbox`
- Conversation App 工程：`D:\wsl\peiliao\T1-reference-apps\reachy_mini_conversation_app`
- Conversation App 参考文档：`D:\wsl\peiliao\T1-reference-apps\reachy_mini_conversation_app\analysis\reachy_mini_conversation_app-ohos-porting-research-20260424.md`

## 1. 总体结论

两个工程同源，包名和入口仍然都是 `reachy_mini_conversation_app`。Conversation App 更像“实时多模态语音大模型机器人应用”，默认围绕 OpenAI Realtime 或 Gemini Live 运行；ChatBox 更像“低成本、可组合、可测试的机器人聊天框架”，默认围绕 ASR -> LLM -> TTS 级联管线运行，并保留 `--realtime` 作为旧路径。

从 OHOS 迁移看：
- 相同部分主要是机器人 SDK、动作系统、profile/tool 机制、摄像头和头部跟踪。
- 差异部分主要是对话后端、音频链路、配置方式、工具语义、测试能力和实时反应系统。
- 若目标是快速产品验证，ChatBox 的 Cascade 结构比 Realtime 结构更容易拆分为端云混合架构。
- 若目标是最自然的低延迟语音对话，Conversation App 的 Realtime/Live 路线仍有优势，但协议和供应商绑定更强。

## 2. 相同点

### 2.1 工程与入口

| 项目 | 相同点 |
| --- | --- |
| 包名 | 均为 `reachy_mini_conversation_app` |
| CLI | 均暴露 `reachy-mini-conversation-app` |
| Reachy app | 均使用 `ReachyMiniConversationApp(ReachyMiniApp)` |
| UI | 均支持 Gradio 与 headless/console 形态 |
| 仿真 | 均通过 `robot.client.get_status()` 判断仿真并自动打开 Gradio |

### 2.2 机器人和动作层

两者都强依赖 Reachy Mini SDK：
- `ReachyMini` 连接机器人或仿真。
- `MovementManager` 做动作队列、idle breathing、语音 wobble、头部跟踪偏移融合。
- `HeadWobbler` 根据输出音频生成头部微动。
- `dance`、`play_emotion`、`move_head` 等工具最终都走 Reachy pose、动作库或录制动作资源。

迁移含义：两个工程都必须先抽象 RobotControl/Media/Camera HAL，否则无法在 OHOS 端闭环。

### 2.3 Profile 与工具系统

共同机制：
- profile 目录包含 `instructions.txt` 和 `tools.txt`。
- `Tool` 抽象提供 function calling schema。
- `ToolDependencies` 注入机器人、动作、摄像头、视觉等运行时对象。
- 支持内置工具、profile-local tool、external tool。
- 系统工具 `task_status`、`task_cancel` 用于后台任务状态和取消。

迁移含义：工具 schema 和 profile 白名单设计可以共用，但 Python 动态加载都不适合直接迁移到 OHOS 原生。

### 2.4 视觉和摄像头能力

共同点：
- 均可选择关闭摄像头。
- 均支持头部跟踪概念。
- 均有本地视觉/云端视觉的分流思路。
- 摄像头帧最终来自 Reachy SDK media/camera 能力。

迁移含义：OHOS 端需要替换摄像头取帧、JPEG 编码和端侧视觉推理栈。

## 3. 核心差异

### 3.1 默认对话架构不同

| 维度 | Conversation App | ChatBox |
| --- | --- | --- |
| 默认模式 | OpenAI Realtime 或 Gemini Live | Cascade ASR -> LLM -> TTS |
| 语音模型 | 单个实时后端同时做语音理解、文本、工具、TTS | ASR、LLM、TTS 三段独立 provider |
| 旧路径 | 不适用 | `--realtime` 回到 OpenAI Realtime |
| 主要编排文件 | `openai_realtime.py`、`gemini_live.py` | `cascade/handler.py`、`cascade/pipeline.py`、`cascade/provider_factory.py` |

影响：
- Conversation App 更依赖实时大模型协议。
- ChatBox 更容易替换单个环节，例如只换 ASR 或只换 TTS。
- ChatBox 的端云拆分粒度更清晰，适合迁移预研。

### 3.2 后端供应商和模型组合不同

Conversation App：
- OpenAI Realtime：音频输入、转写、TTS、工具调用、视觉上下文集中在 Realtime 会话里。
- Gemini Live：同样是实时音视频/工具会话，但需要把工具 schema 转成 Gemini function declarations。

ChatBox：
- ASR provider：`whisper_openai`、`openai_realtime_asr`、`deepgram`、`parakeet_mlx_progressive`、`voxtral_mlx`、`parakeet_nemo_progressive`、`nemotron`。
- LLM provider：OpenAI Chat、Gemini。
- TTS provider：Kokoro、OpenAI TTS、ElevenLabs、Gradium。

影响：
- ChatBox 可以把成本、延迟、隐私和硬件约束拆开调优。
- ChatBox 默认配置里有 Apple Silicon 本地 ASR，对非 macOS arm64 环境不友好，需要改默认 provider。
- Conversation App 的供应商切换颗粒度更大，通常是整个实时后端一起切。

### 3.3 音频链路不同

| 维度 | Conversation App | ChatBox |
| --- | --- | --- |
| 输入分段 | 依赖实时后端或 fastrtc/audio stream | 本地 Silero VAD 状态机 |
| 输出音频 | Realtime/Live 后端直接输出音频 delta | TTS provider 合成 PCM chunk |
| 播放策略 | HeadWobbler 消费后端音频 delta | SpeechOutput + 预热播放线程 + 分句并行 TTS |
| barge-in | 由实时后端和播放队列配合 | 目前主要是 VAD 轮次式处理，barge-in 能力较弱 |

影响：
- ChatBox 的音频链路更可控，但也需要自己处理 VAD、重采样、播放预热和 TTS 排队。
- Conversation App 对后端能力依赖更强，但交互自然度上限更高。

### 3.4 工具语义不同

Conversation App：
- 模型的音频输出天然就是“说话”，工具多用于动作、摄像头、视觉、舞蹈、表情等。
- `camera` 工具可把图片直接送回实时模型或本地 VLM。

ChatBox：
- 新增 `speak` 工具，LLM 必须通过它显式说话。
- 新增 `see_image_through_camera`，返回 JPEG 给 LLM 再进行多模态分析。
- 新增 `describe_camera_image`，仅在本地 VisionManager 可用时开放。
- Pipeline 会对 `speak` 和 `see_image_through_camera` 做特殊处理。

影响：
- ChatBox 的工具协议更接近传统 Chat Completions function calling。
- `speak` 工具把“语言生成”和“语音合成”解耦，是 Cascade 架构的关键适配点。

### 3.5 配置方式不同

Conversation App：
- 核心后端由 `.env` 中 `BACKEND_PROVIDER`、`MODEL_NAME`、API key 等控制。
- `config.py` 负责 OpenAI/Gemini backend 归一化。

ChatBox：
- `cascade.yaml` 管理 ASR/LLM/TTS provider。
- `.env` 主要提供 API key、profile、external path、Hugging Face cache。
- CLI 可覆盖 ASR/LLM/TTS provider。

影响：
- ChatBox 的配置更适合做 provider 矩阵测试。
- OHOS 迁移时，`cascade.yaml` 可以转为设备配置或远端策略。

### 3.6 实时反应系统是 ChatBox 新增重点

ChatBox 新增 `cascade/transcript_analysis`：
- 可在用户说话 partial transcript 阶段触发动作。
- 支持关键词、glob、短语、NER entity、布尔 AND。
- 回调来自 profile Python 文件，例如默认 profile 中的 `react_to_name.py`、`react_to_food_entity.py`、`do_groovy_dance.py`。

Conversation App 没有同等独立的 transcript reaction 子系统，更多依赖模型工具调用。

影响：
- ChatBox 能在 LLM 回复前先做动作反应，产品表现更活跃。
- 迁移时关键词触发很容易端侧化，NER 可后置。

### 3.7 测试能力不同

Conversation App：
- 主要测试集中在工具、视觉、Realtime handler、配置冲突等。
- 端到端语音链路更依赖真实后端和音频环境。

ChatBox：
- 新增 `--autotest` 和 `cascade/autotest_stream.py`。
- 可用文本 utterances 通过输入 TTS 合成用户语音，再跑 ASR -> LLM -> TTS -> 工具链路。
- 更适合做 provider sweep 和端到端回归。

影响：
- ChatBox 对迁移验证更友好，可以把 OHOS 客户端前先用脚本固定测试对话链路。

## 4. 依赖差异

### 4.1 Conversation App 代表性依赖

- `av`、`aiortc`、`fastrtc`、`gradio`
- `openai>=2.30.0`
- `google-genai>=1.0.0`
- `reachy-mini>=1.6.4`
- local vision extra：`accelerate`、`torch`、`transformers==5.3.0`
- yolo/mediapipe vision extras

### 4.2 ChatBox 新增或变化依赖

- 基础 cascade：`sounddevice`、`librosa`、`pyyaml`
- VAD：`torch`、`torchaudio`
- ASR：`mlx-audio`、`parakeet-mlx`、`nemo_toolkit[asr]`、`deepgram-sdk`
- TTS：`kokoro`、`elevenlabs`、`gradium`
- LLM：`google-genai` 被放入 `cascade_gemini` extra，而不是核心依赖
- 平台媒体：增加 `gstreamer-bundle` 和 Linux `PyGObject` 条件依赖

影响：
- ChatBox 依赖面更宽，但大部分变成 optional extras。
- Conversation App 核心依赖更集中，但实时后端 SDK 更关键。
- OHOS 第一阶段建议只接云端 provider，避开本地 ML 依赖。

## 5. 文件结构差异

ChatBox 新增或显著扩展：
- `cascade.yaml`
- `src/reachy_mini_conversation_app/cascade/`
- `src/reachy_mini_conversation_app/tools/speak.py`
- `src/reachy_mini_conversation_app/tools/see_image_through_camera.py`
- `src/reachy_mini_conversation_app/tools/describe_camera_image.py`
- `src/reachy_mini_conversation_app/profiles/*/reactions.yaml`
- `scripts/autotest_providers.py`
- `scripts/sweep_providers.py`
- `tests/cascade/`

Conversation App 独有或更完整：
- `src/reachy_mini_conversation_app/gemini_live.py`
- `src/reachy_mini_conversation_app/camera_frame_encoding.py`
- `src/reachy_mini_conversation_app/vision/head_tracking/`
- 更完整的 `config.py` Realtime/Gemini 后端配置体系。

## 6. OHOS 迁移选择建议

若以“快速落地机器人聊天体验”为目标：
- 优先基于 ChatBox。
- 选云端 ASR、云端 LLM、云端 TTS，OHOS 端先做音频、摄像头、UI 和机器人 HAL。
- 保留 `speak` 工具、TurnResult、TranscriptAnalysisManager 的 keyword 分支。

若以“最低延迟自然对话”为目标：
- 保留 Conversation App 的 Realtime/Live 路线作为并行实验。
- OHOS 端要重点验证低延迟双向音频、WebSocket/HTTP2、barge-in、播放队列清理。

若以“长期可维护和供应商可替换”为目标：
- 以 ChatBox 的 provider 抽象为主干。
- 将 Conversation App 中成熟的 Gemini Live/OpenAI Realtime 能力封装为可选 provider 或独立模式。

## 7. 风险对比

| 风险 | Conversation App | ChatBox |
| --- | --- | --- |
| 供应商锁定 | 高，实时后端承担多项能力 | 中，ASR/LLM/TTS 可拆换 |
| 端侧音频复杂度 | 中，后端承担更多 | 高，需要自己管理 VAD/TTS/playback |
| 本地模型迁移 | 主要集中在 vision | ASR/TTS/VAD/NER/vision 都可能涉及 |
| OHOS UI 替换 | Gradio/FastRTC 替换 | Gradio + Cascade UI 替换 |
| 机器人 SDK 替换 | 高 | 高 |
| 自动化测试 | 一般 | 较好，有 autotest/provider sweep |

## 8. 结论

ChatBox 可以看作 Conversation App 的“可组合 Cascade 改造版”。它没有替代原工程的全部价值，而是把对话链路拆成了更容易控制、测试和迁移的三个 provider 阶段，并增加了 `speak` 工具、实时 transcript reactions 和自动测试链路。

对 OHOS 迁移来说，建议把两者合并看待：
- 机器人控制、动作融合、profile/tool 机制沿用共同底座。
- 主要产品路径采用 ChatBox Cascade。
- Realtime/Live 作为高自然度、高后端依赖的可选模式保留。
- 第一阶段不要迁移本地 ASR/TTS/VLM，先通过云端 provider 和 HAL 把体验跑通。
