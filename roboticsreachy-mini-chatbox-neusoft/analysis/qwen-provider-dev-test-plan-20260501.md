# Qwen Provider 开发与自动化测试方案

日期：2026-05-01

目标：在不改动现有 Cascade pipeline 主体的前提下，将 ASR、LLM、TTS 三个阶段替换为阿里百炼/千问 provider，并先完成无 key 的本地 mock 逻辑验证。

## 1. 第一阶段：无 key 逻辑验证

已新增 provider：

- `src/reachy_mini_conversation_app/cascade/asr/qwen_realtime.py`
- `src/reachy_mini_conversation_app/cascade/llm/qwen.py`
- `src/reachy_mini_conversation_app/cascade/tts/qwen_realtime.py`

已修改配置：

- `cascade.yaml`
  - `qwen_realtime_asr`
  - `qwen-plus`
  - `qwen_realtime_tts`
- `pyproject.toml`
  - `cascade_qwen = ["websockets>=13.0"]`
- `src/reachy_mini_conversation_app/cascade/config.py`
  - 增加 `DASHSCOPE_API_KEY`
- `src/reachy_mini_conversation_app/cascade/provider_factory.py`
  - 增加 `DASHSCOPE_API_KEY` 注入映射

当前实现边界：

- ASR：实现 `StreamingASRProvider`，支持 `start_stream/send_audio_chunk/get_partial_transcript/end_stream`。
- LLM：实现 `LLMProvider.generate()`，使用 DashScope OpenAI 兼容接口，支持 text delta 和 tool call delta。
- TTS：实现 `TTSProvider.synthesize()`，以整句文本输入，WebSocket 流式返回 PCM chunk。
- 暂不实现 LLM token delta 直接喂给 TTS realtime 的深度流水线。

## 2. 无 key 自动化测试

新增测试：

- `tests/cascade/test_qwen_config.py`
- `tests/cascade/test_qwen_llm.py`
- `tests/cascade/test_qwen_realtime_asr.py`
- `tests/cascade/test_qwen_realtime_tts.py`

运行命令：

```powershell
python -m pytest -p no:cacheprovider tests\cascade\test_qwen_config.py tests\cascade\test_qwen_llm.py tests\cascade\test_qwen_realtime_asr.py tests\cascade\test_qwen_realtime_tts.py
```

当前验证结果：

```text
7 passed
```

测试覆盖：

- Qwen provider 在静态配置中已注册。
- `DASHSCOPE_API_KEY` 已加入配置和 provider factory 映射。
- Qwen LLM 能接收 mock streaming response，并输出 `text_delta`、`tool_call`、`done`。
- Qwen realtime ASR 能启动 session、发送 PCM audio append、commit，并返回 final transcript。
- Qwen realtime TTS 能发送 session/text/commit，并把 base64 audio delta 解码为 PCM bytes。

## 3. 第二阶段：挂 key 后真实 provider 冒烟

准备环境：

```powershell
pip install -e ".[cascade,cascade_qwen]"
$env:DASHSCOPE_API_KEY="你的百炼 API Key"
```

Provider 单点测试建议：

```powershell
python scripts/autotest_providers.py
```

端到端脚本测试：

```powershell
python -m reachy_mini_conversation_app.main --autotest --no-camera --asr-provider qwen_realtime_asr --llm-provider qwen-plus --tts-provider qwen_realtime_tts
```

PC + sim + Gradio 测试：

```powershell
python -m reachy_mini_conversation_app.main --gradio --no-camera --asr-provider qwen_realtime_asr --llm-provider qwen-plus --tts-provider qwen_realtime_tts
```

验收标准：

- ASR 能返回用户语音文本。
- LLM 能返回自然语言回复，必要时能输出 `speak` tool call 或由 pipeline 自动注入 `speak`。
- TTS 能流式返回 PCM chunk，并被当前 SpeechOutput 播放。
- `--autotest --no-camera` 至少完成 3 轮 utterance。
- `--gradio --no-camera` 能在 PC + sim 下完成对话。

## 4. 后续增强

后续如要进一步降低首字延迟，需要新增跨阶段流式编排：

```text
LLM text_delta -> TTS realtime append_text -> audio delta playback
```

这会涉及：

- `cascade/pipeline.py`
- `cascade/speech_output.py`
- `cascade/tts/base.py`

该增强不属于第一阶段最小替换范围。
