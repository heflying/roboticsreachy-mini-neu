# AGENTS.md

This file provides guidance to AI agents when working with code in this repository.

## Project Overview

Reachy Mini ChatBox — a modular voice conversation app for the [Reachy Mini](https://github.com/pollen-robotics/reachy_mini/) robot. Uses a **cascade pipeline** (ASR → LLM → TTS) where each stage is a swappable provider. Written in Python 3.12, uses `uv` for dependency management.

## Commands

```bash
# Install (from project root)
uv venv --python python3.12 .venv
uv sync --extra cascade --group dev

# Run the app (requires Reachy Mini daemon running)
reachy-mini-conversation-app --gradio

# Lint and format
uv run ruff check . --fix
uv run ruff format .

# Type check
uv run mypy --pretty --show-error-codes .

# Run all tests
uv run pytest tests/ -v

# Run a single test file
uv run pytest tests/cascade/test_handler.py -v

# Run a single test by name
uv run pytest tests/cascade/test_handler.py::test_function_name -v

# All quality checks at once
uv run mypy --pretty --show-error-codes . && uv run ruff check . --fix && uv run pytest tests/ -v
```

## Architecture

### Two Modes: Cascade (default) vs Realtime

- **Cascade mode** (`reachy-mini-conversation-app`): ASR → LLM → TTS pipeline with swappable providers. Configured via `cascade.yaml`.
- **Realtime mode** (`--realtime`): Audio-to-audio via OpenAI Realtime API or Qwen Omni. Bypasses cascade pipeline entirely. Configured via `BACKEND_PROVIDER` and `MODEL_NAME` env vars.

### Cascade Pipeline Flow

```
Microphone → VAD → ASR → LLM → TTS → Speaker + Robot
                      ↓
              Transcript Analysis (reactions)
```

Entry point: `main.py` → `cascade/entry.py` → `CascadeHandler` (in `cascade/handler.py`)

### Provider System

Each provider type (ASR, LLM, TTS) has an abstract base class and concrete implementations:

| Type | Base class | Implementations |
|------|-----------|----------------|
| ASR | `cascade/asr/base.py:ASRProvider` | deepgram, whisper_openai, parakeet_mlx_progressive, openai_realtime_asr, qwen_realtime, zipformer_sherpa, etc. |
| LLM | `cascade/llm/base.py:LLMProvider` | openai, gemini, ollama, spark, qwen |
| TTS | `cascade/tts/base.py:TTSProvider` | kokoro, kokoro_zh, openai, elevenlabs, piper, qwen_realtime, gradium |

Providers are loaded dynamically via `cascade/provider_factory.py` — it reads `cascade.yaml`, resolves the provider name, imports the module, and instantiates the class. CLI flags (`--asr-provider`, `--llm-provider`, `--tts-provider`) override via env vars before config loads.

### Key Source Directories

```
src/reachy_mini_conversation_app/
├── main.py                  # App entry point, argparse, mode selection
├── config.py                # Global config (env vars, profiles, locked profile)
├── cascade/
│   ├── entry.py             # Cascade mode bootstrap (3 sub-modes: gradio/console/autotest)
│   ├── handler.py           # CascadeHandler — orchestrates the pipeline
│   ├── pipeline.py          # LLM response processing, tool dispatch, conversation history
│   ├── provider_factory.py  # Dynamic provider init from cascade.yaml
│   ├── config.py            # Cascade-specific config (loads cascade.yaml)
│   ├── vad.py / vad_onnx.py # Voice Activity Detection
│   ├── turn_controller.py   # Turn management (user speaking vs robot speaking)
│   ├── interrupt_coordinator.py  # Cancellation tokens for interrupt support
│   ├── speech_output.py     # TTS playback coordination
│   ├── asr/                 # ASR provider implementations
│   ├── llm/                 # LLM provider implementations
│   ├── tts/                 # TTS provider implementations
│   ├── ui/                  # Gradio UI for cascade mode
│   └── transcript_analysis/ # Live reactions (keyword/entity triggers during speech)
├── tools/                   # LLM-callable tools (dance, speak, move_head, camera, etc.)
│   ├── core_tools.py        # ToolDependencies, Tool base class, dispatch_tool_call
│   └── *.py                 # Individual tool implementations
├── memory/                  # User memory system (remember/recall facts, care tasks)
├── vision/                  # Head tracking (YOLO, MediaPipe)
├── audio/                   # HeadWobbler, SpeechTapper
├── profiles/                # Robot personalities (instructions.txt, tools.txt, reactions.yaml, custom tools)
└── prompts.py               # System prompts and template loading
```

### Configuration

- **`cascade.yaml`** (project root): Provider selection, VAD settings, provider-specific params. Required for cascade mode.
- **`.env`**: API keys, `BACKEND_PROVIDER`, `MODEL_NAME`, profile selection, external directories.
- **`config.py`**: Reads `.env` via python-dotenv, validates profile/tool name collisions between external and built-in.
- **`cascade/config.py`**: Parses `cascade.yaml`, env overrides for `CASCADE_ASR_PROVIDER`, `CASCADE_LLM_PROVIDER`, `CASCADE_TTS_PROVIDER`.

### Profiles

Profiles live in `src/reachy_mini_conversation_app/profiles/<name>/` with `instructions.txt` (required), `tools.txt`, `voice.txt`, `reactions.yaml`, and optional `*.py` custom tools. External profiles can override via `REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY`. Name collisions between external and built-in are forbidden.

### Tool System

Tools are subclasses of `Tool` (in `core_tools.py`) with an async `execute()` method. `ToolDependencies` is the DI container passed to all tools. `dispatch_tool_call()` routes by name. The `speak` tool is special — the pipeline auto-injects it when the LLM returns raw text. If no `speak` call is made after tool execution, the LLM is re-invoked to react to tool results (max depth 5).

### Conversation History

Managed in `pipeline.py` as a list of dicts (OpenAI Chat Completions format). Camera images are stored as `frame_index` references, not inline base64. Turn cancellation rolls back history to the checkpoint.

## Code Style

- **Ruff** for linting and formatting: line length 119, double quotes, space indentation
- **Ruff isort**: length-sorted imports, `reachy_mini_conversation_app` treated as local folder
- **mypy** strict mode on `src/`, relaxed on `tests/`
- **pytest-asyncio** with `asyncio_mode = "auto"`
- Tests mock `sounddevice` in `conftest.py`; set `REACHY_MINI_SKIP_DOTENV=1` for test isolation

## This Fork

This is the Neusoft fork with Chinese-language provider additions: `zipformer_sherpa` ASR, `piper_zh` TTS, `ollama` LLM, `spark` LLM (科大讯飞), `qwen_realtime` ASR/TTS, `kokoro_zh` TTS. The `.env.example` shows Chinese-provider defaults (`DASHSCOPE_API_KEY`, `CASCADE_ASR_PROVIDER=zipformer_sherpa`, `CASCADE_LLM_PROVIDER=ollama-qwen2.5-0.5b`, `CASCADE_TTS_PROVIDER=piper_zh`).

## Platform Notes

- Primary targets: Linux, macOS, Windows
- Media backend: `gstreamer` on Linux/Windows, `sounddevice_opencv` on macOS
- Apple Silicon providers (parakeet_mlx, voxtral_mlx) require `sys_platform == 'darwin'`
- NVIDIA providers (nemotron) require CUDA
- `os._exit(0)` used in cascade mode shutdown to avoid PortAudio/sounddevice segfaults
