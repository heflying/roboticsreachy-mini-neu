---
title: Reachy Mini ChatBox
emoji: 💬
colorFrom: red
colorTo: blue
sdk: static
pinned: false
short_description: Live chat with Reachy Mini!
suggested_storage: large
tags:
 - reachy_mini
 - reachy_mini_python_app
---

# Reachy Mini ChatBox

A modular voice conversation app for the [Reachy Mini](https://github.com/pollen-robotics/reachy_mini/) robot. Cheap to run, versatile, customizable, and fun.

ChatBox uses a **cascade pipeline** — ASR → LLM → TTS — where each stage is a swappable provider. Mix cloud APIs and local models, tweak the personality, add live reactions to keywords, and let the robot dance.

![Reachy Mini Dance](docs/assets/reachy_mini_dance.gif)

## How it works

```
🎤 Microphone
    │
    ▼
┌─────────┐     ┌─────────┐     ┌─────────┐     🔊 Speaker
│   ASR   │ ──▶ │   LLM   │ ──▶ │   TTS   │ ──▶ 🤖 Robot moves
└─────────┘     └─────────┘     └─────────┘
 Speech to       Reasoning       Text to
 text            + tool calls    speech
```

Voice Activity Detection (VAD) segments microphone input. The ASR provider transcribes it, the LLM generates a response (with optional tool calls for head movements, dances, emotions, camera…), and the TTS provider speaks it back — all while the robot animates.

## Installation

> [!IMPORTANT]
> Before using this app, install [Reachy Mini's SDK](https://github.com/pollen-robotics/reachy_mini/).

<details open>
<summary><b>Using uv (recommended)</b></summary>

```bash
# Create venv
uv venv --python python3.12 .venv
source .venv/bin/activate

# Install with cascade pipeline
uv sync --extra cascade
```

Install additional providers as needed:

```bash
uv sync --extra cascade_parakeet_progressive  # Parakeet ASR (Apple Silicon)
uv sync --extra cascade_kokoro                # Kokoro TTS (local)
uv sync --extra cascade_gemini                # Gemini LLM
uv sync --extra cascade_elevenlabs            # ElevenLabs TTS
uv sync --extra cascade_deepgram              # Deepgram ASR
uv sync --extra cascade_parakeet              # Parakeet batch/RNNT ASR (Apple Silicon)
uv sync --extra cascade_voxtral_mlx           # Voxtral ASR (Apple Silicon)
uv sync --extra cascade_nemotron              # NeMo ASR (CUDA)
uv sync --extra cascade_gradium               # Gradium TTS (Python ≥3.12)
uv sync --extra cascade_all                   # All cascade providers
```

Vision and head-tracking extras:

```bash
uv sync --extra yolo_vision          # YOLO head-tracking
uv sync --extra mediapipe_vision     # MediaPipe head-tracking
uv sync --extra local_vision         # Local VLM (SmolVLM2)
uv sync --extra all_vision           # All vision features
```

Combine extras freely:

```bash
uv sync --extra cascade --extra cascade_kokoro --extra cascade_gemini --extra yolo_vision --group dev
```

> **Tip:** Use `uv sync --frozen` to install from the lockfile without re-resolving.

</details>

<details>
<summary><b>Using pip</b></summary>

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[cascade]"

# Add providers
pip install -e ".[cascade_kokoro]"
pip install -e ".[cascade_gemini]"
# etc.
```

</details>

### Optional extras reference

| Extra | Purpose | Notes |
|-------|---------|-------|
| `cascade` | Base cascade pipeline (sounddevice, librosa, PyYAML) | Required for cascade mode |
| `cascade_parakeet_progressive` | Parakeet MLX progressive ASR | Apple Silicon only |
| `cascade_kokoro` | Kokoro local TTS | Any PyTorch platform |
| `cascade_gemini` | Google Gemini LLM | Requires `GEMINI_API_KEY` |
| `cascade_deepgram` | Deepgram streaming ASR | Requires `DEEPGRAM_API_KEY` |
| `cascade_elevenlabs` | ElevenLabs TTS | Requires `ELEVENLABS_API_KEY` |
| `cascade_parakeet` | Parakeet batch/RNNT ASR | Apple Silicon only |
| `cascade_voxtral_mlx` | Voxtral Mini 4B multilingual ASR | Apple Silicon only |
| `cascade_nemotron` | NeMo Parakeet/Nemotron ASR | CUDA GPU required |
| `cascade_silero_vad` | Silero VAD (torch-based) | Optional VAD backend |
| `cascade_gradium` | Gradium streaming TTS | Python ≥3.12, requires `GRADIUM_API_KEY` |
| `cascade_all` | All cascade providers | Excludes gradium (Python ≥3.12) |
| `local_vision` | Local VLM (SmolVLM2) via PyTorch/Transformers | GPU recommended |
| `yolo_vision` | YOLO head-tracking | CPU or GPU |
| `mediapipe_vision` | MediaPipe head-tracking | CPU |
| `all_vision` | All vision extras | — |

## Configuration

### Environment variables

Copy `.env.example` to `.env` and fill in the API keys for the providers you use:

| Variable | Used by |
|----------|---------|
| `OPENAI_API_KEY` | Whisper ASR, OpenAI Realtime ASR, GPT LLMs, OpenAI TTS |
| `GEMINI_API_KEY` | Gemini LLMs |
| `DEEPGRAM_API_KEY` | Deepgram ASR |
| `ELEVENLABS_API_KEY` | ElevenLabs TTS |
| `GRADIUM_API_KEY` | Gradium TTS |
| `HF_HOME` | Cache dir for Hugging Face models (default: `./cache`) |
| `HF_TOKEN` | Hugging Face access for gated models |
| `REACHY_MINI_CUSTOM_PROFILE` | Select a profile (default: `default`) |
| `REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY` | Path to external profiles |
| `REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY` | Path to external tool modules |
| `AUTOLOAD_EXTERNAL_TOOLS` | Set `1` to auto-load all external tools |

### cascade.yaml

The `cascade.yaml` file at the project root controls which providers are used and their settings. Structure:

```yaml
asr:
  provider: parakeet_mlx_progressive   # Which ASR to use
  providers:
    parakeet_mlx_progressive:          # Provider definitions
      module: parakeet_mlx_progressive
      class: ParakeetMLXProgressiveASR
      streaming: true
      location: local
      hardware: apple_silicon
      # Provider-specific settings
      model: mlx-community/parakeet-tdt-0.6b-v3
      precision: float16

llm:
  provider: gemini-2.5-flash-lite      # Which LLM to use
  temperature: 1.0
  providers: { ... }

tts:
  provider: kokoro                     # Which TTS to use
  trim_silence: true
  providers: { ... }
```

Change `provider:` under each section to switch. You can also override from the CLI with `--asr-provider`, `--llm-provider`, or `--tts-provider`.

#### ASR providers

| Provider | Location | Streaming | Hardware | Install extra | API key |
|----------|----------|-----------|----------|---------------|---------|
| `parakeet_mlx_progressive` | Local | Yes | Apple Silicon | `cascade_parakeet_progressive` | — |
| `voxtral_mlx` | Local | Yes | Apple Silicon | `cascade_voxtral_mlx` | — |
| `parakeet_nemo_progressive` | Local | Yes | CUDA | `cascade_nemotron` | — |
| `nemotron` | Local | Yes | CUDA | `cascade_nemotron` | — |
| `deepgram` | Cloud | Yes | — | `cascade_deepgram` | `DEEPGRAM_API_KEY` |
| `openai_realtime_asr` | Cloud | Yes | — | — | `OPENAI_API_KEY` |
| `whisper_openai` | Cloud | No (batch) | — | — | `OPENAI_API_KEY` |

#### LLM providers

| Provider | Location | Model | Install extra | API key |
|----------|----------|-------|---------------|---------|
| `gemini-2.5-flash-lite` | Cloud | `gemini-2.5-flash-lite` | `cascade_gemini` | `GEMINI_API_KEY` |
| `gemini-3.1-flash-lite` | Cloud | `gemini-3.1-flash-lite-preview` | `cascade_gemini` | `GEMINI_API_KEY` |
| `gpt-4o-mini` | Cloud | `gpt-4o-mini` | — | `OPENAI_API_KEY` |
| `gpt-5.2-chat` | Cloud | `gpt-5.2-chat-latest` | — | `OPENAI_API_KEY` |

#### TTS providers

| Provider | Location | Hardware | Install extra | API key |
|----------|----------|----------|---------------|---------|
| `kokoro` | Local | Any (PyTorch) | `cascade_kokoro` | — |
| `tts_openai` | Cloud | — | — | `OPENAI_API_KEY` |
| `elevenlabs` | Cloud | — | `cascade_elevenlabs` | `ELEVENLABS_API_KEY` |
| `gradium` | Cloud | — | `cascade_gradium` | `GRADIUM_API_KEY` |

## Running the app

> [!TIP]
> Make sure the Reachy Mini daemon is running before launching. See [Reachy Mini's SDK](https://github.com/pollen-robotics/reachy_mini/) for setup.

```bash
reachy-mini-conversation-app --gradio
```

### CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--gradio` | `False` | Launch Gradio web UI at http://127.0.0.1:7860/. Without this, runs in console mode with VAD. |
| `--asr-provider NAME` | from yaml | Override ASR provider (e.g. `deepgram`, `whisper_openai`) |
| `--llm-provider NAME` | from yaml | Override LLM provider (e.g. `gpt-4o-mini`, `gemini-2.5-flash-lite`) |
| `--tts-provider NAME` | from yaml | Override TTS provider (e.g. `kokoro`, `elevenlabs`) |
| `--head-tracker {yolo,mediapipe}` | `None` | Enable head-tracking. Requires the matching vision extra. |
| `--no-camera` | `False` | Run without camera. |
| `--local-vision` | `False` | Use local VLM (SmolVLM2) instead of cloud vision. Requires `local_vision` extra. |
| `--autotest [FILE]` | — | Run automated testing with text utterances (default file: `cascade/autotest.txt`). |
| `--realtime` | `False` | Use OpenAI realtime audio-to-audio API instead of cascade. |
| `--robot-name NAME` | `None` | Connect to a specific robot when multiple daemons run on the same subnet. |
| `--debug` | `False` | Verbose logging. |

### Examples

```bash
# Gradio UI with YOLO head-tracking
reachy-mini-conversation-app --gradio --head-tracker yolo

# Console mode (no UI, VAD-based)
reachy-mini-conversation-app

# Override providers from CLI
reachy-mini-conversation-app --gradio --asr-provider deepgram --llm-provider gpt-4o-mini

# Audio-only (no camera)
reachy-mini-conversation-app --gradio --no-camera

# Automated testing
reachy-mini-conversation-app --autotest
```

## Profiles

Profiles define the robot's personality: what it says, how it sounds, and which tools it can use.

Each profile is a folder under `src/reachy_mini_conversation_app/profiles/<name>/` containing:

| File | Required | Purpose |
|------|----------|---------|
| `instructions.txt` | Yes | System prompt — the robot's personality and behavior rules |
| `tools.txt` | Recommended | Enabled tools, one per line. Falls back to `default/tools.txt` if missing. |
| `voice.txt` | Optional | TTS voice name override (single line) |
| `reactions.yaml` | Optional | Live reaction triggers (see [Live reactions](#live-reactions)) |
| `*.py` | Optional | Custom tool implementations |

### Selecting a profile

- **Environment variable:** `REACHY_MINI_CUSTOM_PROFILE=pirate` in `.env`
- **Gradio UI:** Open the "Personality" accordion to switch profiles, edit instructions, or create new ones.

### Template placeholders in instructions.txt

Reuse shared prompt snippets by referencing files under `src/reachy_mini_conversation_app/prompts/`:

```
[passion_for_lobster_jokes]
[identities/witty_identity]
```

### Locked profile mode

Set `LOCKED_PROFILE` in `src/reachy_mini_conversation_app/config.py` to lock the app to a single profile. The Gradio UI shows "(locked)" and disables profile editing. Useful for dedicated app variants.

## Live reactions

Reactions let the robot respond to keywords or entities in real time — *while the user is still speaking* — without waiting for the LLM.

Define them in `profiles/<name>/reactions.yaml`:

```yaml
- name: music_excitement
  callback: excited_about_music
  trigger:
    words: [music, guitar, piano, drum, violin]

- name: food_reaction
  callback: react_to_food_entity
  trigger:
    entities: [food]
  repeatable: true

- name: groovy_dance
  callback: do_groovy_dance
  trigger:
    all:
      - words: [danc*]
      - words: [groov*]

- name: reachy_name
  callback: react_to_name
  trigger:
    words: [reachy, richie, reechy]
  params:
    emotion: helpful1
```

### Trigger types

| Type | Description | Example |
|------|-------------|---------|
| `words` | Keyword/glob match on transcript | `[music, danc*, "grand piano"]` |
| `entities` | Named entity recognition via GLiNER | `[food, person, location]` |
| `all` | Boolean AND — all sub-triggers must match | See `groovy_dance` above |

### Behavior

- Reactions fire **at most once per conversation turn** by default.
- Set `repeatable: true` to allow multiple triggers per turn (deduplicated by entity text for entity triggers).
- `params` are passed as `**kwargs` to the callback.

### Callback signature

Each `callback` value maps to a Python module in the profile folder exporting an async function:

```python
async def my_callback(deps: ToolDependencies, match: TriggerMatch, **kwargs) -> None:
    ...
```

`match.words` contains matched keywords; `match.entities` contains matched entities (with `.text`, `.label`, `.confidence`).

## LLM tools

Tools the LLM can call during a conversation:

| Tool | Action |
|------|--------|
| `speak` | Synthesize and play a speech segment (used internally by the pipeline) |
| `dance` | Queue a dance from the dances library |
| `stop_dance` | Clear queued dances |
| `play_emotion` | Play a recorded emotion clip from [pollen-robotics/reachy-mini-emotions-library](https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library) |
| `stop_emotion` | Clear queued emotions |
| `move_head` | Move head to a named position (left/right/up/down/front) |
| `see_image_through_camera` | Capture a camera frame and send it to the LLM for analysis |
| `describe_camera_image` | Describe the current camera view |
| `head_tracking` | Enable/disable head-tracking (requires `--head-tracker`) |
| `task_status` | Check status of a background task |
| `task_cancel` | Cancel a running background task |
| `do_nothing` | Explicitly remain idle |

Profiles can also define custom tools (e.g. `turn_left`, `turn_right`, `center_position` in the default profile).

## Advanced features

<details>
<summary><b>External profiles and tools</b></summary>

Store profiles and tools outside the source tree:

```text
external_content/
├── external_profiles/
│   └── my_profile/
│       ├── instructions.txt
│       ├── tools.txt
│       └── voice.txt
└── external_tools/
    └── my_custom_tool.py
```

Set in `.env`:

```env
REACHY_MINI_CUSTOM_PROFILE=my_profile
REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY=./external_content/external_profiles
REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY=./external_content/external_tools
```

- **Default mode:** `tools.txt` must list every tool explicitly. Names resolve against built-in tools first, then external tools.
- **Auto-load mode** (`AUTOLOAD_EXTERNAL_TOOLS=1`): all `*.py` modules in the external tools directory are loaded automatically.

</details>

<details>
<summary><b>Multiple robots on the same subnet</b></summary>

```bash
reachy-mini-conversation-app --robot-name <name>
```

`<name>` must match the daemon's `--robot-name` value.

</details>

<details>
<summary><b>Autotest mode</b></summary>

Run the full pipeline with synthetic text utterances instead of a microphone:

```bash
reachy-mini-conversation-app --autotest
reachy-mini-conversation-app --autotest my_test_script.txt
```

Each line in the test file is treated as a user utterance. Useful for end-to-end testing without audio hardware.

</details>

<details>
<summary><b>OpenAI Realtime mode (legacy)</b></summary>

The original audio-to-audio mode using OpenAI's realtime API is still available:

```bash
reachy-mini-conversation-app --realtime --gradio
```

This bypasses the cascade pipeline entirely. Requires `OPENAI_API_KEY`.

</details>

## Contributing

We welcome bug fixes, features, profiles, and documentation improvements. Please review our
[contribution guide](CONTRIBUTING.md) for branch conventions, quality checks, and PR workflow.

Quick start:
- Fork and clone the repo
- Follow the [installation steps](#installation) (include the `dev` dependency group)
- Run contributor checks listed in [CONTRIBUTING.md](CONTRIBUTING.md)

## License

Apache 2.0
