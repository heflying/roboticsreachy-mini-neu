# Cascade Mode - Developer Documentation

This document provides detailed technical documentation for the cascade conversation mode in the Reachy Mini Conversation App.

## Overview

Cascade Mode implements a traditional **ASR → LLM → TTS** conversation pipeline for the Reachy Mini robot. It processes user speech through automatic speech recognition, generates AI responses via configurable LLM providers, and synthesizes speech output with synchronized head animations.

### Key Design Principles

- **Provider abstraction** - Swap ASR/LLM/TTS implementations via configuration
- **Latency optimization** - Pre-warmed audio playback threads, parallel TTS generation
- **Clean separation** - Handler (business logic) vs UI (presentation & I/O)

### Current Scope

- Gradio UI with continuous VAD recording
- Console mode with VAD-based speech detection
- Test file mode for automated end-to-end testing (`--test-file`)
- Optional streaming ASR support
- Head wobbler animation synchronized with audio playback

---

## File Structure

```
cascade/
├── __init__.py                        # Package init (no public exports)
├── entry.py                           # Entry point from main.py
├── handler.py                         # Core orchestrator (lifecycle, ASR routing, state)
├── provider_factory.py                # Provider initialization (ASR/LLM/TTS factory functions)
├── pipeline.py                        # LLM response processing & tool execution
├── speech_output.py                   # SpeechOutput protocol + Gradio/Console implementations
├── turn_result.py                     # TurnResult + TurnItem dataclasses
├── config.py                          # Configuration loader (cascade.yaml)
├── timing.py                          # Latency tracking & profiling
├── vad.py                             # Silero VAD + VADStateMachine (shared by console & Gradio)
├── console.py                         # Console mode with VAD (CascadeLocalStream)
├── autotest_stream.py                     # Test file mode (automated TTS→ASR→LLM→TTS testing)
│
├── ui/                                # Gradio interface components
│   ├── __init__.py                    # Exports CascadeGradioUI
│   ├── audio_playback.py              # Pre-warmed audio output system
│   ├── audio_recording.py             # VAD-based continuous recording
│   └── gradio_app.py                  # Main Gradio interface
│
├── asr/                               # Automatic Speech Recognition
│   ├── __init__.py                    # Provider exports
│   ├── base.py                        # ASRProvider abstract base
│   ├── base_streaming.py              # StreamingASRProvider abstract base
│   ├── audio_utils.py                 # Shared WAV parsing & resampling (librosa)
│   ├── progressive_base.py            # ProgressiveASRBase — shared sliding window logic + DecodeResult/SentenceSegment
│   ├── whisper_openai.py              # OpenAI Whisper implementation
│   ├── parakeet_mlx_progressive.py    # Parakeet MLX progressive (Apple Silicon, inherits ProgressiveASRBase)
│   ├── parakeet_nemo_progressive.py   # Parakeet NeMo progressive (CUDA, inherits ProgressiveASRBase)
│   ├── deepgram.py                    # Deepgram streaming implementation
│   ├── nemotron.py                    # Nemotron ASR implementation
│   └── openai_realtime_asr.py         # OpenAI Realtime streaming implementation
│
├── llm/                               # Large Language Models
│   ├── __init__.py                    # Provider exports
│   ├── base.py                        # LLMProvider + LLMChunk abstractions
│   ├── openai.py                      # OpenAI GPT implementation
│   └── gemini.py                      # Google Gemini implementation
│
├── tts/                               # Text-to-Speech
│   ├── __init__.py                    # Provider exports
│   ├── base.py                        # TTSProvider abstract base
│   ├── utils.py                       # Shared TTS utilities (silence trimming)
│   ├── openai.py                      # OpenAI TTS implementation
│   ├── kokoro.py                      # Kokoro local TTS
│   └── elevenlabs.py                  # ElevenLabs API implementation
│
└── transcript_analysis/               # Real-time reactions to user speech
    ├── __init__.py                    # Package exports (EntityAnalyzer optional)
    ├── base.py                        # Data classes + TranscriptAnalyzer ABC
    ├── loader.py                      # YAML loader, callback importer
    ├── manager.py                     # Orchestrator, deduplication, dispatch
    ├── keyword_analyzer.py            # Keyword/glob matching
    └── entity_analyzer.py             # NER via GLiNER (optional dependency)
```

---

## Core Components

### CascadeHandler (`handler.py`)

The central orchestrator that manages conversation state, ASR routing, and lifecycle.
Provider initialization is in `provider_factory.py`; LLM/tool pipeline logic is in `pipeline.py`.

**Responsibilities:**
- Manage conversation history (OpenAI message format)
- Route audio to ASR (manual and streaming paths)
- Transcript analysis callbacks (fire-and-forget)
- Cost tracking
- Run async event loop in background thread

**Key Attributes:**
```python
deps: ToolDependencies            # Tool dependencies (robot, vision, movement, etc.)
speech_output: SpeechOutput|None  # TTS playback backend (set by console or Gradio frontend)
asr: ASRProvider                  # Current ASR implementation
llm: LLMProvider                  # Current LLM implementation
tts: TTSProvider                  # Current TTS implementation
conversation_history: List        # OpenAI format: [{"role": "...", "content": ...}]
processing_lock: asyncio.Lock     # Prevent concurrent audio processing
tool_specs: List[Dict]            # Tools in Chat Completions format
is_streaming_asr: bool            # Whether current ASR provider supports streaming
transcript_manager                # TranscriptAnalysisManager | NoOpTranscriptManager
_captured_frames: list[bytes]     # Side-channel storage for see_image_through_camera JPEG frames
_current_turn_items: list[TurnItem]  # Per-turn accumulator for displayable items
_turn_results: list[TurnResult]   # Completed turns (private; use turn_results property)
cumulative_cost: float            # Running cost total across all turns
```

**Key Properties:**
```python
turn_results -> list[TurnResult]
    # Public read-only accessor for completed turns (used by UI poller)
```

**Key Methods:**
```python
async process_audio_manual(audio_bytes) -> TurnResult
    # Entry point for push-to-talk recording

async process_audio_streaming_start() -> None
    # Initialize streaming ASR session

async process_audio_streaming_chunk(chunk) -> Optional[str]
    # Send chunk to streaming ASR, get partial transcript

async process_audio_streaming_end() -> TurnResult
    # Finalize streaming ASR, run LLM pipeline

def start() -> None
    # Start background event loop (Gradio mode only)

def stop() -> None
    # Stop background event loop

def clear_state() -> None
    # Reset conversation history, captured frames, and turn results

async _run_pipeline_after_transcription(transcript) -> TurnResult
    # Shared post-ASR pipeline (validate → history → LLM → TTS → result)
    # Called by both manual and streaming paths
```

### Provider Factory (`provider_factory.py`)

Pure factory functions for constructing ASR/LLM/TTS providers and transcript analysis
from `cascade.yaml` config. No runtime state — called once during `CascadeHandler.__init__`.

**Key Functions:**
```python
init_asr_provider() -> ASRProvider
init_llm_provider() -> LLMProvider
init_tts_provider() -> TTSProvider
init_transcript_analysis(deps) -> TranscriptAnalysisManager | NoOpTranscriptManager
```

### Pipeline (`pipeline.py`)

Module-level async functions for LLM response processing and tool execution.
Mutates conversation history, turn items, and captured frames in-place via `PipelineContext`.

**PipelineContext:**
```python
@dataclass
class PipelineContext:
    llm: LLMProvider
    tts: TTSProvider
    speech_output: SpeechOutput | None
    conversation_history: list[dict[str, Any]]
    tool_specs: list[dict[str, Any]]
    current_turn_items: list[TurnItem]
    captured_frames: list[bytes]
    deps: ToolDependencies
    aggregate_cost_fn: Callable
```

Built once in `handler._run_pipeline_after_transcription()` and threaded through all pipeline calls.

**Key Functions:**
```python
async process_llm_response(ctx: PipelineContext) -> None
    # Stream LLM, collect text/tool calls, dispatch to execute_tool_calls
    # Auto-injects a synthetic speak tool call if the LLM returns text without
    # using the speak tool (fallback for models that skip the tool)

async execute_tool_calls(tool_calls, ctx: PipelineContext) -> None
    # Execute individual tools, handle see_image_through_camera/speak specially:
    #   speak → calls speech_output.speak() for TTS synthesis + playback
    #   see_image_through_camera → stores JPEG in captured_frames, replaces b64 in history, re-calls LLM for analysis
```

### TurnResult (`turn_result.py`)

Structured result returned by the handler after each conversation turn. Decouples handler output from UI rendering — the UI never parses `conversation_history` directly.

```python
@dataclass
class TurnItem:
    kind: str         # "speak" | "image" | "tool" | "assistant"
    text: str         # For speak/assistant items
    image_jpeg: bytes # For image items (raw JPEG bytes)
    tool_name: str    # For tool items
    tool_content: str # For tool items (JSON string)

@dataclass
class TurnResult:
    transcript: str           # User's speech transcript
    items: list[TurnItem]     # Ordered displayable items
    cost: float               # ASR + LLM cost for this turn (not TTS)

    speak_text: str           # (property) All speak items joined with ". "
    has_speak: bool           # (property) Whether any speak item exists
```

**How items are populated:**
- `speak` — from `pipeline.execute_tool_calls()` when speak tool is called
- `image` — from `pipeline.execute_tool_calls()` when see_image_through_camera tool returns JPEG
- `tool` — from `pipeline.execute_tool_calls()` for other tools (movements, etc.)
- `assistant` — from `pipeline.process_llm_response()` when LLM returns text + tool calls but no speak

The handler stores completed turns in `_turn_results` (exposed as `handler.turn_results` property), used by the UI's continuous-mode poller.

### UI Components (`ui/`)

The Gradio interface is split into focused modules:

#### AudioPlaybackSystem (`ui/audio_playback.py`)

Pre-warmed audio output system for low-latency TTS playback.

**Responsibilities:**
- Detect playback backend (sounddevice vs robot.media)
- Manage persistent playback and wobbler threads
- Queue-based communication for zero-startup-cost playback

**Key Interface:**
```python
playback = AudioPlaybackSystem(robot, head_wobbler, shutdown_event)
playback.put_audio(chunk)      # Queue audio for playback
playback.put_wobbler(chunk)    # Queue wobbler data
playback.signal_end_of_turn()  # Signal end of speech
playback.close()               # Shutdown threads
```

#### Recording Classes (`ui/audio_recording.py`)

**ContinuousState:** Enum for Gradio VAD lifecycle (IDLE → LISTENING → RECORDING → PROCESSING). IDLE is Gradio-specific; LISTENING/RECORDING/PROCESSING map to `VADState` values.

**StreamingASRCallbacks:** Dataclass for injecting ASR callbacks without coupling to handler.

**ContinuousVADRecorder:** VAD-based continuous recording, backed by `VADStateMachine`.
```python
recorder = ContinuousVADRecorder(
    sample_rate, streaming_callbacks, on_speech_captured
)
recorder.start()   # Start VAD loop (creates VADStateMachine)
recorder.stop()    # Stop VAD loop
recorder.state     # Current ContinuousState (maps from VADStateMachine.state)
```

#### CascadeGradioUI (`ui/gradio_app.py`)

Main orchestrator that ties everything together.

**Responsibilities:**
- Build Gradio interface (chatbot, buttons, status)
- Coordinate audio pipeline (ASR → LLM → TTS)
- Coordinate playback and VAD recording subsystems
- Handle Gradio events and lifecycle

**Key Attributes:**
```python
handler: CascadeHandler        # Reference to cascade handler
robot: ReachyMini | None       # Robot instance
playback: AudioPlaybackSystem  # Audio output
shutdown_event: threading.Event # Coordinated shutdown
```

**Key Methods:**
```python
create_interface() -> gr.Blocks     # Build Gradio UI
launch(**kwargs) -> None            # Start server
close() -> None                     # Shutdown all subsystems
```

### SpeechOutput (`speech_output.py`)

Protocol and implementations for TTS synthesis + playback. The handler calls `speech_output.speak(text)` inside `execute_tool_calls` when the `speak` tool fires — this is where TTS actually happens.

**Protocol:**
```python
class SpeechOutput(Protocol):
    async def speak(text: str) -> None
```

**Implementations:**

| Class | Used by | Behavior |
|-------|---------|----------|
| `ConsoleSpeechOutput` | `CascadeLocalStream`, `CascadeTestStream` | Streams TTS chunks to a playback callback with rate limiting; drives head wobbler per chunk |
| `GradioSpeechOutput` | `CascadeGradioUI` | Splits text into sentences via `split_into_sentences()`, generates TTS in parallel with gate-event ordering, queues to `AudioPlaybackSystem` |

**`split_into_sentences(text, min_length=8)`** — Splits on `.!?,;—`, keeps punctuation attached, merges short segments under `min_length` characters.

### VAD State Machine (`vad.py`)

`VADStateMachine` extracts the shared pre-roll → speech-start → speech-end logic used by both console and Gradio modes. Callers feed audio chunks via `process_chunk()` and react to returned `VADEvent`s.

```python
vad_sm = VADStateMachine(vad)
event = vad_sm.process_chunk(audio_chunk)

if event == VADEvent.SPEECH_STARTED:
    # Pre-roll + current chunk available in vad_sm.speech_chunks
elif event == VADEvent.SPEECH_ENDED:
    # All speech frames in vad_sm.speech_chunks
    vad_sm.finish_processing()  # Reset to LISTENING
elif vad_sm.state == VADState.RECORDING:
    # Mid-recording: stream current chunk
```

**States:** `LISTENING` → `RECORDING` → `PROCESSING` → (finish_processing) → `LISTENING`

### Console Mode (`console.py`)

VAD-based console interface for the cascade pipeline, used when neither `--gradio` nor `--test-file` is specified. Records from system mic, detects speech via `VADStateMachine`, and plays responses through the robot speaker.

**Key Classes:**

- `CascadeLocalStream` — Stream manager that runs two concurrent async loops:
  - `_record_loop()` — reads mic frames, processes through `VADStateMachine`
  - `_play_loop()` — pulls audio from playback queue, resamples to output rate, pushes to `robot.media`

```python
stream = CascadeLocalStream(handler, robot)
stream.launch()   # Blocking: runs asyncio event loop
stream.close()    # Stop media, cancel tasks
```

Wires `ConsoleSpeechOutput` into `handler.speech_output` at init time so the pipeline plays audio through the robot speaker.

---

## Provider Abstractions

### ASR Providers (`asr/`)

**Base class** (`base.py`):
```python
class ASRProvider(ABC):
    async transcribe(audio_bytes, language) -> str
```

**Streaming base** (`base_streaming.py`):
```python
class StreamingASRProvider(ASRProvider):
    async start_stream() -> None
    async send_audio_chunk(audio_chunk: bytes) -> None
    async get_partial_transcript() -> Optional[str]
    async end_stream() -> str                  # Final transcript
```

**Implementations:**
| Provider | Type | Description |
|----------|------|-------------|
| `WhisperOpenAIASR` | Batch | OpenAI Whisper API |
| `ParakeetMLXProgressiveASR` | Streaming | Local progressive with sliding window via mlx-audio (Apple Silicon) |
| `ParakeetNeMoProgressiveASR` | Streaming | Local progressive with sliding window via NeMo (CUDA) |
| `DeepgramASR` | Streaming | Deepgram Nova via WebSocket |
| `NemotronASR` | Streaming | NVIDIA Nemotron ASR |
| `OpenAIRealtimeASR` | Streaming | OpenAI Realtime API via WebSocket |

**Progressive ASR Base** (`progressive_base.py`):

Both Parakeet providers inherit from `ProgressiveASRBase`, which contains the sentence-aware sliding window logic (~150 lines). Subclasses only implement three methods:
- `_decode(audio_np) -> DecodeResult` — run inference, return text + sentence segments
- `_decode_full(audio_np) -> str` — full-context decode for final transcription
- `_warmup()` — model warmup (e.g. transcribe silence)

`DecodeResult` and `SentenceSegment` are shared dataclasses defined in `progressive_base.py`.

### LLM Providers (`llm/`)

**Base class** (`base.py`):
```python
@dataclass
class LLMChunk:
    type: str  # "text_delta" | "tool_call" | "done"
    content: Optional[str]      # For text_delta
    tool_call: Optional[Dict]   # For tool_call

class LLMProvider(ABC):
    async generate(messages, tools, temperature) -> AsyncIterator[LLMChunk]
    def parse_tool_call(tool_call) -> (call_id, tool_name, args_dict)
        # Default implementation handles OpenAI-style tool call format
        # Subclasses can override if needed
```

**Implementations:**
| Provider | Models |
|----------|--------|
| `OpenAILLM` | GPT-4o-mini, GPT-5.2-chat (any OpenAI chat model) |
| `GeminiLLM` | Gemini 2.5 Flash Lite (any Gemini model) |

### TTS Providers (`tts/`)

**Base class** (`base.py`):
```python
class TTSProvider(ABC):
    @property
    def sample_rate(self) -> int:  # Default 24000; override for non-24kHz providers
    async synthesize(text, voice=None) -> AsyncIterator[bytes]
```

**Implementations:**
| Provider | Description |
|----------|-------------|
| `OpenAITTS` | OpenAI TTS API (24kHz PCM int16) |
| `KokoroTTS` | Local TTS via Kokoro model |
| `ElevenLabsTTS` | ElevenLabs API |

---

## Data Flow

### VAD-Triggered Pipeline

```
VAD detects speech end
  │
  ├─ Concatenate captured frames to WAV
  │
  └─> handler.process_audio_manual(wav_bytes)
      │
      ├─ PHASE 1: ASR
      │   └─ asr.transcribe(audio_bytes) → transcript
      │       └─ Add to conversation_history as user message
      │
      ├─ PHASE 2: LLM
      │   └─ llm.generate(history, tools) → response + tool_calls
      │       └─ Add assistant message to history
      │       └─ If text-only (no tool calls): auto-inject speak tool call
      │
      ├─ PHASE 3: Tool Execution + TTS
      │   └─ For each tool_call:
      │       ├─ Execute tool, add result to history
      │       └─ Special handling:
      │           • speak: call speech_output.speak(message)
      │             └─ TTS synthesis + playback happens HERE
      │           • see_image_through_camera: store JPEG, replace b64 in history, re-call LLM for analysis
      │
      └─> Return TurnResult to UI
          (transcript, items=[speak/image/tool/assistant], cost)

UI uses TurnResult.items for display only:
  │
  ├─ speak items → show in chatbot
  ├─ image items → decode JPEG, display
  └─ tool items → display with metadata
```

### Streaming ASR Flow

```
User clicks START (streaming provider)
  │
  ├─ handler.process_audio_streaming_start()
  │   └─ Open WebSocket/stream connection
  │
During recording:
  │
  ├─ For each audio chunk:
  │   ├─ handler.process_audio_streaming_chunk(chunk)
  │   │   └─ Get partial transcript
  │   └─ Update UI with partial text
  │
User clicks STOP
  │
  └─ handler.process_audio_streaming_end()
      ├─ Close stream
      ├─ Get final transcript
      └─ Run LLM pipeline (same as batch mode)
```

---

## Configuration

### Config File: `cascade.yaml`

Each section has a `provider:` key selecting the active provider and a `providers:` dict defining all available providers. Each provider entry contains metadata keys (`module`, `class`, `streaming`, `location`, `requires`, `hardware`, `description`, `import_check`, `install_extra`) plus provider-specific settings.

#### Hardware Tags

| Tag | Validation | Used by |
|---|---|---|
| `apple_silicon` | Hard error if not arm64 + Darwin | parakeet_mlx_progressive |
| `cuda` | Hard error if `torch.cuda.is_available()` is False | parakeet_nemo_progressive, nemotron |
| `null` | No check | kokoro, cloud providers |

#### Dependency Checks

Local providers can declare `import_check` (module to import) and `install_extra` (uv extra name). At config load time, if the import fails, a `RuntimeError` is raised with install instructions.

```yaml
asr:
  provider: parakeet_mlx_progressive  # Selected provider name
  providers:
    whisper_openai:
      module: whisper_openai
      class: WhisperOpenAIASR
      streaming: false
      requires: [OPENAI_API_KEY]
      model: whisper-1
    deepgram:
      module: deepgram
      class: DeepgramASR
      streaming: true
      requires: [DEEPGRAM_API_KEY]
      model: nova-2
    # ... other providers (parakeet_mlx_progressive, nemotron, openai_realtime_asr)

llm:
  provider: gemini-2.5-flash-lite
  temperature: 1.0
  providers:
    gpt-4o-mini:
      module: openai
      class: OpenAILLM
      requires: [OPENAI_API_KEY]
      model: gpt-4o-mini
    gemini-2.5-flash-lite:
      module: gemini
      class: GeminiLLM
      requires: [GEMINI_API_KEY]
      model: gemini-2.5-flash-lite

tts:
  provider: kokoro
  trim_silence: true
  providers:
    tts_openai:
      module: openai
      class: OpenAITTS
      requires: [OPENAI_API_KEY]
      voice: alloy
    kokoro:
      module: kokoro
      class: KokoroTTS
      hardware: null
      import_check: kokoro
      install_extra: cascade_kokoro
      voice: am_adam
    elevenlabs:
      module: elevenlabs
      class: ElevenLabsTTS
      requires: [ELEVENLABS_API_KEY]
      voice_id: "..."
      model: eleven_flash_v2_5
```

### Environment Variables

```
OPENAI_API_KEY      # Required for OpenAI ASR/LLM/TTS
GEMINI_API_KEY      # Required for Gemini LLM
DEEPGRAM_API_KEY    # Required for Deepgram streaming ASR
ELEVENLABS_API_KEY  # Required for ElevenLabs TTS
```

### Config Loading (`config.py`)

Lazy singleton via `get_config()` (created on first call, not on import):

```python
from cascade.config import get_config
config = get_config()

# Key attributes:
config.asr_provider       # str — selected ASR provider name
config.llm_provider       # str — selected LLM provider name
config.tts_provider       # str — selected TTS provider name
config.asr_providers      # dict — all ASR provider definitions
config.llm_providers      # dict — all LLM provider definitions
config.tts_providers      # dict — all TTS provider definitions
config.llm_temperature    # float — LLM temperature (default 1.0)
config.tts_trim_silence   # bool — trim silence from TTS output
config.gliner_model       # str — GLiNER model for entity recognition
config.OPENAI_API_KEY     # str|None — from environment
config.DEEPGRAM_API_KEY   # str|None
config.GEMINI_API_KEY     # str|None
config.ELEVENLABS_API_KEY # str|None

# Helper methods:
config.get_asr_settings()    # Provider settings (excludes metadata)
config.is_asr_streaming()    # Whether selected ASR supports streaming
config.get_llm_settings()
config.get_tts_settings()
```

`set_config(cfg)` is available for test overrides.

---

## Audio Playback System

### Architecture

Pre-warmed persistent threads created at UI startup:

```
┌─────────────────────────┐
│ Playback Thread         │  ← Runs forever, blocks on queue
│ (sounddevice or robot)  │
└───────────▲─────────────┘
            │
       audio_queue
       (maxsize=100)
            │
┌───────────┴─────────────┐
│ GradioSpeechOutput      │  ← Enqueues TTS chunks (from speech_output.py)
│ .speak()                │
└───────────┬─────────────┘
            │
       wobbler_queue
       (maxsize=100)
            │
            ▼
┌─────────────────────────┐
│ Wobbler Thread          │  ← Runs forever, blocks on queue
│ (head animation)        │
└─────────────────────────┘
```

### Playback Backends

**sounddevice** (laptop speakers):
- `sd.OutputStream(samplerate=24000, channels=1, dtype=int16)`
- Pre-initialized stream at startup
- Direct chunk writes

**robot.media** (robot speakers):
- `robot.media.start_playing()` called once
- Convert int16 → float32, resample if needed
- `robot.media.push_audio_sample(audio_float)`

### Device Detection

The UI queries the default output device and checks for robot speaker keywords:
- "respeaker", "xvf3800", "reachy"
- Only uses robot.media if default matches a robot speaker

---

## Module Dependencies

```
main.py
  │
  └─> entry.py (run_cascade_mode)
      │
      ├─> handler.py (CascadeHandler)
      │   │
      │   ├─> provider_factory.py (init_asr/llm/tts_provider)
      │   │   ├─> config.py
      │   │   ├─> asr/ providers
      │   │   ├─> llm/ providers
      │   │   └─> tts/ providers
      │   ├─> pipeline.py (process_llm_response, execute_tool_calls)
      │   │   └─> speech_output.py (SpeechOutput.speak() for TTS)
      │   └─> transcript_analysis/ (TranscriptAnalysisManager)
      │       ├─> loader.py (reads profiles/<name>/reactions.yaml)
      │       ├─> keyword_analyzer.py
      │       └─> entity_analyzer.py (optional, requires gliner)
      │
      ├─[--gradio]─> ui/gradio_app.py (CascadeGradioUI)
      │   ├─> ui/audio_playback.py (AudioPlaybackSystem)
      │   ├─> ui/audio_recording.py (Recorders)
      │   ├─> speech_output.py (GradioSpeechOutput → AudioPlaybackSystem)
      │   └─> handler (reference)
      │
      ├─[default]─> console.py (CascadeLocalStream)
      │   ├─> speech_output.py (ConsoleSpeechOutput → robot.media)
      │   ├─> vad.py (SileroVAD)
      │   └─> handler (reference)
      │
      └─[--test-file]─> autotest_stream.py (CascadeTestStream)
          ├─> speech_output.py (ConsoleSpeechOutput → sounddevice)
          └─> handler (reference)
```

**Key relationships:**
- `entry.py` creates handler + one stream manager, wires `speech_output` into handler
- Handler owns conversation state, provider instances, and transcript analysis
- Pipeline calls `handler.speech_output.speak()` during tool execution — TTS happens inside the pipeline
- Each stream manager provides its own `SpeechOutput` implementation:
  - Gradio: `GradioSpeechOutput` (parallel sentence synthesis → pre-warmed playback threads)
  - Console: `ConsoleSpeechOutput` (rate-limited streaming → robot speaker)
  - Test: `ConsoleSpeechOutput` (rate-limited streaming → sounddevice callback)
- AudioPlaybackSystem handles pre-warmed playback threads (Gradio mode only)
- Recording classes encapsulate VAD-based continuous recording (Gradio mode only)
- Transcript analysis runs in parallel with the main ASR → LLM → TTS pipeline

---

## Entry Point & Integration

### Entry Function (`entry.py`)

`run_cascade_mode()` selects one of three stream managers based on CLI flags:

| Priority | Condition | Stream Manager | Source file | Handler lifecycle |
|----------|-----------|----------------|-------------|-------------------|
| 1 | `--test-file` | `CascadeTestStream` | `autotest_stream.py` | Synchronous (no `handler.start()`) |
| 2 | `--gradio` | `CascadeGradioUI.create_interface()` | `ui/gradio_app.py` | Background event loop (`handler.start()` / `handler.stop()`) |
| 3 | (default) | `CascadeLocalStream` | `console.py` | Synchronous (no `handler.start()`) |

All three share the same shutdown sequence: `stream_manager.close()` → stop services → `robot.media.close()` → `robot.client.disconnect()` → `os._exit(0)`.

### Command Line Usage

```bash
reachy-mini-conversation-app --gradio --head-tracker yolo
```

---

## Latency Tracking (`timing.py`)

### LatencyTracker

Profiles end-to-end latency from user action to first audio playback.

**Key Events (two flows):**

Button flow (Gradio push-to-talk):
- `user_stop_click` → `recording_ready` → `transcribing_start` → `asr_complete` → `llm_start` → `llm_complete` → `tts_start` → `tts_first_chunk_ready` → `audio_playback_started`

VAD flow (console mode, continuous mode, test file):
- `vad_speech_end` → `recording_captured` → `transcribing_start` → `asr_complete` → `llm_start` → `llm_complete` → `tts_start` → `tts_first_chunk_ready` → `audio_playback_started`

`print_summary()` auto-detects which flow was used and displays the appropriate stages.

**Usage:**
```python
from cascade.timing import tracker

tracker.reset("vad_speech_end")  # or "pipeline_start" for button flow
tracker.mark("event_name", {"metadata": "value"})
tracker.get_duration("start_event", "end_event")  # -> ms or None
tracker.print_summary()
```

---

## Transcript Analysis (Live Reactions)

The transcript analysis system triggers reactive robot behaviors (sounds, movements, emotions) in real time as the user speaks, **independently of the LLM pipeline**. Reactions fire based on keywords or named entities detected in the ASR transcript.

### Architecture

```
ASR transcript (partial or final)
  │
  └─> TranscriptAnalysisManager
      │
      ├─ KeywordAnalyzer.analyze(text)  ──→ {reaction_name: [matched_words]}
      │
      ├─ EntityAnalyzer.analyze(text)   ──→ [EntityMatch(text, label, confidence)]
      │
      ├─ Deduplication (per-turn)
      │
      └─ Dispatch → reaction.callback(deps, match, **params)
```

The handler calls the manager at three points:
- **`_on_transcript_partial(text)`** — on each streaming ASR partial (debounced at 400ms)
- **`_on_transcript_final(text)`** — when the final transcript is ready (fire-and-forget, parallel with LLM)
- **`_on_turn_complete()`** — resets all deduplication state for the next turn

### Reaction Configuration (`reactions.yaml`)

Each profile can define a `reactions.yaml` file. The loader (`loader.py`) reads it and imports callbacks from the profile's Python modules.

```yaml
# Simple keyword trigger — fires if any word matches
- name: music_excitement
  callback: excited_about_music
  trigger:
    words: [music, guitar, piano, drum, violin]

# Entity trigger with repeatable — fires once per unique entity
- name: food_reaction
  callback: react_to_food_entity
  trigger:
    entities: [food]
  repeatable: true

# Boolean AND trigger — all sub-groups must match
- name: groovy_dance
  callback: do_groovy_dance
  trigger:
    all:
      - words: [danc*]
      - words: [groov*]

# With extra params passed as kwargs to the callback
- name: turn_left
  callback: turn_to_direction
  trigger:
    all:
      - words: [turn*]
      - words: [left]
  params:
    direction: left
```

**Fields:**

| Field | Required | Description |
|-------|----------|-------------|
| `name` | yes | Unique reaction identifier |
| `callback` | yes | Python module name in the profile directory (module must contain a function with the same name) |
| `trigger` | yes | What activates the reaction (see Trigger Types below) |
| `params` | no | Extra kwargs passed to the callback |
| `repeatable` | no | If `true`, can fire multiple times per turn (default: `false`) |

### Trigger Types

#### `words` — Keyword matching

The word list supports three forms, mixable in a single list:

| Form | Example | Matching strategy |
|------|---------|-------------------|
| Plain word | `guitar` | **Substring** match on the full transcript text. Matches "guitar", "guitars", "guitarist". |
| Glob pattern | `music*` | **`fnmatch`** against individual whitespace-split tokens. Matches "music", "musical", "musician". Supports `*` and `?`. |
| Multi-word phrase | `"grand piano"` | **Substring** match (use YAML quotes). Matches any occurrence of "grand piano" in the text. |

Any one match in the list is enough to trigger the reaction (OR logic).

#### `entities` — Named Entity Recognition

Uses GLiNER (optional dependency) to detect entities by semantic label (e.g. `food`, `person`, `location`). The model is configurable in `cascade.yaml`:

```yaml
transcript_analysis:
  gliner_model: "urchade/gliner_small-v2.1"
```

Entity analysis runs in a thread executor since GLiNER inference is CPU-bound.

#### `all` — Boolean AND

A list of sub-triggers that must **all** match for the reaction to fire. Internally, each sub-trigger is registered as a synthetic keyword entry (`reaction_name__all_0`, `reaction_name__all_1`, etc.) and merged back after analysis.

### Deduplication

By default, each reaction fires **at most once per turn** (per conversation exchange). The manager tracks fired reactions in `triggered_reactions: set[str]` and skips duplicates.

**Repeatable reactions** (`repeatable: true`) bypass this gate. For entity-triggered repeatable reactions, deduplication is per unique entity text: "pizza" triggers once, "zucchini" triggers once, but "pizza" again is skipped. This is tracked via `_triggered_entity_keys: set[tuple[str, str]]`.

All deduplication state resets when `_on_turn_complete()` is called.

### Callback Signature

Every callback must be an async function with this signature:

```python
async def my_callback(
    deps: ToolDependencies,
    match: TriggerMatch,
    **kwargs,    # receives params from YAML
) -> None:
```

`TriggerMatch` contains what matched:
- `match.words: list[str]` — matched keywords (for word triggers)
- `match.entities: list[EntityMatch]` — matched entities, each with `.text`, `.label`, `.confidence`

Callbacks are imported from `profiles/<profile_name>/<callback_name>.py` — the module must export a function with the same name as the module.

### Data Classes (`base.py`)

```python
@dataclass
class ReactionConfig:
    name: str
    callback: Callable[..., Awaitable[None]]
    trigger: TriggerConfig
    params: dict[str, Any] = field(default_factory=dict)
    repeatable: bool = False

@dataclass
class TriggerConfig:
    words: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    all: list[TriggerConfig] = field(default_factory=list)

@dataclass
class TriggerMatch:
    words: list[str] = field(default_factory=list)
    entities: list[EntityMatch] = field(default_factory=list)

@dataclass
class EntityMatch:
    text: str          # e.g. "pizza"
    label: str         # e.g. "food"
    confidence: float  # 0.0 – 1.0
```

---

## Design Decisions

**Why TTS is driven by a `SpeechOutput` protocol, not hardcoded:**
- `speak` tool calls `speech_output.speak()` inside the pipeline — TTS happens as part of tool execution
- Each stream manager injects its own implementation:
  - Gradio: `GradioSpeechOutput` — parallel sentence synthesis, pre-warmed playback threads
  - Console/Test: `ConsoleSpeechOutput` — rate-limited streaming with playback callback
- Handler and pipeline stay decoupled from audio I/O details

**Why conversation history lives in Handler:**
- Handler needs full context for LLM generation
- Tool results must be added for multi-turn reasoning
- Camera tool adds images to conversation for analysis
- UI reads history for display only

---

## Troubleshooting & Known Issues

### MLX Thread Affinity (Apple Silicon)

**Note:** MLX-based providers (Parakeet) call MLX synchronously inside async methods, intentionally blocking the event loop for ~10-50ms per chunk. Do **not** wrap these calls in `asyncio.to_thread()` — MLX has thread affinity requirements and will produce empty transcriptions if inference runs in a different thread than model loading. The blocking is negligible at this scale. If a future model makes inference significantly slower (>100ms), consider a dedicated single-thread executor instead.

### OpenAI Realtime ASR - "Streaming" Misconception

**Problem:** Partial transcripts don't appear during speech - they all arrive at once after speech ends.

**Root Cause:** OpenAI's "streaming transcription" means text streams out quickly *after* audio is committed, NOT that you get real-time partials while speaking.

**How it actually works:**
1. Audio chunks sent → buffer on server
2. Server VAD detects silence → commits buffer
3. THEN transcription starts and deltas stream rapidly (~200ms for full text)

**Additional issue - Connection latency:**
- WebSocket connection takes ~800-1000ms to establish after speech starts
- Audio recorded during this time overflows and is lost
- Server VAD may not track speech properly due to this discontinuity

**Configuration options (`cascade.yaml`):**
```yaml
openai_realtime_asr:
  use_server_vad: true   # Real-time partials after each silence detection
  use_server_vad: false  # All transcription at end (manual commit)
```

**With `use_server_vad: true`:**
- Server VAD (500ms silence) and local Silero VAD (700ms) can coexist
- Server streams partials after each detected pause
- Local VAD controls when `end_stream()` is called

**Comparison with Deepgram:**
- Deepgram also streams partials after audio is processed, not during speech
- Both exhibit similar "batch of partials at end" behavior
- True real-time mid-speech partials would require periodic audio commits (causing fragmented transcripts)

**Potential fix (not implemented):** Pre-warm WebSocket connection before speech starts, keeping it in standby mode to eliminate connection latency.

---

## Test File Mode (`autotest_stream.py`)

Automated end-to-end testing of the cascade pipeline without human interaction. Reads text utterances from a file, synthesizes them to audio via TTS, and feeds the audio through the full pipeline (TTS→ASR→LLM→TTS→robot).

### Usage

```bash
# Uses cascade pipeline (default)
reachy-mini-conversation-app --test-file scripts/test_utterances.txt --no-camera
```

### Test File Format

Plain text, one utterance per line. `#` comments and blank lines are ignored:

```
# Greetings
Hello, what is your name?

# Movement commands
Can you look to the left?
```

### Architecture

`CascadeTestStream` is a stream manager (like `CascadeLocalStream` or `CascadeGradioUI`) that replaces mic/VAD input with synthetic audio.

```
Text file → Input TTS (af_heart voice) → PCM audio
                                            │
                    ┌───────────────────────┤
                    │                       │
                    ▼                       ▼
            Speaker playback        ASR (streaming or batch)
            (sounddevice callback)          │
                                            ▼
                                    LLM → Tool calls
                                            │
                                            ▼
                                    Output TTS (robot voice)
                                            │
                                            ▼
                                    Speaker playback
                                    (same sounddevice stream)
```

**Key components:**

- **Input TTS** — Separate `TTSProvider` instance using a distinct voice (`af_heart`) so user and robot are distinguishable
- **Callback-based playback** — `sd.OutputStream` in callback mode pulls from a `bytearray` buffer at a steady rate. Both user and robot audio share this buffer, avoiding the choppy playback that blocking `write()` calls cause in an async context
- **Two processing paths:**
  - **Streaming ASR** (`_process_streaming`): resamples PCM to 16kHz, feeds 32ms chunks at real-time pace via `process_audio_streaming_start/chunk/end`. This triggers partial transcripts and transcript analysis reactions during "speech", matching the live VAD flow
  - **Batch ASR** (`_process_manual`): plays user audio, then sends full WAV via `process_audio_manual`

### What This Tests

- Full ASR→LLM→TTS pipeline with real provider calls
- Tool calling (speak, movements, emotions)
- Transcript analysis / live reactions (streaming path only)
- Head wobbler animation synchronized with response audio
- Movement manager integration
- Cost tracking across turns
- Latency tracking (uses `vad_speech_end` reference for correct summary)
- Conversation history accumulation across turns

### What This Does NOT Test

Since `CascadeTestStream` is its own stream manager, it bypasses several components that the other modes use:

- **VAD speech detection** — Utterances are pre-defined text, not detected from audio. Silero VAD is never invoked
- **Microphone input / robot.media recording** — No `start_recording()` or `get_audio_sample()` calls. Audio comes from TTS, not hardware
- **Gradio UI** — No web interface, no push-to-talk buttons, no chatbot display, no `AudioPlaybackSystem`
- **Robot speaker output** — Response audio plays through computer speakers (`sounddevice`), not `robot.media.push_audio_sample()`
- **ContinuousVADRecorder** — This recording class is not used
- **Barge-in / interruption handling** — Utterances are sequential with fixed delays; there is no overlap between user speech and robot response
- **Audio resampling for robot hardware** — The `_play_loop` resampling path in `CascadeLocalStream` (TTS rate → robot output rate) is not exercised
- **Camera / vision pipeline** — Typically run with `--no-camera`; `describe_camera_image` tool calls would fail without it

### Latency Tracking

The tracker is reset **after** input TTS playback finishes (not when utterance generation starts), so the summary reflects only the pipeline latency. It uses `vad_speech_end` as the reference event so `print_summary()` recognizes the VAD flow and shows all stages:

```
1. Recording Capture
2. ASR Processing
3. LLM Generation
4. TTS time to first audio
   ↳ Audio system delay
TOTAL PERCEIVED LATENCY: Speech End → First Audio
```

---

## Future Extensions

- Streaming TTS integration
- Multi-language support with per-provider language codes
- Visual/audio cues for tool execution feedback
- Conversation reset without app restart
- Multi-modal prompts with vision in system prompt
