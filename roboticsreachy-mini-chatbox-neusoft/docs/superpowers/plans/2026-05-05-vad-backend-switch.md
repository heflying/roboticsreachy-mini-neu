# VAD Backend Switch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add ONNX Runtime as alternative VAD backend with config-driven switching and inference latency logging.

**Architecture:** Strategy pattern — `VADBackend` Protocol defines inference interface, `SileroVADPyTorch` and `SileroVADOnnx` implement it. `SileroVAD` facade delegates and injects timing logs.

**Tech Stack:** Silero VAD model, PyTorch (existing), ONNX Runtime (new), numpy

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/.../cascade/vad.py` | Modify | Protocol + SileroVADPyTorch + SileroVAD facade + timing |
| `src/.../cascade/vad_onnx.py` | Create | SileroVADOnnx backend with model download |
| `src/.../cascade/config.py` | Modify | Parse vad config section |
| `src/.../cascade/ui/audio_recording.py` | Modify | Read VAD params from config |
| `src/.../cascade/console.py` | Modify | Read VAD params from config |
| `cascade.yaml` | Modify | Add vad section (default: onnx) |
| `pyproject.toml` | Modify | Add `cascade_silero_vad_onnx` extra |
| `tests/cascade/test_vad_backends.py` | Create | Backend + facade + config tests |

---

### Task 1: VADBackend Protocol + SileroVADPyTorch + Facade Refactor

**Files:**
- Modify: `src/reachy_mini_conversation_app/cascade/vad.py`
- Create: `tests/cascade/test_vad_backends.py`

- [ ] **Step 1: Write failing test for Protocol conformance and facade timing**

```python
# tests/cascade/test_vad_backends.py
"""Tests for VAD backend switching and timing."""

import importlib
import logging
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest
from typing import Any

from reachy_mini_conversation_app.cascade.vad import (
    SILERO_SAMPLE_RATE,
    VADBackend,
    SileroVAD,
    SileroVADPyTorch,
    VADEvent,
    VADStateMachine,
)


def _silence_audio() -> npt.NDArray[Any]:
    return np.zeros(512, dtype=np.int16)


def _speech_like_audio() -> npt.NDArray[Any]:
    return np.random.randint(-5000, 5000, 512, dtype=np.int16)


class TestVADBackendProtocol:
    """Test VADBackend Protocol conformance."""

    def test_mock_backend_satisfies_protocol(self):
        """A class with get_speech_prob and reset_states satisfies VADBackend."""

        class FakeBackend:
            def get_speech_prob(self, audio: npt.NDArray[Any], sample_rate: int) -> float:
                return 0.5

            def reset_states(self) -> None:
                pass

        backend = FakeBackend()
        assert isinstance(backend, VADBackend)

    def test_missing_method_fails_protocol(self):
        """A class missing reset_states does not satisfy VADBackend."""

        class Incomplete:
            def get_speech_prob(self, audio: npt.NDArray[Any], sample_rate: int) -> float:
                return 0.5

        assert not isinstance(Incomplete(), VADBackend)


class TestSileroVADFacade:
    """Test SileroVAD facade with mocked backend."""

    def _make_facade(self, backend_name: str = "mock") -> SileroVAD:
        mock = MagicMock()
        mock.get_speech_prob.return_value = 0.8
        mock.reset_states.return_value = None
        vad = SileroVAD.__new__(SileroVAD)
        vad._backend = mock
        vad._backend_name = backend_name
        vad.threshold = 0.5
        vad.min_speech_duration_ms = 250
        vad.min_silence_duration_ms = 500
        vad._speech_frames = 0.0
        vad._silence_frames = 0.0
        vad._is_speaking = False
        vad._prob_log_count = 0
        return vad

    def test_delegates_get_speech_prob(self):
        vad = self._make_facade()
        prob = vad.get_speech_prob(_silence_audio())
        assert prob == 0.8
        vad._backend.get_speech_prob.assert_called_once()

    def test_delegates_reset(self):
        vad = self._make_facade()
        vad.reset()
        vad._backend.reset_states.assert_called_once()

    def test_timing_log_output(self, caplog: pytest.LogCaptureFixture):
        vad = self._make_facade("onnx")
        with caplog.at_level(logging.DEBUG, logger="reachy_mini_conversation_app.cascade.vad"):
            vad.get_speech_prob(_silence_audio())
        assert any("backend=onnx" in r.message and "inference=" in r.message for r in caplog.records)

    def test_process_chunk_no_speech(self):
        vad = self._make_facade()
        vad._backend.get_speech_prob.return_value = 0.1
        started, ended = vad.process_chunk(_silence_audio())
        assert not started
        assert not ended

    def test_process_chunk_speech_start(self):
        vad = self._make_facade()
        vad._backend.get_speech_prob.return_value = 0.9
        # Need enough frames to exceed min_speech_duration_ms (250ms)
        started = False
        for _ in range(10):
            started, _ = vad.process_chunk(_speech_like_audio())
            if started:
                break
        assert started

    def test_is_speech_delegates(self):
        vad = self._make_facade()
        vad._backend.get_speech_prob.return_value = 0.6
        assert vad.is_speech(_silence_audio()) is True


class TestSileroVADPyTorch:
    """Test PyTorch backend (requires torch installed)."""

    @pytest.mark.skipif(
        not importlib.util.find_spec("torch"),
        reason="torch not installed",
    )
    def test_get_speech_prob_returns_valid_float(self):
        backend = SileroVADPyTorch()
        prob = backend.get_speech_prob(_silence_audio(), SILERO_SAMPLE_RATE)
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    @pytest.mark.skipif(
        not importlib.util.find_spec("torch"),
        reason="torch not installed",
    )
    def test_reset_states_no_error(self):
        backend = SileroVADPyTorch()
        backend.get_speech_prob(_speech_like_audio(), SILERO_SAMPLE_RATE)
        backend.reset_states()  # should not raise

    @pytest.mark.skipif(
        not importlib.util.find_spec("torch"),
        reason="torch not installed",
    )
    def test_silence_has_low_probability(self):
        backend = SileroVADPyTorch()
        prob = backend.get_speech_prob(_silence_audio(), SILERO_SAMPLE_RATE)
        assert prob < 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/d/wsl/peiliao/T1-reference-apps/roboticsreachy-mini-chatbox && python -m pytest tests/cascade/test_vad_backends.py -v --co 2>&1 | head -30`
Expected: ImportError for `VADBackend`, `SileroVADPyTorch` — these don't exist yet.

- [ ] **Step 3: Refactor vad.py — add Protocol + extract SileroVADPyTorch + refactor SileroVAD facade**

Replace the entire content of `src/reachy_mini_conversation_app/cascade/vad.py` with:

```python
"""Voice Activity Detection using Silero VAD.

Supports multiple inference backends (PyTorch, ONNX Runtime) via VADBackend Protocol.
SileroVAD facade delegates inference and adds timing logs.
"""

import logging
from enum import Enum, auto
from time import perf_counter
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


logger = logging.getLogger(__name__)

SILERO_SAMPLE_RATE = 16000
VAD_CHUNK_SIZE = 512


@runtime_checkable
class VADBackend(Protocol):
    """Protocol for VAD inference backends."""

    def get_speech_prob(self, audio: npt.NDArray[Any], sample_rate: int) -> float: ...
    def reset_states(self) -> None: ...


class SileroVADPyTorch:
    """Silero VAD inference using PyTorch."""

    def __init__(self) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Silero VAD PyTorch backend requires torch. Install with: "
                "uv sync --extra cascade_silero_vad"
            ) from exc
        self._torch = torch
        logger.info("Loading Silero VAD model (PyTorch)...")
        self.model, _utils = self._torch.hub.load(
            "snakers4/silero-vad",
            "silero_vad",
            trust_repo=True,
        )
        self.model.eval()

    def get_speech_prob(self, audio: npt.NDArray[Any], sample_rate: int = SILERO_SAMPLE_RATE) -> float:
        if sample_rate != SILERO_SAMPLE_RATE:
            raise ValueError(f"Silero VAD requires {SILERO_SAMPLE_RATE}Hz audio, got {sample_rate}Hz")

        if audio.dtype == np.int16:
            audio_float = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.float32:
            audio_float = audio
        else:
            audio_float = audio.astype(np.float32)

        tensor = self._torch.from_numpy(audio_float)
        with self._torch.no_grad():
            return self.model(tensor, sample_rate).item()  # type: ignore[no-any-return]

    def reset_states(self) -> None:
        self.model.reset_states()


class SileroVAD:
    """Voice Activity Detection facade with pluggable backends.

    Delegates inference to a VADBackend implementation and adds
    timing logs for performance comparison.
    """

    def __init__(
        self,
        backend: str = "onnx",
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 500,
    ) -> None:
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self._backend_name = backend

        if backend == "onnx":
            from reachy_mini_conversation_app.cascade.vad_onnx import SileroVADOnnx

            self._backend: VADBackend = SileroVADOnnx()
        elif backend == "pytorch":
            self._backend = SileroVADPyTorch()
        else:
            raise ValueError(f"Unknown VAD backend: {backend!r}. Use 'onnx' or 'pytorch'.")

        self._speech_frames: float = 0
        self._silence_frames: float = 0
        self._is_speaking = False
        self._prob_log_count = 0
        logger.info(f"Silero VAD initialized (backend={backend}, threshold={threshold})")

    def get_speech_prob(self, audio: npt.NDArray[Any], sample_rate: int = SILERO_SAMPLE_RATE) -> float:
        t0 = perf_counter()
        prob = self._backend.get_speech_prob(audio, sample_rate)
        elapsed_ms = (perf_counter() - t0) * 1000
        self._prob_log_count += 1
        if self._prob_log_count % 50 == 0:
            logger.debug(f"VAD backend={self._backend_name} inference={elapsed_ms:.3f}ms prob={prob:.3f}")
        return prob

    def is_speech(self, audio: npt.NDArray[Any], sample_rate: int = SILERO_SAMPLE_RATE) -> bool:
        prob = self.get_speech_prob(audio, sample_rate)
        return prob >= self.threshold

    def process_chunk(self, audio: npt.NDArray[Any], sample_rate: int = SILERO_SAMPLE_RATE) -> tuple[bool, bool]:
        chunk_duration_ms = len(audio) / sample_rate * 1000
        prob = self.get_speech_prob(audio, sample_rate)
        is_speech = prob >= self.threshold

        speech_started = False
        speech_ended = False

        if is_speech:
            self._silence_frames = 0
            self._speech_frames += chunk_duration_ms
            if not self._is_speaking and self._speech_frames >= self.min_speech_duration_ms:
                self._is_speaking = True
                speech_started = True
                logger.debug(f"Speech started (accumulated {self._speech_frames:.0f}ms)")
        else:
            self._speech_frames = 0
            if self._is_speaking:
                self._silence_frames += chunk_duration_ms
                if self._silence_frames >= self.min_silence_duration_ms:
                    self._is_speaking = False
                    speech_ended = True
                    logger.debug(f"Speech ended (silence {self._silence_frames:.0f}ms)")

        return speech_started, speech_ended

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    def reset(self) -> None:
        self._speech_frames = 0.0
        self._silence_frames = 0.0
        self._is_speaking = False
        self._prob_log_count = 0
        self._backend.reset_states()
        logger.debug("VAD state reset")


class VADEvent(Enum):
    NOTHING = auto()
    SPEECH_STARTED = auto()
    SPEECH_ENDED = auto()


class VADState(Enum):
    LISTENING = "listening"
    RECORDING = "recording"
    PROCESSING = "processing"


class VADStateMachine:
    """Shared VAD state machine for pre-roll buffering and speech detection."""

    def __init__(
        self,
        vad: SileroVAD,
        chunk_size: int = VAD_CHUNK_SIZE,
        preroll_duration_s: float = 0.5,
    ) -> None:
        self._vad = vad
        self._state = VADState.LISTENING
        self._chunk_size = chunk_size
        self._max_preroll = int(preroll_duration_s * SILERO_SAMPLE_RATE / chunk_size)
        self._preroll_chunks: list[npt.NDArray[np.int16]] = []
        self.speech_chunks: list[npt.NDArray[np.int16]] = []

    @property
    def state(self) -> VADState:
        return self._state

    def process_chunk(self, audio_chunk: npt.NDArray[np.int16]) -> VADEvent:
        if self._state == VADState.PROCESSING:
            return VADEvent.NOTHING

        if self._state == VADState.LISTENING:
            speech_started, _ = self._vad.process_chunk(audio_chunk, SILERO_SAMPLE_RATE)
            self._preroll_chunks.append(audio_chunk)
            if len(self._preroll_chunks) > self._max_preroll:
                self._preroll_chunks = self._preroll_chunks[-self._max_preroll :]
            if speech_started:
                self._state = VADState.RECORDING
                self.speech_chunks = list(self._preroll_chunks)
                self._preroll_chunks = []
                logger.info("Speech detected, recording...")
                return VADEvent.SPEECH_STARTED
            return VADEvent.NOTHING

        self.speech_chunks.append(audio_chunk)
        _, speech_ended = self._vad.process_chunk(audio_chunk, SILERO_SAMPLE_RATE)
        if speech_ended:
            self._state = VADState.PROCESSING
            logger.info(f"Speech ended, {len(self.speech_chunks)} chunks")
            return VADEvent.SPEECH_ENDED
        return VADEvent.NOTHING

    def finish_processing(self) -> None:
        self.speech_chunks = []
        self._preroll_chunks = []
        self._vad.reset()
        self._state = VADState.LISTENING
```

- [ ] **Step 4: Run tests (mock-based tests pass, ONNX tests fail as expected)**

Run: `cd /mnt/d/wsl/peiliao/T1-reference-apps/roboticsreachy-mini-chatbox && python -m pytest tests/cascade/test_vad_backends.py::TestVADBackendProtocol tests/cascade/test_vad_backends.py::TestSileroVADFacade -v`
Expected: All Protocol and Facade tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/reachy_mini_conversation_app/cascade/vad.py tests/cascade/test_vad_backends.py
git commit -m "feat: add VADBackend Protocol + SileroVADPyTorch + facade with timing log"
```

---

### Task 2: SileroVADOnnx Backend + Tests

**Files:**
- Create: `src/reachy_mini_conversation_app/cascade/vad_onnx.py`
- Modify: `tests/cascade/test_vad_backends.py`

- [ ] **Step 1: Write failing test for ONNX backend**

Append to `tests/cascade/test_vad_backends.py`:

```python
def _onnx_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


class TestSileroVADOnnx:
    """Test ONNX backend (requires onnxruntime installed)."""

    @pytest.mark.skipif(not _onnx_available(), reason="onnxruntime not installed")
    def test_get_speech_prob_returns_valid_float(self):
        from reachy_mini_conversation_app.cascade.vad_onnx import SileroVADOnnx

        backend = SileroVADOnnx()
        prob = backend.get_speech_prob(_silence_audio(), SILERO_SAMPLE_RATE)
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    @pytest.mark.skipif(not _onnx_available(), reason="onnxruntime not installed")
    def test_silence_has_low_probability(self):
        from reachy_mini_conversation_app.cascade.vad_onnx import SileroVADOnnx

        backend = SileroVADOnnx()
        prob = backend.get_speech_prob(_silence_audio(), SILERO_SAMPLE_RATE)
        assert prob < 0.5

    @pytest.mark.skipif(not _onnx_available(), reason="onnxruntime not installed")
    def test_reset_states_no_error(self):
        from reachy_mini_conversation_app.cascade.vad_onnx import SileroVADOnnx

        backend = SileroVADOnnx()
        backend.get_speech_prob(_speech_like_audio(), SILERO_SAMPLE_RATE)
        backend.reset_states()

    @pytest.mark.skipif(not _onnx_available(), reason="onnxruntime not installed")
    def test_pytorch_and_onnx_agree_on_silence(self):
        """Both backends should report low probability for silence."""
        if not importlib.util.find_spec("torch"):
            pytest.skip("torch not installed")

        from reachy_mini_conversation_app.cascade.vad_onnx import SileroVADOnnx

        pytorch = SileroVADPyTorch()
        onnx = SileroVADOnnx()
        audio = _silence_audio()
        prob_pt = pytorch.get_speech_prob(audio, SILERO_SAMPLE_RATE)
        prob_onnx = onnx.get_speech_prob(audio, SILERO_SAMPLE_RATE)
        # Both should be below threshold (allow small numerical diff)
        assert prob_pt < 0.5
        assert prob_onnx < 0.5
        assert abs(prob_pt - prob_onnx) < 0.15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/cascade/test_vad_backends.py::TestSileroVADOnnx -v`
Expected: ImportError for `vad_onnx` module.

- [ ] **Step 3: Create vad_onnx.py**

```python
# src/reachy_mini_conversation_app/cascade/vad_onnx.py
"""Silero VAD ONNX Runtime backend."""

import logging
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


logger = logging.getLogger(__name__)

_MODEL_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
_MODEL_DIR = Path("models") / "VAD" / "silero"
SILERO_SAMPLE_RATE = 16000


def _ensure_model() -> Path:
    """Download silero_vad.onnx if not cached locally."""
    model_path = _MODEL_DIR / "silero_vad.onnx"
    if model_path.exists():
        return model_path
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading Silero VAD ONNX model to {model_path}...")
    urllib.request.urlretrieve(_MODEL_URL, str(model_path))
    logger.info("Silero VAD ONNX model downloaded")
    return model_path


class SileroVADOnnx:
    """Silero VAD inference using ONNX Runtime."""

    def __init__(self) -> None:
        try:
            import onnxruntime
        except ImportError as exc:
            raise RuntimeError(
                "Silero VAD ONNX backend requires onnxruntime. Install with: "
                "uv sync --extra cascade_silero_vad_onnx"
            ) from exc

        model_path = _ensure_model()
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = onnxruntime.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"], sess_options=opts,
        )
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)
        logger.info("Silero VAD ONNX backend initialized")

    def get_speech_prob(self, audio: npt.NDArray[Any], sample_rate: int = SILERO_SAMPLE_RATE) -> float:
        if sample_rate != SILERO_SAMPLE_RATE:
            raise ValueError(f"Silero VAD requires {SILERO_SAMPLE_RATE}Hz audio, got {sample_rate}Hz")

        if audio.dtype == np.int16:
            audio_float = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.float32:
            audio_float = audio
        else:
            audio_float = audio.astype(np.float32)

        audio_input = np.concatenate([self._context, audio_float.reshape(1, -1)], axis=1)
        ort_inputs = {
            "input": audio_input,
            "state": self._state,
            "sr": np.array(sample_rate, dtype=np.int64),
        }
        out, state = self._session.run(None, ort_inputs)
        self._state = state
        self._context = audio_input[:, -64:]
        return out.item()  # type: ignore[no-any-return]

    def reset_states(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/cascade/test_vad_backends.py -v`
Expected: All mock-based tests PASS. ONNX tests PASS if onnxruntime installed, otherwise SKIP.

- [ ] **Step 5: Commit**

```bash
git add src/reachy_mini_conversation_app/cascade/vad_onnx.py tests/cascade/test_vad_backends.py
git commit -m "feat: add SileroVADOnnx backend with auto model download"
```

---

### Task 3: Configuration + Consumer Updates + Dependencies

**Files:**
- Modify: `cascade.yaml`
- Modify: `src/reachy_mini_conversation_app/cascade/config.py`
- Modify: `src/reachy_mini_conversation_app/cascade/ui/audio_recording.py`
- Modify: `src/reachy_mini_conversation_app/cascade/console.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add vad section to cascade.yaml**

Insert before the `asr:` line:

```yaml
vad:
  backend: onnx
  threshold: 0.5
  min_speech_duration_ms: 250
  min_silence_duration_ms: 500
```

- [ ] **Step 2: Update config.py to parse vad config**

In `config.py`, after `self._cascade = _load_cascade_config()` (line 68), add:

```python
        # VAD config
        vad_cfg = self._cascade.get("vad", {})
        self.vad_backend: str = os.getenv("CASCADE_VAD_BACKEND") or vad_cfg.get("backend", "onnx")
        self.vad_threshold: float = float(vad_cfg.get("threshold", 0.5))
        self.vad_min_speech_duration_ms: int = int(vad_cfg.get("min_speech_duration_ms", 250))
        self.vad_min_silence_duration_ms: int = int(vad_cfg.get("min_silence_duration_ms", 500))
```

In `_validate()`, before the provider validation loop, add VAD backend validation:

```python
        # Validate VAD backend
        if self.vad_backend == "onnx":
            try:
                importlib.import_module("onnxruntime")
            except ImportError:
                raise RuntimeError(
                    "VAD backend 'onnx' requires 'onnxruntime'. "
                    "Install with: uv sync --extra cascade_silero_vad_onnx"
                )
        elif self.vad_backend == "pytorch":
            try:
                importlib.import_module("torch")
            except ImportError:
                raise RuntimeError(
                    "VAD backend 'pytorch' requires 'torch'. "
                    "Install with: uv sync --extra cascade_silero_vad"
                )
        else:
            raise RuntimeError(
                f"Unknown VAD backend '{self.vad_backend}'. Use 'onnx' or 'pytorch'."
            )
```

In `_log_config()`, add:

```python
        logger.info(f"Cascade: VAD backend={self.vad_backend}, ASR={self.asr_provider}, LLM={self.llm_provider}, TTS={self.tts_provider}")
```

- [ ] **Step 3: Update audio_recording.py to read VAD from config**

In `audio_recording.py`, change the `start()` method's VAD initialization block (around line 326-333):

Before:
```python
            from reachy_mini_conversation_app.cascade.vad import SileroVAD

            self._vad = SileroVAD(
                threshold=self.vad_threshold,
                min_speech_duration_ms=self.min_speech_duration_ms,
                min_silence_duration_ms=self.min_silence_duration_ms,
            )
```

After:
```python
            from reachy_mini_conversation_app.cascade.vad import SileroVAD
            from reachy_mini_conversation_app.cascade.config import get_config

            cfg = get_config()
            self._vad = SileroVAD(
                backend=cfg.vad_backend,
                threshold=cfg.vad_threshold,
                min_speech_duration_ms=cfg.vad_min_speech_duration_ms,
                min_silence_duration_ms=cfg.vad_min_silence_duration_ms,
            )
```

- [ ] **Step 4: Update console.py to read VAD from config**

In `console.py`, change the `__init__` VAD block (around line 46-50):

Before:
```python
        vad = SileroVAD(
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=700,
        )
```

After:
```python
        from reachy_mini_conversation_app.cascade.config import get_config

        cfg = get_config()
        vad = SileroVAD(
            backend=cfg.vad_backend,
            threshold=cfg.vad_threshold,
            min_speech_duration_ms=cfg.vad_min_speech_duration_ms,
            min_silence_duration_ms=700,  # console uses longer silence threshold
        )
```

- [ ] **Step 5: Update pyproject.toml**

Add new extra dependency after the `cascade_silero_vad` line:

```toml
cascade_silero_vad_onnx = ["onnxruntime>=1.17.0"]
```

Update `cascade_all` to include it:

```toml
cascade_all = [
  "reachy_mini_conversation_app[cascade]",
  "reachy_mini_conversation_app[cascade_silero_vad]",
  "reachy_mini_conversation_app[cascade_silero_vad_onnx]",
  ...existing entries...
]
```

- [ ] **Step 6: Run all cascade tests**

Run: `python -m pytest tests/cascade/ -v`
Expected: All existing tests PASS, new VAD backend tests PASS or SKIP.

- [ ] **Step 7: Commit**

```bash
git add cascade.yaml src/reachy_mini_conversation_app/cascade/config.py src/reachy_mini_conversation_app/cascade/ui/audio_recording.py src/reachy_mini_conversation_app/cascade/console.py pyproject.toml
git commit -m "feat: add VAD config switching (onnx/pytorch) with cascade.yaml support"
```
