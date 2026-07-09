# Zipformer Sherpa-ONNX 本地流式 ASR Provider 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增基于 sherpa-onnx 的 Zipformer 本地流式中文 ASR provider，纯 CPU 运行，RTF 0.15，~160MB。

**Architecture:** 直接继承 StreamingASRProvider，4 个 async 方法映射到 sherpa-onnx 的 OnlineRecognizer/OnlineStream 同步 API（通过 run_in_executor 包装）。即时加载模型，首次运行自动从 HuggingFace 下载。

**Tech Stack:** sherpa-onnx (ONNX Runtime), huggingface_hub, numpy

**Design Spec:** `docs/superpowers/specs/2026-05-06-zipformer-sherpa-asr-design.md`

---

## File Structure

| 文件 | 职责 |
|---|---|
| `src/reachy_mini_conversation_app/cascade/asr/zipformer_sherpa.py` | ASR provider 实现 |
| `tests/cascade/test_zipformer_sherpa.py` | 单元测试 |
| `src/reachy_mini_conversation_app/cascade/timing.py` | 新增本地 ASR 事件映射和阈值 |
| `cascade.yaml` | 新增 provider 配置项 |
| `pyproject.toml` | 新增 cascade_zipformer extra |
| `scripts/download_zipformer_zh.py` | 可选离线下载脚本 |
| `docs/级联架构性能指标设计.md` | 新增本地 ASR B 指标章节 |

---

### Task 1: Mock 框架与测试骨架

**Files:**
- Create: `tests/cascade/test_zipformer_sherpa.py`

- [ ] **Step 1: 创建测试文件，包含 mock 框架和基础测试**

```python
"""Tests for Zipformer sherpa-ONNX ASR provider."""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

from reachy_mini_conversation_app.cascade.asr.audio_utils import pcm_to_wav


# ---------------------------------------------------------------------------
# Mock sherpa_onnx (C++ extension, 不可在测试环境安装)
# ---------------------------------------------------------------------------

class MockOnlineStream:
    """Mock sherpa_onnx.OnlineStream."""

    def __init__(self) -> None:
        self._audio_chunks: list[list[float]] = []
        self._finished = False

    def accept_waveform(self, sample_rate: int, samples: list[float]) -> None:
        self._audio_chunks.append(samples)

    def input_finished(self) -> None:
        self._finished = True


class MockOnlineRecognizer:
    """Mock sherpa_onnx.OnlineRecognizer."""

    def __init__(self, *, partial: str = "", final: str = "你好世界") -> None:
        self._partial = partial
        self._final = final
        self._total_samples = 0

    @classmethod
    def from_transducer(cls, **kwargs: Any) -> "MockOnlineRecognizer":
        return cls()

    def create_stream(self) -> MockOnlineStream:
        return MockOnlineStream()

    def is_ready(self, stream: MockOnlineStream) -> bool:
        return True

    def decode_stream(self, stream: MockOnlineStream) -> None:
        for chunk in stream._audio_chunks:
            self._total_samples += len(chunk)

    def get_result(self, stream: MockOnlineStream) -> str:
        if stream._finished:
            return self._final
        if self._total_samples > 0:
            return self._partial
        return ""

    def reset(self, stream: MockOnlineStream) -> None:
        stream._audio_chunks.clear()
        stream._finished = False
        self._total_samples = 0


def _install_sherpa_mock() -> None:
    """Install mock sherpa_onnx module if not already available."""
    if "sherpa_onnx" in sys.modules:
        return
    mock_module = types.ModuleType("sherpa_onnx")
    mock_module.OnlineRecognizer = MockOnlineRecognizer
    mock_module.OnlineStream = MockOnlineStream
    sys.modules["sherpa_onnx"] = mock_module


_install_sherpa_mock()


# ---------------------------------------------------------------------------
# Helper: create WAV audio
# ---------------------------------------------------------------------------

def _silence_wav(duration_ms: int = 100, sample_rate: int = 16000) -> bytes:
    """Generate silent WAV audio."""
    num_samples = int(sample_rate * duration_ms / 1000)
    pcm = b"\x00\x00" * num_samples
    return pcm_to_wav(pcm, sample_rate)
```

- [ ] **Step 2: 运行测试确认 mock 可导入**

Run: `python -c "from tests.cascade.test_zipformer_sherpa import MockOnlineRecognizer; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add tests/cascade/test_zipformer_sherpa.py
git commit -m "test: add mock framework for Zipformer sherpa ASR tests"
```

---

### Task 2: Provider 构造函数与模型下载 — TDD

**Files:**
- Create: `src/reachy_mini_conversation_app/cascade/asr/zipformer_sherpa.py`
- Modify: `tests/cascade/test_zipformer_sherpa.py`

- [ ] **Step 1: 写失败的测试 — 构造函数加载模型**

在 `test_zipformer_sherpa.py` 末尾追加：

```python
# ---------------------------------------------------------------------------
# Tests: 构造函数与模型加载
# ---------------------------------------------------------------------------

def test_init_loads_model_and_creates_recognizer(tmp_path: Path):
    """__init__ should download model files and create OnlineRecognizer."""
    model_dir = tmp_path / "zipformer-zh"

    with patch("reachy_mini_conversation_app.cascade.asr.zipformer_sherpa.ZipformerSherpaASR._ensure_model") as mock_ensure:
        from reachy_mini_conversation_app.cascade.asr.zipformer_sherpa import ZipformerSherpaASR
        provider = ZipformerSherpaASR(
            model_id="test/repo",
            model_dir=str(model_dir),
            sample_rate=16000,
        )
        mock_ensure.assert_called_once()
        assert provider._sample_rate == 16000
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /mnt/d/wsl/peiliao/T1-reference-apps/roboticsreachy-mini-chatbox && python -m pytest tests/cascade/test_zipformer_sherpa.py::test_init_loads_model_and_creates_recognizer -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: 写最小实现 — ZipformerSherpaASR 构造函数**

创建 `src/reachy_mini_conversation_app/cascade/asr/zipformer_sherpa.py`：

```python
"""Zipformer ASR provider via sherpa-onnx (local, CPU streaming)."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from .audio_utils import wav_to_float32
from .base_streaming import StreamingASRProvider

logger = logging.getLogger(__name__)


class ZipformerSherpaASR(StreamingASRProvider):
    """Local streaming ASR using sherpa-onnx Zipformer transducer model.

    Pure CPU inference, ~160MB INT8 model, RTF 0.15.
    Model is loaded eagerly in __init__ for best first-turn performance.
    """

    # Expected model files
    _MODEL_FILES = ("encoder.int8.onnx", "decoder.onnx", "joiner.int8.onnx", "tokens.txt")

    def __init__(
        self,
        model_id: str = "csukuangfj/sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30",
        model_dir: str = "models/zipformer-zh",
        num_threads: int = 1,
        sample_rate: int = 16000,
        decoding_method: str = "greedy_search",
        enable_endpoint: bool = True,
        rule1_min_trailing_silence: float = 2.4,
        rule2_min_trailing_silence: float = 1.2,
    ) -> None:
        self._model_id = model_id
        self._model_dir = Path(model_dir)
        self._num_threads = num_threads
        self._sample_rate = sample_rate
        self._decoding_method = decoding_method
        self._enable_endpoint = enable_endpoint
        self._rule1_min_trailing_silence = rule1_min_trailing_silence
        self._rule2_min_trailing_silence = rule2_min_trailing_silence

        # Recognizer (singleton, shared across streams)
        self._recognizer: Any = None
        self._model_loaded = False

        # Streaming state (per-stream)
        self._stream: Any = None
        self._partial_text: str = ""

        # Eager load
        self._ensure_model()

    def _ensure_model(self) -> None:
        """Download model if missing, then create OnlineRecognizer."""
        if self._model_loaded:
            return

        self._download_if_missing()
        self._create_recognizer()
        self._model_loaded = True

    def _download_if_missing(self) -> None:
        """Download model from HuggingFace if local files are missing."""
        if self._all_model_files_present():
            logger.info(f"Zipformer model files found in {self._model_dir}")
            return

        logger.info(f"Downloading Zipformer model from {self._model_id}...")
        from huggingface_hub import snapshot_download

        t0 = time.perf_counter()
        snapshot_download(repo_id=self._model_id, local_dir=str(self._model_dir))
        elapsed = time.perf_counter() - t0
        logger.info(f"Model downloaded in {elapsed:.1f}s")

    def _all_model_files_present(self) -> bool:
        """Check if all required model files exist and are non-empty."""
        if not self._model_dir.exists():
            return False
        for fname in self._MODEL_FILES:
            fpath = self._model_dir / fname
            if not fpath.exists() or fpath.stat().st_size == 0:
                return False
        return True

    def _create_recognizer(self) -> None:
        """Create sherpa_onnx OnlineRecognizer from local model files."""
        import sherpa_onnx

        encoder = str(self._model_dir / "encoder.int8.onnx")
        decoder = str(self._model_dir / "decoder.onnx")
        joiner = str(self._model_dir / "joiner.int8.onnx")
        tokens = str(self._model_dir / "tokens.txt")

        t0 = time.perf_counter()
        self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            num_threads=self._num_threads,
            sample_rate=self._sample_rate,
            feature_dim=80,
            enable_endpoint_detection=self._enable_endpoint,
            rule1_min_trailing_silence=self._rule1_min_trailing_silence,
            rule2_min_trailing_silence=self._rule2_min_trailing_silence,
            decoding_method=self._decoding_method,
            provider="cpu",
        )
        elapsed = time.perf_counter() - t0
        logger.info(f"Zipformer recognizer created in {elapsed:.2f}s")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /mnt/d/wsl/peiliao/T1-reference-apps/roboticsreachy-mini-chatbox && python -m pytest tests/cascade/test_zipformer_sherpa.py::test_init_loads_model_and_creates_recognizer -v`
Expected: PASS

- [ ] **Step 5: 写失败的测试 — 模型下载逻辑**

在 `test_zipformer_sherpa.py` 末尾追加：

```python
def test_download_triggered_when_files_missing(tmp_path: Path):
    """_download_if_missing should call snapshot_download when files are missing."""
    model_dir = tmp_path / "missing-model"

    with patch("reachy_mini_conversation_app.cascade.asr.zipformer_sherpa.ZipformerSherpaASR._create_recognizer"):
        with patch("reachy_mini_conversation_app.cascade.asr.zipformer_sherpa.snapshot_download") as mock_dl:
            from reachy_mini_conversation_app.cascade.asr.zipformer_sherpa import ZipformerSherpaASR
            provider = ZipformerSherpaASR.__new__(ZipformerSherpaASR)
            provider._model_id = "test/repo"
            provider._model_dir = model_dir
            provider._model_loaded = False
            provider._recognizer = None
            provider._num_threads = 1
            provider._sample_rate = 16000
            provider._decoding_method = "greedy_search"
            provider._enable_endpoint = True
            provider._rule1_min_trailing_silence = 2.4
            provider._rule2_min_trailing_silence = 1.2
            provider._stream = None
            provider._partial_text = ""

            # Create fake model files so _create_recognizer doesn't fail
            model_dir.mkdir(parents=True, exist_ok=True)
            for fname in ZipformerSherpaASR._MODEL_FILES:
                (model_dir / fname).write_bytes(b"\x00" * 100)

            provider._ensure_model()
            mock_dl.assert_called_once_with(repo_id="test/repo", local_dir=str(model_dir))
            assert provider._model_loaded is True
```

注意：需要在 `zipformer_sherpa.py` 中延迟导入 `huggingface_hub`（在 `_download_if_missing` 内部导入），这样测试可以 mock `snapshot_download`。在文件顶部添加导入补丁：

在 `zipformer_sherpa.py` 的 `_download_if_missing` 中已经有延迟导入。测试需要 mock 的是模块级的 `snapshot_download`，需要调整导入方式。在 `_download_if_missing` 中改为：

```python
def _download_if_missing(self) -> None:
    if self._all_model_files_present():
        logger.info(f"Zipformer model files found in {self._model_dir}")
        return

    logger.info(f"Downloading Zipformer model from {self._model_id}...")
    from huggingface_hub import snapshot_download
    ...
```

测试中 mock 的路径应该是 `reachy_mini_conversation_app.cascade.asr.zipformer_sherpa.snapshot_download`。但因为这是在方法内部延迟导入，mock 路径应该是 `huggingface_hub.snapshot_download`。调整测试：

```python
def test_download_triggered_when_files_missing(tmp_path: Path):
    with patch("huggingface_hub.snapshot_download") as mock_dl:
        ...
```

- [ ] **Step 6: 运行测试确认通过**

Run: `cd /mnt/d/wsl/peiliao/T1-reference-apps/roboticsreachy-mini-chatbox && python -m pytest tests/cascade/test_zipformer_sherpa.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/reachy_mini_conversation_app/cascade/asr/zipformer_sherpa.py tests/cascade/test_zipformer_sherpa.py
git commit -m "feat: ZipformerSherpaASR constructor with model download logic"
```

---

### Task 3: 流式生命周期 — TDD

**Files:**
- Modify: `src/reachy_mini_conversation_app/cascade/asr/zipformer_sherpa.py`
- Modify: `tests/cascade/test_zipformer_sherpa.py`

- [ ] **Step 1: 写失败的测试 — start_stream / send_audio_chunk / get_partial_transcript / end_stream**

在 `test_zipformer_sherpa.py` 末尾追加：

```python
# ---------------------------------------------------------------------------
# Tests: Streaming lifecycle
# ---------------------------------------------------------------------------

def _make_provider(tmp_path: Path, *, partial: str = "你好", final: str = "你好世界") -> Any:
    """Create a ZipformerSherpaASR with mock recognizer (no model download)."""
    from reachy_mini_conversation_app.cascade.asr.zipformer_sherpa import ZipformerSherpaASR

    model_dir = tmp_path / "zipformer-zh"
    model_dir.mkdir(parents=True, exist_ok=True)
    for fname in ZipformerSherpaASR._MODEL_FILES:
        (model_dir / fname).write_bytes(b"\x00" * 100)

    with patch("huggingface_hub.snapshot_download"):
        with patch.object(ZipformerSherpaASR, "_create_recognizer") as mock_create:
            mock_recognizer = MockOnlineRecognizer(partial=partial, final=final)
            mock_create.side_effect = lambda: setattr(
                _make_provider, "_recognizer", mock_recognizer
            )
            provider = ZipformerSherpaASR(
                model_dir=str(model_dir),
                sample_rate=16000,
            )
            provider._recognizer = mock_recognizer
            return provider


def test_streaming_lifecycle(tmp_path: Path):
    """Full streaming lifecycle: start → send chunks → get partials → end."""
    async def run():
        provider = _make_provider(tmp_path, partial="你好", final="你好世界")

        # start_stream
        await provider.start_stream()
        assert provider._stream is not None

        # send_audio_chunk + get_partial_transcript
        wav = _silence_wav(100, 16000)
        await provider.send_audio_chunk(wav)
        partial = await provider.get_partial_transcript()
        assert partial == "你好"

        # end_stream
        result = await provider.end_stream()
        assert result == "你好世界"

    asyncio.run(run())


def test_send_empty_chunk_is_noop(tmp_path: Path):
    """Empty audio chunk should be silently skipped."""
    async def run():
        provider = _make_provider(tmp_path)

        await provider.start_stream()
        await provider.send_audio_chunk(b"")  # empty
        partial = await provider.get_partial_transcript()
        assert partial is None

    asyncio.run(run())


def test_end_stream_falls_back_to_partial(tmp_path: Path):
    """If final decode produces nothing, return the last partial."""
    async def run():
        # Recognizer that never returns final text
        provider = _make_provider(tmp_path, partial="中间结果", final="")

        await provider.start_stream()
        wav = _silence_wav(100, 16000)
        await provider.send_audio_chunk(wav)
        await provider.get_partial_transcript()

        result = await provider.end_stream()
        assert result == "中间结果"

    asyncio.run(run())


def test_multiple_start_stream_resets_state(tmp_path: Path):
    """Starting a new stream should reset previous state."""
    async def run():
        provider = _make_provider(tmp_path, partial="你好", final="你好")

        # First stream
        await provider.start_stream()
        wav = _silence_wav(100, 16000)
        await provider.send_audio_chunk(wav)
        await provider.end_stream()

        # Second stream (should reset)
        await provider.start_stream()
        assert provider._partial_text == ""
        assert provider._stream is not None

    asyncio.run(run())
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /mnt/d/wsl/peiliao/T1-reference-apps/roboticsreachy-mini-chatbox && python -m pytest tests/cascade/test_zipformer_sherpa.py::test_streaming_lifecycle -v`
Expected: FAIL (AttributeError: no start_stream)

- [ ] **Step 3: 实现流式方法**

在 `zipformer_sherpa.py` 的 `ZipformerSherpaASR` 类中追加：

```python
    async def start_stream(self) -> None:
        """Create a new OnlineStream for this session."""
        from reachy_mini_conversation_app.cascade.timing import tracker

        self._stream = self._recognizer.create_stream()
        self._partial_text = ""

        tracker.mark("asr_local_ready")
        tracker.mark("asr_local_stream_start")
        logger.debug("Zipformer streaming session started")

    async def send_audio_chunk(self, audio_chunk: bytes) -> None:
        """Feed audio chunk to the recognizer."""
        if not audio_chunk:
            return

        audio = wav_to_float32(audio_chunk, self._sample_rate)
        if len(audio) == 0:
            return

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._sync_feed, audio)

    def _sync_feed(self, audio: npt.NDArray[np.float32]) -> None:
        """Synchronous: feed audio and decode one step."""
        self._stream.accept_waveform(self._sample_rate, audio.tolist())
        if self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)

    async def get_partial_transcript(self) -> Optional[str]:
        """Return current partial transcript, or None."""
        from reachy_mini_conversation_app.cascade.timing import tracker

        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(
            None, self._recognizer.get_result, self._stream
        )
        text = text.strip() if text else ""
        if text:
            self._partial_text = text
            tracker.mark("asr_local_chunk_decode")
            return self._partial_text
        return None if not self._partial_text else self._partial_text

    async def end_stream(self) -> str:
        """Signal end of audio and return final transcript."""
        from reachy_mini_conversation_app.cascade.timing import tracker

        tracker.mark("asr_local_final_decode")

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._stream.input_finished)

        if self._recognizer.is_ready(self._stream):
            await loop.run_in_executor(None, self._recognizer.decode_stream, self._stream)

        text = await loop.run_in_executor(
            None, self._recognizer.get_result, self._stream
        )
        result = text.strip() if text else ""
        if not result:
            result = self._partial_text

        tracker.mark("asr_result_delivered", {"transcript_len": len(result)})
        logger.info(f"Zipformer final transcript: '{result}'")

        self._stream = None
        return result
```

需要在文件顶部添加 `import asyncio`（已隐含在 `from __future__ import annotations` 中但需要显式导入）：

在 `zipformer_sherpa.py` 的 import 区域添加：

```python
import asyncio
```

- [ ] **Step 4: 运行全部测试确认通过**

Run: `cd /mnt/d/wsl/peiliao/T1-reference-apps/roboticsreachy-mini-chatbox && python -m pytest tests/cascade/test_zipformer_sherpa.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/reachy_mini_conversation_app/cascade/asr/zipformer_sherpa.py tests/cascade/test_zipformer_sherpa.py
git commit -m "feat: ZipformerSherpaASR streaming lifecycle methods"
```

---

### Task 4: Timing 系统适配 — TDD

**Files:**
- Modify: `src/reachy_mini_conversation_app/cascade/timing.py`
- Modify: `tests/cascade/test_timing_metrics.py`

- [ ] **Step 1: 写失败的测试 — 本地 ASR 指标计算**

在 `tests/cascade/test_timing_metrics.py` 末尾追加：

```python
class TestLocalASRMetrics:
    """Tests for local ASR (sherpa-onnx Zipformer) timing metrics."""

    def test_b1_local_asr_shows_reuse(self):
        """Local ASR B1 should show 0ms reuse when asr_local_ready event exists."""
        tracker = LatencyTracker()
        tracker.reset("test")
        tracker.events = [
            {"name": "vad_speech_end", "canonical": "speech_end", "timestamp": tracker.start_time + 0.1, "elapsed_ms": 100, "metadata": {}},
            {"name": "asr_local_ready", "canonical": "asr_reuse", "timestamp": tracker.start_time + 0.1, "elapsed_ms": 100, "metadata": {}},
            {"name": "asr_local_stream_start", "canonical": "asr_b2_end", "timestamp": tracker.start_time + 0.1001, "elapsed_ms": 100.1, "metadata": {}},
        ]
        results = tracker.calculate_l2_asr_metrics()
        b1 = [m for m in results if m.code == "B1"]
        assert len(b1) == 1
        assert b1[0].is_reuse is True
        assert b1[0].value_ms == 0.0

    def test_b4_local_asr_uses_local_threshold(self):
        """Local ASR B4 should use asr_local_final_decode → asr_result_delivered."""
        tracker = LatencyTracker()
        tracker.reset("test")
        t0 = tracker.start_time
        tracker.events = [
            {"name": "asr_local_final_decode", "canonical": "asr_b4_start", "timestamp": t0 + 0.1, "elapsed_ms": 100, "metadata": {}},
            {"name": "asr_result_delivered", "canonical": "asr_b5_end", "timestamp": t0 + 0.135, "elapsed_ms": 135, "metadata": {"transcript_len": 4}},
        ]
        results = tracker.calculate_l2_asr_metrics()
        b4 = [m for m in results if m.code == "B4"]
        assert len(b4) == 1
        assert b4[0].value_ms == pytest.approx(35.0, abs=1.0)

    def test_b4_cloud_asr_unchanged(self):
        """Cloud ASR B4 path should remain unchanged."""
        tracker = LatencyTracker()
        tracker.reset("test")
        t0 = tracker.start_time
        tracker.events = [
            {"name": "asr_commit_sent", "canonical": "asr_b4_start", "timestamp": t0 + 0.1, "elapsed_ms": 100, "metadata": {}},
            {"name": "asr_result_delivered", "canonical": "asr_b5_end", "timestamp": t0 + 0.4, "elapsed_ms": 400, "metadata": {}},
        ]
        results = tracker.calculate_l2_asr_metrics()
        b4 = [m for m in results if m.code == "B4"]
        assert len(b4) == 1
        assert b4[0].value_ms == pytest.approx(300.0, abs=1.0)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /mnt/d/wsl/peiliao/T1-reference-apps/roboticsreachy-mini-chatbox && python -m pytest tests/cascade/test_timing_metrics.py::TestLocalASRMetrics -v`
Expected: FAIL (asr_local_ready not in EVENT_ALIASES, B4 local path not implemented)

- [ ] **Step 3: 修改 timing.py — 新增 EVENT_ALIASES 和阈值**

在 `timing.py` 的 `EVENT_ALIASES` 字典中，在 ASR events 区域追加：

```python
        # ASR events - 本地 ASR (sherpa-onnx Zipformer)
        "asr_local_ready": "asr_reuse",
        "asr_local_stream_start": "asr_b2_end",
        "asr_local_chunk_decode": "asr_b3_end",
        "asr_local_final_decode": "asr_b4_start",
```

在 `ASR_THRESHOLDS` 后面新增本地 ASR 阈值：

```python
# L2 ASR 阈值 (本地推理模式, 如 Zipformer sherpa-onnx)
ASR_LOCAL_THRESHOLDS = {
    "B4_asr_local_process": ThresholdConfig(30, 50, 100, 100, core_hotspot=True),
}
```

- [ ] **Step 4: 修改 timing.py — calculate_l2_asr_metrics 增加本地 ASR 分支**

在 `LatencyTracker` 类中新增检测方法：

```python
    def _is_local_asr(self) -> bool:
        """检测是否为本地 ASR 模式 (如 Zipformer sherpa-onnx)。"""
        return self.has_event("asr_local_ready")
```

修改 `calculate_l2_asr_metrics` 方法。将 B1 部分改为同时支持 reuse 和本地 ASR：

在现有的 B1 reuse 分支（`has_event("asr_ws_reused")`）后追加 `elif`：

```python
        elif self.has_event("asr_local_ready"):
            results.append(MetricResult(
                name="ASR连接建立",
                code="B1",
                value_ms=0.0,
                rating=RatingLevel.EXCELLENT,
                is_hotspot=False,
                is_core_hotspot=False,
                is_reuse=True,
            ))
```

B2 部分同理，在 reuse 后追加：

```python
        elif self.has_event("asr_local_stream_start"):
            results.append(MetricResult(
                name="ASR初始化",
                code="B2",
                value_ms=0.0,
                rating=RatingLevel.EXCELLENT,
                is_hotspot=False,
                is_core_hotspot=False,
                is_reuse=True,
            ))
```

B4 部分，在云端路径后追加本地 ASR 分支：

```python
        # B4 本地 ASR 处理 (本地推理)
        if self.has_event("asr_local_final_decode") and self.has_event("asr_result_delivered"):
            duration = self.get_duration("asr_local_final_decode", "asr_result_delivered")
            if duration is not None:
                rating, is_hotspot, is_core = MetricsCalculator.rate(
                    duration, ASR_LOCAL_THRESHOLDS["B4_asr_local_process"]
                )
                results.append(MetricResult(
                    name="ASR本地处理",
                    code="B4",
                    value_ms=duration,
                    rating=rating,
                    is_hotspot=is_hotspot,
                    is_core_hotspot=is_core,
                ))
```

- [ ] **Step 5: 修改 timing.py — validate_trace_formula 适配本地 ASR**

在 `validate_trace_formula` 方法中，event_chain 的 B4_start 需要适配本地 ASR：

找到 `("B4_start", "asr_commit_sent"),` 这行，将其改为条件判断：

```python
            ("B4_start", "asr_local_final_decode" if self._is_local_asr() else "asr_commit_sent"),
```

- [ ] **Step 6: 修改 timing.py — _get_hotspot_description 适配本地 ASR**

在 `_get_hotspot_description` 中更新 B4 描述：

```python
            "B4": "ASR本地处理延迟" if self._is_local_asr() else "ASR云端处理延迟",
```

- [ ] **Step 7: 运行测试确认通过**

Run: `cd /mnt/d/wsl/peiliao/T1-reference-apps/roboticsreachy-mini-chatbox && python -m pytest tests/cascade/test_timing_metrics.py::TestLocalASRMetrics -v`
Expected: ALL PASS

- [ ] **Step 8: 运行全部 timing 测试确认无回归**

Run: `cd /mnt/d/wsl/peiliao/T1-reference-apps/roboticsreachy-mini-chatbox && python -m pytest tests/cascade/test_timing_metrics.py -v`
Expected: ALL PASS

- [ ] **Step 9: Commit**

```bash
git add src/reachy_mini_conversation_app/cascade/timing.py tests/cascade/test_timing_metrics.py
git commit -m "feat: timing system adaptation for local ASR (Zipformer)"
```

---

### Task 5: 配置文件 — cascade.yaml + pyproject.toml

**Files:**
- Modify: `cascade.yaml`
- Modify: `pyproject.toml`

- [ ] **Step 1: 在 cascade.yaml 的 LOCAL PROVIDERS 区域新增配置**

在 `cascade.yaml` 的 `asr.providers` 中，在 `# ========== LOCAL PROVIDERS` 注释后（parakeet_mlx_progressive 之前或之后）插入：

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
      # Settings
      model_id: csukuangfj/sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30
      model_dir: models/zipformer-zh
      num_threads: 1
      sample_rate: 16000
      decoding_method: greedy_search
      enable_endpoint: true
      rule1_min_trailing_silence: 2.4
      rule2_min_trailing_silence: 1.2
```

- [ ] **Step 2: 在 pyproject.toml 新增 cascade_zipformer extra**

在 `[project.optional-dependencies]` 区域的 `cascade_qwen` 行后追加：

```toml
cascade_zipformer = ["sherpa-onnx>=1.10.0"]
```

在 `cascade_all` 列表中追加：

```toml
  "reachy_mini_conversation_app[cascade_zipformer]",
```

- [ ] **Step 3: Commit**

```bash
git add cascade.yaml pyproject.toml
git commit -m "feat: add Zipformer sherpa ASR config and pyproject extra"
```

---

### Task 6: 可选下载脚本

**Files:**
- Create: `scripts/download_zipformer_zh.py`

- [ ] **Step 1: 创建下载脚本**

```python
"""Download Zipformer Chinese streaming ASR model for offline use.

Downloads sherpa-onnx-streaming-zipformer-zh-int8 from HuggingFace
to models/zipformer-zh/ (~160MB).
"""

from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "csukuangfj/sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30"
MODEL_DIR = Path("models/zipformer-zh")


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading from {REPO_ID} -> {MODEL_DIR}/\n")
    snapshot_download(repo_id=REPO_ID, local_dir=str(MODEL_DIR))

    print(f"\nFiles in {MODEL_DIR}/:")
    for p in sorted(MODEL_DIR.rglob("*")):
        if p.is_file():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  {p.relative_to(MODEL_DIR)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/download_zipformer_zh.py
git commit -m "feat: add Zipformer model download script"
```

---

### Task 7: 性能指标文档更新

**Files:**
- Modify: `docs/级联架构性能指标设计.md`

- [ ] **Step 1: 在 ASR 阶段表格后新增本地 ASR 章节**

在"ASR 阶段（语音转文字）"章节末尾，B5 删除说明之后，追加：

```markdown
##### 本地 ASR（如 Zipformer sherpa-onnx）⭐ v2.1 新增

| 编号 | 指标名称 | 定义 | 热点判定 | 说明 |
|------|---------|------|---------|------|
| **B1** | ASR 连接(本地) | 始终 0ms | - | 本地无需连接，标记 reuse |
| **B2** | ASR 初始化(本地) | 始终 0ms | - | 本地无会话，标记 reuse |
| **B3** | ASR chunk 解码 | 每个 chunk 解码耗时 | 仅监控 | 本地内存操作 |
| **B4** | ASR 本地处理 | end_stream→result_delivered | >100ms | **核心热点**，关键路径 |

> **自动检测逻辑**：
> - 本地 ASR：有 `asr_local_ready` 事件
> - 云端 ASR：有 `asr_ws_connected` 或 `asr_ws_reused` 事件
```

在 L2 内部指标阈值的 ASR 阶段表格后追加：

```markdown

##### ASR 阶段（本地 Zipformer）⭐ v2.1 新增

| 指标 | ✅ Excellent | 👍 Good | ⚠️ Acceptable | 热点判定 | TTFB公式 |
|------|-------------|---------|---------------|----------|---------|
| B1 ASR 连接(本地) | 始终 0ms | - | - | - | ❌ reuse |
| B2 ASR 初始化(本地) | 始终 0ms | - | - | - | ❌ reuse |
| B3 ASR chunk 解码 | - | - | - | 仅监控 | ❌ |
| **B4 ASR 本地处理** | ≤30ms | ≤50ms | ≤100ms | >100ms | ✅ **核心热点** |
```

在变更记录表中追加：

```markdown
| v2.1 | 2026-05-07 | 新增本地 ASR (Zipformer sherpa-onnx) B 指标：B1/B2 本地 reuse，B4 本地处理阈值 |
```

- [ ] **Step 2: Commit**

```bash
git add docs/级联架构性能指标设计.md
git commit -m "docs: add local ASR timing metrics to performance design doc"
```

---

### Task 8: 全量测试验证

- [ ] **Step 1: 运行 Zipformer 相关全部测试**

Run: `cd /mnt/d/wsl/peiliao/T1-reference-apps/roboticsreachy-mini-chatbox && python -m pytest tests/cascade/test_zipformer_sherpa.py tests/cascade/test_timing_metrics.py -v`
Expected: ALL PASS

- [ ] **Step 2: 运行全量测试套件确认无回归**

Run: `cd /mnt/d/wsl/peiliao/T1-reference-apps/roboticsreachy-mini-chatbox && python -m pytest tests/ -v --timeout=60`
Expected: ALL PASS

- [ ] **Step 3: 验证 cascade.yaml 可被正确解析**

Run: `cd /mnt/d/wsl/peiliao/T1-reference-apps/roboticsreachy-mini-chatbox && python -c "from reachy_mini_conversation_app.cascade.config import get_config; c = get_config(); info = c.get_asr_provider_info('zipformer_sherpa'); print(f'OK: {info[\"class\"]}')"` 
Expected: `OK: ZipformerSherpaASR`

- [ ] **Step 4: Final commit (if any fixes)**

```bash
git add -A
git commit -m "test: full test suite verification for Zipformer ASR provider"
```
