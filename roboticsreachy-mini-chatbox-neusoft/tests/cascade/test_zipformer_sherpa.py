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


def test_download_triggered_when_files_missing(tmp_path: Path):
    """_download_if_missing should call snapshot_download when files are missing."""
    model_dir = tmp_path / "missing-model"

    with patch("huggingface_hub.snapshot_download") as mock_dl:
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
        with patch.object(ZipformerSherpaASR, "_create_recognizer"):
            provider = ZipformerSherpaASR(
                model_dir=str(model_dir),
                sample_rate=16000,
            )
            provider._recognizer = MockOnlineRecognizer(partial=partial, final=final)
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


# ---------------------------------------------------------------------------
# Regression: small-chunk buffering (GetFrames assertion fix)
# ---------------------------------------------------------------------------

def test_small_chunks_are_buffered(tmp_path: Path):
    """Chunks smaller than _min_feed_samples should be buffered, not sent."""
    async def run():
        provider = _make_provider(tmp_path, partial="", final="最终")
        await provider.start_stream()

        # Send a tiny chunk (10ms @ 16kHz = 160 samples, well below 4800)
        tiny_wav = _silence_wav(10, 16000)
        await provider.send_audio_chunk(tiny_wav)

        # Buffer should hold the samples, nothing sent to stream yet
        assert len(provider._audio_buffer) > 0
        assert len(provider._stream._audio_chunks) == 0

    asyncio.run(run())


def test_buffered_audio_flushed_on_large_chunk(tmp_path: Path):
    """Buffered audio is flushed when enough data accumulates."""
    async def run():
        provider = _make_provider(tmp_path, partial="", final="最终")
        await provider.start_stream()

        # Send multiple small chunks that together exceed the threshold
        for _ in range(10):
            await provider.send_audio_chunk(_silence_wav(50, 16000))

        # At least one accept_waveform should have been called
        assert len(provider._stream._audio_chunks) >= 1

    asyncio.run(run())


def test_buffer_flushed_on_end_stream(tmp_path: Path):
    """Remaining buffered audio is flushed when end_stream is called."""
    async def run():
        provider = _make_provider(tmp_path, partial="", final="最终结果")
        await provider.start_stream()

        # Send a small chunk that stays buffered
        await provider.send_audio_chunk(_silence_wav(10, 16000))
        assert len(provider._audio_buffer) > 0

        result = await provider.end_stream()
        # Buffer should be flushed
        assert len(provider._audio_buffer) == 0
        # The buffered audio should have been sent before input_finished
        assert len(provider._stream._audio_chunks) >= 1
        assert result == "最终结果"

    asyncio.run(run())
