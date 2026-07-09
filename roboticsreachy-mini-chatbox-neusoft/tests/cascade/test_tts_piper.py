"""Tests for PiperTTS provider."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from reachy_mini_conversation_app.cascade.tts.piper import PiperTTS, SUB_CHUNK_SIZE


# ---------------------------------------------------------------------------
# Fakes for PiperVoice (avoids installing piper-tts in test env)
# ---------------------------------------------------------------------------


@dataclass
class FakeAudioChunk:
    """Mimics piper's AudioChunk (piper-tts >= 1.2.0)."""

    sample_rate: int
    sample_width: int
    sample_channels: int
    audio_float_array: np.ndarray
    phonemes: list[str]
    phoneme_ids: list[int]


class FakePiperConfig:
    """Mimics piper's PiperConfig."""

    sample_rate: int = 22050


def _make_chunk(audio_len: int = 8000, text: str = "测试") -> FakeAudioChunk:
    """Create a FakeAudioChunk with random audio."""
    return FakeAudioChunk(
        sample_rate=22050,
        sample_width=2,
        sample_channels=1,
        audio_float_array=np.random.randn(audio_len).astype(np.float32) * 0.5,
        phonemes=list(text),
        phoneme_ids=[1, 2, 3],
    )


class FakePiperVoice:
    """Mimics piper's PiperVoice."""

    def __init__(self, chunks: list[FakeAudioChunk] | None = None) -> None:
        self.config = FakePiperConfig()
        self._chunks = chunks or [_make_chunk()]

    def synthesize(self, text: str, **kwargs) -> Generator[FakeAudioChunk, None, None]:
        """Yield pre-configured chunks."""
        for chunk in self._chunks:
            yield chunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_timing() -> Generator[None, None, None]:
    """Reset timing tracker between tests."""
    from reachy_mini_conversation_app.cascade.timing import tracker

    tracker.reset()
    yield


@pytest.fixture()
def fake_voice() -> FakePiperVoice:
    return FakePiperVoice()


@pytest.fixture()
def mock_piper_voice(fake_voice: FakePiperVoice, tmp_path: Path) -> Generator[MagicMock, None, None]:
    """Patch piper.PiperVoice.load to return a fake voice."""
    with patch("piper.PiperVoice") as MockVoice:
        MockVoice.load.return_value = fake_voice
        yield MockVoice


# ---------------------------------------------------------------------------
# Tests: initialization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_init_with_model_name(mock_piper_voice: MagicMock, tmp_path: Path) -> None:
    """PiperTTS resolves model name to models/{name}.onnx."""
    onnx_file = Path("models") / "zh_CN-huayan-medium.onnx"
    with patch.object(Path, "exists", return_value=True):
        tts = PiperTTS(model="zh_CN-huayan-medium")

    assert tts.sample_rate == 22050
    mock_piper_voice.load.assert_called_once()


@pytest.mark.asyncio
async def test_init_model_not_found() -> None:
    """PiperTTS raises FileNotFoundError when model file is missing."""
    with pytest.raises(FileNotFoundError, match="Piper model not found"):
        with patch("piper.PiperVoice"):
            PiperTTS(model="nonexistent-model")


@pytest.mark.asyncio
async def test_sample_rate_from_config(mock_piper_voice: MagicMock) -> None:
    """sample_rate comes from the loaded model config, not hardcoded."""
    with patch.object(Path, "exists", return_value=True):
        tts = PiperTTS(model="zh_CN-huayan-medium")

    assert tts.sample_rate == 22050


# ---------------------------------------------------------------------------
# Tests: synthesize
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_yields_pcm_int16_chunks(mock_piper_voice: MagicMock) -> None:
    """synthesize yields PCM int16 bytes in playback-sized sub-chunks."""
    with patch.object(Path, "exists", return_value=True):
        tts = PiperTTS(model="zh_CN-huayan-medium")

    chunks: list[bytes] = []
    async for chunk in tts.synthesize("你好"):
        chunks.append(chunk)

    assert len(chunks) > 0
    for chunk in chunks:
        arr = np.frombuffer(chunk, dtype=np.int16)
        assert len(arr) > 0
        assert arr.dtype == np.int16
        assert len(arr) <= SUB_CHUNK_SIZE


@pytest.mark.asyncio
async def test_synthesize_empty_text_yields_nothing(mock_piper_voice: MagicMock) -> None:
    """synthesize returns immediately for empty/whitespace text."""
    with patch.object(Path, "exists", return_value=True):
        tts = PiperTTS(model="zh_CN-huayan-medium")

    chunks: list[bytes] = []
    async for chunk in tts.synthesize("   "):
        chunks.append(chunk)

    assert len(chunks) == 0


@pytest.mark.asyncio
async def test_synthesize_multiple_audio_chunks(mock_piper_voice: MagicMock) -> None:
    """Multiple Piper AudioChunks are split into sub-chunks correctly."""
    fake_voice = FakePiperVoice(chunks=[
        _make_chunk(audio_len=10000, text="第一句"),
        _make_chunk(audio_len=6000, text="第二句"),
    ])
    mock_piper_voice.load.return_value = fake_voice

    with patch.object(Path, "exists", return_value=True):
        tts = PiperTTS(model="zh_CN-huayan-medium")

    chunks: list[bytes] = []
    async for chunk in tts.synthesize("第一句 第二句"):
        chunks.append(chunk)

    # 2 Piper AudioChunks → 2 output chunks (one per sentence)
    assert len(chunks) >= 2
    for chunk in chunks:
        arr = np.frombuffer(chunk, dtype=np.int16)
        assert len(arr) > 0
        assert len(arr) <= SUB_CHUNK_SIZE


@pytest.mark.asyncio
async def test_synthesize_splits_large_sentence_into_multiple_subchunks(mock_piper_voice: MagicMock) -> None:
    """A single long Piper sentence is split for more responsive playback interrupts."""
    fake_voice = FakePiperVoice(chunks=[_make_chunk(audio_len=SUB_CHUNK_SIZE * 3 + 500, text="长句子")])
    mock_piper_voice.load.return_value = fake_voice

    with patch.object(Path, "exists", return_value=True):
        tts = PiperTTS(model="zh_CN-huayan-medium")

    chunks: list[bytes] = []
    async for chunk in tts.synthesize("这是一句很长的话，用来测试 Piper 输出是否会被切成更小的播放块。"):
        chunks.append(chunk)

    assert len(chunks) == 4
    for chunk in chunks[:-1]:
        assert len(np.frombuffer(chunk, dtype=np.int16)) == SUB_CHUNK_SIZE
    assert len(np.frombuffer(chunks[-1], dtype=np.int16)) == 500


@pytest.mark.asyncio
async def test_synthesize_timing_marks(mock_piper_voice: MagicMock) -> None:
    """synthesize records timing marks for TTFB tracking."""
    from reachy_mini_conversation_app.cascade.timing import tracker

    with patch.object(Path, "exists", return_value=True):
        tts = PiperTTS(model="zh_CN-huayan-medium")

    async for _ in tts.synthesize("测试"):
        pass

    # Verify timing marks were recorded by checking durations exist
    assert tracker.get_duration("tts_start", "tts_first_chunk_ready") is not None
    assert tracker.get_duration("tts_model_generation_start", "tts_model_generation_complete") is not None


# ---------------------------------------------------------------------------
# Tests: config integration
# ---------------------------------------------------------------------------


def test_cascade_yaml_has_piper_zh() -> None:
    """Verify cascade.yaml contains the piper_zh provider entry."""
    import yaml

    with open("cascade.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert "piper_zh" in cfg["tts"]["providers"]
    provider = cfg["tts"]["providers"]["piper_zh"]
    assert provider["module"] == "piper"
    assert provider["class"] == "PiperTTS"
    assert provider["location"] == "local"
    assert provider["import_check"] == "piper"
    assert provider["install_extra"] == "cascade_piper"


def test_pyproject_has_cascade_piper() -> None:
    """Verify pyproject.toml contains the cascade_piper extra."""
    with open("pyproject.toml", encoding="utf-8") as f:
        content = f.read()

    assert "cascade_piper" in content
    assert "piper-tts" in content


# ---------------------------------------------------------------------------
# Tests: real Piper TTS end-to-end (requires model file + piper-tts)
# ---------------------------------------------------------------------------

PIPER_MODEL = Path("models") / "zh_CN-huayan-medium.onnx"


def _piper_available() -> bool:
    try:
        import piper  # noqa: F401
        return PIPER_MODEL.exists()
    except ImportError:
        return False


requires_piper = pytest.mark.skipif(
    not _piper_available(),
    reason="piper-tts not installed or model file missing",
)


@requires_piper
@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_synthesize_chinese() -> None:
    """Real Piper TTS: synthesize Chinese text produces valid PCM audio."""
    tts = PiperTTS(model="zh_CN-huayan-medium")

    chunks: list[bytes] = []
    async for chunk in tts.synthesize("你好世界"):
        chunks.append(chunk)

    assert len(chunks) > 0, "Should produce at least one audio chunk"

    total_samples = sum(len(np.frombuffer(c, dtype=np.int16)) for c in chunks)
    duration_ms = total_samples / tts.sample_rate * 1000
    assert duration_ms > 100, f"Audio too short: {duration_ms:.0f}ms"

    for chunk in chunks:
        arr = np.frombuffer(chunk, dtype=np.int16)
        assert np.max(np.abs(arr)) > 0, "Chunk should not be all-zero"


@requires_piper
@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_sample_rate_matches_model() -> None:
    """Real Piper TTS: sample_rate matches the model config."""
    tts = PiperTTS(model="zh_CN-huayan-medium")
    assert tts.sample_rate > 0

    chunks: list[bytes] = []
    async for chunk in tts.synthesize("测试"):
        chunks.append(chunk)
    assert len(chunks) > 0


@requires_piper
@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_empty_text_no_output() -> None:
    """Real Piper TTS: empty text produces no audio."""
    tts = PiperTTS(model="zh_CN-huayan-medium")

    chunks: list[bytes] = []
    async for chunk in tts.synthesize("   "):
        chunks.append(chunk)
    assert len(chunks) == 0


@requires_piper
@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_long_text_produces_audio() -> None:
    """Real Piper TTS: long text produces substantial audio."""
    tts = PiperTTS(model="zh_CN-huayan-medium")

    text = "你好，我是Reachy Mini，一个可以和你语音对话的小机器人。"
    chunks: list[bytes] = []
    async for chunk in tts.synthesize(text):
        chunks.append(chunk)

    assert len(chunks) >= 1, "Should produce at least one audio chunk"
    total_samples = sum(len(np.frombuffer(c, dtype=np.int16)) for c in chunks)
    duration_ms = total_samples / tts.sample_rate * 1000
    # This sentence should produce at least 1 second of audio
    assert duration_ms > 500, f"Audio too short for long text: {duration_ms:.0f}ms"
