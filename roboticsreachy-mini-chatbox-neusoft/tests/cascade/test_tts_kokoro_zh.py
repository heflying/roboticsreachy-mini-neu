"""Tests for KokoroZHTTS provider (Chinese Kokoro-82M-v1.1-zh)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator
from unittest.mock import MagicMock, patch
import sys

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fakes for kokoro.KModel / KPipeline (avoids installing kokoro in test env)
# ---------------------------------------------------------------------------


@dataclass
class FakeGraphemeResult:
    """Mimics a KPipeline yield result with .phonemes attribute."""

    phonemes: str = "n i3 h ao3"


@dataclass
class FakeAudioResult:
    """Mimics a KPipeline yield result with .audio attribute."""

    audio: np.ndarray = field(default_factory=lambda: np.random.randn(8000).astype(np.float32) * 0.3)


class FakeKModel:
    """Mimics kokoro.KModel."""

    def __init__(self, **kwargs: object) -> None:
        self._device = "cpu"

    def to(self, device: str) -> "FakeKModel":
        self._device = device
        return self

    def eval(self) -> "FakeKModel":
        return self


class FakeKPipeline:
    """Mimics kokoro.KPipeline for Chinese pipeline."""

    def __init__(
        self,
        lang_code: str = "z",
        repo_id: str = "hexgrad/Kokoro-82M-v1.1-zh",
        model: FakeKModel | None | bool = None,
        en_callable: object | None = None,
    ) -> None:
        self.lang_code = lang_code
        self.repo_id = repo_id
        self._model = model
        self._en_callable = en_callable
        self._default_audio = np.random.randn(8000).astype(np.float32) * 0.3

    def __call__(
        self, text: str, voice: str = "zf_001", speed: object = None, **kwargs: object
    ):
        """Return generator of results (mimics KPipeline iteration).

        Yields FakeGraphemeResult for English pipeline (lang_code='a'),
        FakeAudioResult for Chinese pipeline (lang_code='z').
        """
        if self.lang_code == "a":
            yield FakeGraphemeResult()
        else:
            yield FakeAudioResult(audio=self._default_audio)


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
def mock_kokoro() -> Generator[None, None, None]:
    """Inject fake kokoro module via sys.modules so imports succeed."""
    fake_kokoro = MagicMock()
    fake_kokoro.KModel = FakeKModel
    fake_kokoro.KPipeline = FakeKPipeline

    with patch.dict(sys.modules, {"kokoro": fake_kokoro}):
        yield


@pytest.fixture()
def tts(mock_kokoro: None) -> "KokoroZHTTS":
    """Create a KokoroZHTTS instance with mocked dependencies."""
    from reachy_mini_conversation_app.cascade.tts.kokoro_zh import KokoroZHTTS

    return KokoroZHTTS(voice="zf_001")


# ---------------------------------------------------------------------------
# Tests: initialization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_init_default_voice(mock_kokoro: None) -> None:
    """KokoroZHTTS initializes with default Chinese voice."""
    from reachy_mini_conversation_app.cascade.tts.kokoro_zh import KokoroZHTTS

    tts = KokoroZHTTS()
    assert tts.sample_rate == 24000
    assert tts.default_voice == "zf_001"


@pytest.mark.asyncio
async def test_init_custom_voice(mock_kokoro: None) -> None:
    """KokoroZHTTS accepts custom Chinese voice."""
    from reachy_mini_conversation_app.cascade.tts.kokoro_zh import KokoroZHTTS

    tts = KokoroZHTTS(voice="zm_010")
    assert tts.default_voice == "zm_010"


@pytest.mark.asyncio
async def test_init_creates_model_and_pipelines(mock_kokoro: None) -> None:
    """KokoroZHTTS creates KModel + en_pipeline + zh_pipeline."""
    from reachy_mini_conversation_app.cascade.tts.kokoro_zh import KokoroZHTTS

    tts = KokoroZHTTS()
    assert hasattr(tts, "_model")
    assert hasattr(tts, "_en_pipeline")
    assert hasattr(tts, "_zh_pipeline")


@pytest.mark.asyncio
async def test_sample_rate_is_24000(mock_kokoro: None) -> None:
    """KokoroZHTTS reports 24kHz sample rate."""
    from reachy_mini_conversation_app.cascade.tts.kokoro_zh import KokoroZHTTS

    tts = KokoroZHTTS()
    assert tts.sample_rate == 24000


# ---------------------------------------------------------------------------
# Tests: synthesize
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_yields_pcm_int16(tts: "KokoroZHTTS") -> None:
    """synthesize yields PCM int16 bytes."""
    chunks: list[bytes] = []
    async for chunk in tts.synthesize("你好世界"):
        chunks.append(chunk)

    assert len(chunks) > 0
    for chunk in chunks:
        arr = np.frombuffer(chunk, dtype=np.int16)
        assert len(arr) > 0
        assert arr.dtype == np.int16


@pytest.mark.asyncio
async def test_synthesize_empty_text_yields_nothing(tts: "KokoroZHTTS") -> None:
    """synthesize returns immediately for empty/whitespace text."""
    chunks: list[bytes] = []
    async for chunk in tts.synthesize("   "):
        chunks.append(chunk)

    assert len(chunks) == 0


@pytest.mark.asyncio
async def test_synthesize_timing_marks(tts: "KokoroZHTTS") -> None:
    """synthesize marks tts_start, tts_model_generation_start, tts_first_chunk_ready."""
    from reachy_mini_conversation_app.cascade.timing import tracker

    tracker.reset("test_turn")

    chunks: list[bytes] = []
    async for chunk in tts.synthesize("测试"):
        chunks.append(chunk)

    assert tracker.has_event("tts_start")
    assert tracker.has_event("tts_model_generation_start")
    assert tracker.has_event("tts_first_chunk_ready")


@pytest.mark.asyncio
async def test_synthesize_local_tts_detection(tts: "KokoroZHTTS") -> None:
    """KokoroZHTTS marks local TTS events (detected by _is_local_tts)."""
    from reachy_mini_conversation_app.cascade.timing import tracker

    tracker.reset("test_turn")

    async for _ in tts.synthesize("你好"):
        pass

    assert tracker.has_event("tts_model_generation_start")
    assert not tracker.has_event("tts_ws_connected")
    assert tracker._is_local_tts()


@pytest.mark.asyncio
async def test_synthesize_voice_override(mock_kokoro: None) -> None:
    """synthesize accepts voice parameter override."""
    from reachy_mini_conversation_app.cascade.tts.kokoro_zh import KokoroZHTTS

    tts = KokoroZHTTS(voice="zf_001")

    chunks: list[bytes] = []
    async for chunk in tts.synthesize("测试", voice="zm_010"):
        chunks.append(chunk)

    assert len(chunks) > 0


# ---------------------------------------------------------------------------
# Tests: speed callable for long text
# ---------------------------------------------------------------------------


def test_speed_callable_returns_1_for_short_text() -> None:
    """Speed callable returns 1.0 for short phoneme sequences."""
    from reachy_mini_conversation_app.cascade.tts.kokoro_zh import make_speed_callable

    callable_fn = make_speed_callable()
    result = callable_fn(50)
    assert result == 1.0


def test_speed_callable_slows_for_long_text() -> None:
    """Speed callable slows down for long phoneme sequences (>83)."""
    from reachy_mini_conversation_app.cascade.tts.kokoro_zh import make_speed_callable

    callable_fn = make_speed_callable()
    result = callable_fn(150)
    assert result < 1.0
    assert result > 0.8


def test_speed_callable_monotonic_decrease() -> None:
    """Speed callable monotonically decreases as phoneme count increases."""
    from reachy_mini_conversation_app.cascade.tts.kokoro_zh import make_speed_callable

    callable_fn = make_speed_callable()
    prev = 1.0
    for n in [50, 83, 100, 150, 200, 300]:
        current = callable_fn(n)
        assert current <= prev
        prev = current


# ---------------------------------------------------------------------------
# Tests: en_callable for mixed Chinese-English
# ---------------------------------------------------------------------------


def test_en_callable_produces_phonemes(mock_kokoro: None) -> None:
    """en_callable returns phoneme string from English pipeline."""
    from reachy_mini_conversation_app.cascade.tts.kokoro_zh import make_en_callable

    callable_fn = make_en_callable(repo_id="hexgrad/Kokoro-82M-v1.1-zh")
    result = callable_fn("hello")
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# Tests: config validation
# ---------------------------------------------------------------------------


def test_cascade_yaml_has_kokoro_zh() -> None:
    """cascade.yaml should have kokoro_zh TTS provider config."""
    import yaml
    from pathlib import Path

    config_path = Path("cascade.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    providers = config["tts"]["providers"]
    assert "kokoro_zh" in providers
    assert providers["kokoro_zh"]["module"] == "kokoro_zh"
    assert providers["kokoro_zh"]["class"] == "KokoroZHTTS"
    assert providers["kokoro_zh"]["location"] == "local"


def test_pyproject_has_cascade_kokoro_zh() -> None:
    """pyproject.toml should have cascade_kokoro_zh optional dependency."""
    import tomllib
    from pathlib import Path

    pyproject = Path("pyproject.toml")
    with open(pyproject, "rb") as f:
        data = tomllib.load(f)

    extras = data["project"].get("optional-dependencies", {})
    assert "cascade_kokoro_zh" in extras


# ---------------------------------------------------------------------------
# End-to-end tests: real Kokoro-82M-v1.1-zh (requires kokoro package)
# ---------------------------------------------------------------------------


def _kokoro_zh_available() -> bool:
    try:
        import kokoro  # noqa: F401
        return True
    except ImportError:
        return False


requires_kokoro_zh = pytest.mark.skipif(
    not _kokoro_zh_available(),
    reason="kokoro not installed (uv sync --extra cascade_kokoro_zh)",
)


@requires_kokoro_zh
@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_synthesize_chinese() -> None:
    """Real Kokoro-zh: synthesize Chinese text produces valid PCM audio."""
    from reachy_mini_conversation_app.cascade.tts.kokoro_zh import KokoroZHTTS

    tts = KokoroZHTTS(voice="zf_001")

    chunks: list[bytes] = []
    async for chunk in tts.synthesize("你好世界"):
        chunks.append(chunk)

    assert len(chunks) > 0, "Should produce at least one audio chunk"

    total_samples = sum(len(np.frombuffer(c, dtype=np.int16)) for c in chunks)
    duration_ms = total_samples / tts.sample_rate * 1000
    assert duration_ms > 50, f"Audio too short: {duration_ms:.0f}ms"

    for chunk in chunks:
        arr = np.frombuffer(chunk, dtype=np.int16)
        assert np.max(np.abs(arr)) > 0, "Chunk should not be all-zero"


@requires_kokoro_zh
@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_sample_rate_24khz() -> None:
    """Real Kokoro-zh: sample rate is 24000Hz."""
    from reachy_mini_conversation_app.cascade.tts.kokoro_zh import KokoroZHTTS

    tts = KokoroZHTTS(voice="zf_001")
    assert tts.sample_rate == 24000


@requires_kokoro_zh
@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_empty_text_no_output() -> None:
    """Real Kokoro-zh: empty text produces no audio."""
    from reachy_mini_conversation_app.cascade.tts.kokoro_zh import KokoroZHTTS

    tts = KokoroZHTTS(voice="zf_001")

    chunks: list[bytes] = []
    async for chunk in tts.synthesize("   "):
        chunks.append(chunk)
    assert len(chunks) == 0


@requires_kokoro_zh
@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_long_text_produces_audio() -> None:
    """Real Kokoro-zh: long Chinese text produces substantial audio."""
    from reachy_mini_conversation_app.cascade.tts.kokoro_zh import KokoroZHTTS

    tts = KokoroZHTTS(voice="zf_001")

    text = "你好，我是Reachy Mini，一个可以和你语音对话的小机器人。"
    chunks: list[bytes] = []
    async for chunk in tts.synthesize(text):
        chunks.append(chunk)

    assert len(chunks) >= 1, "Should produce at least one audio chunk"
    total_samples = sum(len(np.frombuffer(c, dtype=np.int16)) for c in chunks)
    duration_ms = total_samples / tts.sample_rate * 1000
    assert duration_ms > 300, f"Audio too short for long text: {duration_ms:.0f}ms"


@requires_kokoro_zh
@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_male_voice() -> None:
    """Real Kokoro-zh: zm_010 male voice works."""
    from reachy_mini_conversation_app.cascade.tts.kokoro_zh import KokoroZHTTS

    tts = KokoroZHTTS(voice="zm_010")

    chunks: list[bytes] = []
    async for chunk in tts.synthesize("这是男声测试"):
        chunks.append(chunk)

    assert len(chunks) > 0
    total_samples = sum(len(np.frombuffer(c, dtype=np.int16)) for c in chunks)
    duration_ms = total_samples / tts.sample_rate * 1000
    assert duration_ms > 50, f"Audio too short: {duration_ms:.0f}ms"


@requires_kokoro_zh
@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_mixed_chinese_english() -> None:
    """Real Kokoro-zh: mixed Chinese-English text works via en_callable."""
    from reachy_mini_conversation_app.cascade.tts.kokoro_zh import KokoroZHTTS

    tts = KokoroZHTTS(voice="zf_001")

    chunks: list[bytes] = []
    async for chunk in tts.synthesize("你好，我的名字叫Hello World"):
        chunks.append(chunk)

    assert len(chunks) > 0
    total_samples = sum(len(np.frombuffer(c, dtype=np.int16)) for c in chunks)
    duration_ms = total_samples / tts.sample_rate * 1000
    assert duration_ms > 50, f"Audio too short: {duration_ms:.0f}ms"


@requires_kokoro_zh
@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_timing_marks_complete() -> None:
    """Real Kokoro-zh: all local TTS timing marks are recorded."""
    from reachy_mini_conversation_app.cascade.timing import tracker
    from reachy_mini_conversation_app.cascade.tts.kokoro_zh import KokoroZHTTS

    tts = KokoroZHTTS(voice="zf_001")
    tracker.reset("e2e_turn")

    async for _ in tts.synthesize("测试timing标记"):
        pass

    assert tracker.has_event("tts_start")
    assert tracker.has_event("tts_model_generation_start")
    assert tracker.has_event("tts_first_chunk_ready")
    assert tracker.has_event("tts_model_generation_complete")

    # Verify local TTS auto-detection
    assert tracker._is_local_tts()
