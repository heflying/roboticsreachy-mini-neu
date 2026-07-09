"""Tests for VAD backend switching and timing."""

import importlib
import logging
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest

from reachy_mini_conversation_app.cascade.vad import (
    SILERO_SAMPLE_RATE,
    VADBackend,
    SileroVAD,
    SileroVADPyTorch,
)


def _silence_audio() -> npt.NDArray[Any]:
    return np.zeros(512, dtype=np.int16)


def _speech_like_audio() -> npt.NDArray[Any]:
    return np.random.randint(-5000, 5000, 512, dtype=np.int16)


class TestVADBackendProtocol:
    """Test VADBackend Protocol conformance."""

    def test_complete_backend_satisfies_protocol(self):
        class FakeBackend:
            def get_speech_prob(self, audio: npt.NDArray[Any], sample_rate: int) -> float:
                return 0.5

            def reset_states(self) -> None:
                pass

        assert isinstance(FakeBackend(), VADBackend)

    def test_missing_reset_states_fails_protocol(self):
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
        vad.sentence_pause_threshold_ms = 200
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
            # Timing log only fires every 50th call
            for _ in range(50):
                vad.get_speech_prob(_silence_audio())
        assert any(
            "backend=onnx" in r.message and "inference=" in r.message for r in caplog.records
        )

    def test_process_chunk_no_speech(self):
        vad = self._make_facade()
        vad._backend.get_speech_prob.return_value = 0.1
        started, ended, sentence_pause = vad.process_chunk(_silence_audio())
        assert not started
        assert not ended
        assert not sentence_pause

    def test_process_chunk_speech_start(self):
        vad = self._make_facade()
        vad._backend.get_speech_prob.return_value = 0.9
        started = False
        for _ in range(10):
            started, _, _ = vad.process_chunk(_speech_like_audio())
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
        backend.reset_states()

    @pytest.mark.skipif(
        not importlib.util.find_spec("torch"),
        reason="torch not installed",
    )
    def test_silence_has_low_probability(self):
        backend = SileroVADPyTorch()
        prob = backend.get_speech_prob(_silence_audio(), SILERO_SAMPLE_RATE)
        assert prob < 0.5


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
        if not importlib.util.find_spec("torch"):
            pytest.skip("torch not installed")

        from reachy_mini_conversation_app.cascade.vad_onnx import SileroVADOnnx

        pytorch = SileroVADPyTorch()
        onnx = SileroVADOnnx()
        audio = _silence_audio()
        prob_pt = pytorch.get_speech_prob(audio, SILERO_SAMPLE_RATE)
        prob_onnx = onnx.get_speech_prob(audio, SILERO_SAMPLE_RATE)
        assert prob_pt < 0.5
        assert prob_onnx < 0.5
        assert abs(prob_pt - prob_onnx) < 0.15
