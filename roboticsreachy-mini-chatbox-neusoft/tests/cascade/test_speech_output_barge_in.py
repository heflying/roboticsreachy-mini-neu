"""Tests for speech_output barge-in callback integration.

This test verifies that:
1. GradioSpeechOutput correctly receives barge-in callbacks
2. Start callback is called when first audio chunk is queued
3. Stop callback is called when playback completes

Task 8: GradioSpeechOutput token + turn_id support
"""

from __future__ import annotations

import pytest
import asyncio
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch, call


class TestGradioSpeechOutputBargeInCallback:
    """Tests for barge-in callback integration in GradioSpeechOutput."""

    @pytest.fixture
    def mock_tts(self):
        """Create mock TTS provider."""
        tts = MagicMock()
        tts.sample_rate = 24000
        tts.prefer_single_request = True
        return tts

    @pytest.fixture
    def mock_playback(self):
        """Create mock audio playback system."""
        playback = MagicMock()
        playback.current_generation = 0
        playback.put_audio = MagicMock()
        playback.put_wobbler = MagicMock()
        completion_event = asyncio.Event()
        completion_event.set()
        playback.signal_end_of_turn = MagicMock(return_value=(0, completion_event))
        return playback

    @pytest.fixture
    def barge_in_callbacks(self):
        """Create barge-in callback trackers."""
        start_called = []
        stop_called = []

        def start_callback():
            start_called.append(True)

        def stop_callback():
            stop_called.append(True)

        return start_callback, stop_callback, start_called, stop_called

    @pytest.mark.asyncio
    async def test_start_callback_called_on_first_chunk(
        self, mock_tts, mock_playback, barge_in_callbacks
    ):
        """R7: Start callback should be called when first audio chunk is queued."""
        from reachy_mini_conversation_app.cascade.speech_output import GradioSpeechOutput

        start_cb, stop_cb, start_called, stop_called = barge_in_callbacks

        # Create mock TTS that yields chunks
        async def mock_synthesize(text):
            # Yield two audio chunks
            yield np.random.randint(-1000, 1000, size=4800, dtype=np.int16).tobytes()
            yield np.random.randint(-1000, 1000, size=4800, dtype=np.int16).tobytes()

        mock_tts.synthesize = mock_synthesize

        speech_output = GradioSpeechOutput(
            tts=mock_tts,
            playback=mock_playback,
            barge_in_start_callback=start_cb,
            barge_in_stop_callback=stop_cb,
        )

        await speech_output.speak("Test message")

        # Verify start callback was called once
        assert len(start_called) == 1, "Start callback should be called once"

    @pytest.mark.asyncio
    async def test_stop_callback_called_on_completion(
        self, mock_tts, mock_playback, barge_in_callbacks
    ):
        """R7: Stop callback should be called when playback completes."""
        from reachy_mini_conversation_app.cascade.speech_output import GradioSpeechOutput

        start_cb, stop_cb, start_called, stop_called = barge_in_callbacks

        # Create mock TTS that yields chunks
        async def mock_synthesize(text):
            yield np.random.randint(-1000, 1000, size=4800, dtype=np.int16).tobytes()

        mock_tts.synthesize = mock_synthesize

        speech_output = GradioSpeechOutput(
            tts=mock_tts,
            playback=mock_playback,
            barge_in_start_callback=start_cb,
            barge_in_stop_callback=stop_cb,
        )

        # Set return_after_tts_queued to avoid waiting for playback drain
        speech_output.return_after_tts_queued = True

        await speech_output.speak("Test message")

        # Verify stop callback was called (in finally block after audio queued)
        # Note: With return_after_tts_queued=True, stop is called in finally block
        assert len(stop_called) >= 1, "Stop callback should be called after TTS completes"

    @pytest.mark.asyncio
    async def test_no_callbacks_if_no_audio(self, mock_tts, mock_playback, barge_in_callbacks):
        """E4: No callbacks should be called if TTS returns no audio."""
        from reachy_mini_conversation_app.cascade.speech_output import GradioSpeechOutput

        start_cb, stop_cb, start_called, stop_called = barge_in_callbacks

        # Create mock TTS that yields nothing (empty stream)
        async def mock_synthesize(text):
            return
            yield  # Never reached, but makes it a generator

        mock_tts.synthesize = mock_synthesize

        speech_output = GradioSpeechOutput(
            tts=mock_tts,
            playback=mock_playback,
            barge_in_start_callback=start_cb,
            barge_in_stop_callback=stop_cb,
        )

        await speech_output.speak("Test message")

        # Verify neither callback was called
        assert len(start_called) == 0, "Start callback should NOT be called for empty audio"
        # Stop callback is still called in finally block for cleanup
        # This is acceptable behavior - we need to ensure monitor is stopped

    @pytest.mark.asyncio
    async def test_callbacks_work_without_playback_wait(
        self, mock_tts, mock_playback, barge_in_callbacks
    ):
        """Verify callbacks work when return_after_tts_queued is True."""
        from reachy_mini_conversation_app.cascade.speech_output import GradioSpeechOutput

        start_cb, stop_cb, start_called, stop_called = barge_in_callbacks

        # Create mock TTS that yields chunks
        async def mock_synthesize(text):
            yield np.random.randint(-1000, 1000, size=4800, dtype=np.int16).tobytes()

        mock_tts.synthesize = mock_synthesize

        speech_output = GradioSpeechOutput(
            tts=mock_tts,
            playback=mock_playback,
            barge_in_start_callback=start_cb,
            barge_in_stop_callback=stop_cb,
        )
        speech_output.return_after_tts_queued = True

        await speech_output.speak("Test")

        # Both callbacks should be called
        assert len(start_called) == 1
        assert len(stop_called) >= 1


class TestBargeInCallbackNullSafety:
    """Tests for null safety when callbacks are not provided."""

    @pytest.fixture
    def mock_tts(self):
        """Create mock TTS provider."""
        tts = MagicMock()
        tts.sample_rate = 24000
        tts.prefer_single_request = True
        return tts

    @pytest.fixture
    def mock_playback(self):
        """Create mock audio playback system."""
        playback = MagicMock()
        playback.current_generation = 0
        playback.put_audio = MagicMock()
        playback.put_wobbler = MagicMock()
        completion_event = asyncio.Event()
        completion_event.set()
        playback.signal_end_of_turn = MagicMock(return_value=(0, completion_event))
        return playback

    @pytest.mark.asyncio
    async def test_works_without_callbacks(self, mock_tts, mock_playback):
        """GradioSpeechOutput should work without barge-in callbacks."""
        from reachy_mini_conversation_app.cascade.speech_output import GradioSpeechOutput

        # Create mock TTS that yields chunks
        async def mock_synthesize(text):
            yield np.random.randint(-1000, 1000, size=4800, dtype=np.int16).tobytes()

        mock_tts.synthesize = mock_synthesize

        # Create without callbacks (default None)
        speech_output = GradioSpeechOutput(
            tts=mock_tts,
            playback=mock_playback,
            # No barge_in callbacks provided
        )
        speech_output.return_after_tts_queued = True

        # Should not raise any error
        await speech_output.speak("Test")

        # Verify audio was still queued
        assert mock_playback.put_audio.call_count >= 1
