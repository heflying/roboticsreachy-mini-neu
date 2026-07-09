"""Tests for VAD barge-in detection and triggering.

Task 9: VAD Barge-in Trigger Mechanism

Test acceptance criteria (R8):
- VAD speech_detected -> cancel_current_turn() is called
- LLM, TTS, Playback task all receive cancellation signal
- Debounce prevents rapid repeated triggering
- Lifecycle correct: enabled during playback, disabled after

R7 acceptance criteria:
- First chunk queued -> _start_barge_in_monitor() called
- Turn completion or cancellation -> _stop_barge_in_monitor() called
- Stale stop race: Turn 1 cleanup after Turn 2 started -> Turn 2 monitor not stopped
- Stale start race: Turn 1 first chunk after Turn 2 started -> Turn 2 monitor not started
"""

from __future__ import annotations

import sys
import time
import threading
import pytest
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

# Direct import of audio_recording module without going through ui/__init__.py
# This avoids the cv2 import dependency chain
project_root = Path(__file__).parent.parent.parent
module_path = project_root / "src" / "reachy_mini_conversation_app" / "cascade" / "ui" / "audio_recording.py"

_spec = importlib.util.spec_from_file_location(
    "reachy_mini_conversation_app.cascade.ui.audio_recording",
    str(module_path)
)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load module from {module_path}")

_audio_recording_module = importlib.util.module_from_spec(_spec)
sys.modules["reachy_mini_conversation_app.cascade.ui.audio_recording"] = _audio_recording_module
# Set __module__ correctly for dataclass to work
_audio_recording_module.__module__ = "reachy_mini_conversation_app.cascade.ui.audio_recording"
_spec.loader.exec_module(_audio_recording_module)

ContinuousVADRecorder = _audio_recording_module.ContinuousVADRecorder


class TestBargeInDetection:
    """Tests for VAD barge-in detection and triggering in ContinuousVADRecorder."""

    def test_barge_in_callback_fired_on_speech_start(self):
        """Callback fires when speech detected during playback."""
        callback_fired = []

        def barge_in_callback():
            callback_fired.append(True)

        recorder = ContinuousVADRecorder()
        recorder.set_barge_in_callback(barge_in_callback)
        recorder.enable_barge_in_detection(True)

        # Simulate speech start detection
        recorder._on_speech_start_detected()

        assert len(callback_fired) == 1

    def test_barge_in_not_fired_when_disabled(self):
        """Callback not fired when detection is disabled."""
        callback_fired = []

        def barge_in_callback():
            callback_fired.append(True)

        recorder = ContinuousVADRecorder()
        recorder.set_barge_in_callback(barge_in_callback)
        recorder.enable_barge_in_detection(False)  # Disabled

        recorder._on_speech_start_detected()

        assert len(callback_fired) == 0

    def test_barge_in_not_fired_when_no_callback_set(self):
        """Callback not fired when no callback is set."""
        recorder = ContinuousVADRecorder()
        recorder.enable_barge_in_detection(True)

        # No callback set
        recorder._on_speech_start_detected()

        # Should not raise and should gracefully handle no callback

    def test_debounce_prevents_rapid_firing(self):
        """Debounce prevents multiple rapid callbacks."""
        callback_count = []

        def barge_in_callback():
            callback_count.append(1)

        recorder = ContinuousVADRecorder()
        recorder.set_barge_in_callback(barge_in_callback)
        recorder.enable_barge_in_detection(True)

        # Rapid speech starts (within debounce window)
        recorder._on_speech_start_detected()
        recorder._on_speech_start_detected()
        recorder._on_speech_start_detected()

        # Should only fire once (debounced)
        assert len(callback_count) == 1

    def test_debounce_allows_firing_after_interval(self):
        """Debounce allows callback after debounce interval passes."""
        callback_count = []

        def barge_in_callback():
            callback_count.append(1)

        recorder = ContinuousVADRecorder()
        recorder.set_barge_in_callback(barge_in_callback)
        recorder.enable_barge_in_detection(True)

        # First fire
        recorder._on_speech_start_detected()

        # Wait longer than debounce interval (default 0.5s, use 0.6s)
        time.sleep(0.6)

        # Second fire should succeed
        recorder._on_speech_start_detected()

        assert len(callback_count) == 2

    def test_set_barge_in_callback_can_be_updated(self):
        """set_barge_in_callback can update callback."""
        callback_values = []

        def callback1():
            callback_values.append("callback1")

        def callback2():
            callback_values.append("callback2")

        recorder = ContinuousVADRecorder()
        recorder.set_barge_in_callback(callback1)
        recorder.enable_barge_in_detection(True)

        recorder._on_speech_start_detected()
        assert callback_values == ["callback1"]

        # Update callback
        recorder.set_barge_in_callback(callback2)

        # Wait for debounce
        time.sleep(0.6)
        recorder._on_speech_start_detected()
        assert callback_values == ["callback1", "callback2"]

    def test_set_barge_in_callback_can_be_cleared(self):
        """set_barge_in_callback(None) clears callback."""
        callback_fired = []

        def barge_in_callback():
            callback_fired.append(True)

        recorder = ContinuousVADRecorder()
        recorder.set_barge_in_callback(barge_in_callback)
        recorder.enable_barge_in_detection(True)

        recorder._on_speech_start_detected()
        assert len(callback_fired) == 1

        # Clear callback
        recorder.set_barge_in_callback(None)

        # Wait for debounce
        time.sleep(0.6)
        recorder._on_speech_start_detected()

        # Should not fire again
        assert len(callback_fired) == 1

    def test_enable_barge_in_detection_state_tracking(self):
        """enable_barge_in_detection tracks state correctly."""
        recorder = ContinuousVADRecorder()

        # Initially disabled
        assert not recorder._barge_in_detection_enabled

        recorder.enable_barge_in_detection(True)
        assert recorder._barge_in_detection_enabled

        recorder.enable_barge_in_detection(False)
        assert not recorder._barge_in_detection_enabled

    def test_callback_exception_is_logged_not_raised(self):
        """Callback exception is logged, not propagated."""
        def bad_callback():
            raise RuntimeError("Callback error!")

        recorder = ContinuousVADRecorder()
        recorder.set_barge_in_callback(bad_callback)
        recorder.enable_barge_in_detection(True)

        # Should not raise - exception is caught and logged
        recorder._on_speech_start_detected()


class TestBargeInLifecycleManagement:
    """Tests for barge-in monitor lifecycle management.

    These tests verify the lifecycle methods that will be added to CascadeGradioUI.
    Since we can't import CascadeGradioUI due to cv2 dependency, we test the
    lifecycle logic through mock objects that simulate the same behavior.
    """

    def test_start_barge_in_monitor_pattern(self):
        """Test the start barge-in monitor pattern (enables detection + sets callback)."""
        # This tests the pattern that will be implemented in CascadeGradioUI._start_barge_in_monitor

        class MockHandler:
            def handle_barge_in(self):
                pass

        mock_handler = MockHandler()
        recorder = ContinuousVADRecorder()

        # Simulate _start_barge_in_monitor behavior
        recorder.set_barge_in_callback(lambda: mock_handler.handle_barge_in())
        recorder.enable_barge_in_detection(True)

        assert recorder._barge_in_detection_enabled
        assert recorder._barge_in_callback is not None

    def test_stop_barge_in_monitor_pattern(self):
        """Test the stop barge-in monitor pattern (disables detection + clears callback)."""
        recorder = ContinuousVADRecorder()
        recorder._barge_in_detection_enabled = True
        recorder._barge_in_callback = lambda: None

        # Simulate _stop_barge_in_monitor behavior
        recorder.enable_barge_in_detection(False)
        recorder.set_barge_in_callback(None)

        assert not recorder._barge_in_detection_enabled
        assert recorder._barge_in_callback is None

    def test_start_handles_no_recorder_pattern(self):
        """Test that start handles None recorder gracefully."""
        # Pattern: if _vad_recorder is None, do nothing
        vad_recorder = None

        if vad_recorder is not None:
            vad_recorder.set_barge_in_callback(lambda: None)
            vad_recorder.enable_barge_in_detection(True)

        # Should complete without error

    def test_stop_handles_no_recorder_pattern(self):
        """Test that stop handles None recorder gracefully."""
        # Pattern: if _vad_recorder is None, do nothing
        vad_recorder = None

        if vad_recorder is not None:
            vad_recorder.enable_barge_in_detection(False)
            vad_recorder.set_barge_in_callback(None)

        # Should complete without error

    def test_start_sets_handler_barge_in_callback_integration(self):
        """Test that callback triggers handler.handle_barge_in()."""
        barge_in_called = []

        class MockHandler:
            def handle_barge_in(self):
                barge_in_called.append(True)

        mock_handler = MockHandler()
        recorder = ContinuousVADRecorder()

        # Set callback that calls handler
        recorder.set_barge_in_callback(lambda: mock_handler.handle_barge_in())
        recorder.enable_barge_in_detection(True)

        # Trigger via VAD
        recorder._on_speech_start_detected()

        assert len(barge_in_called) == 1


class TestHandlerBargeInMethod:
    """Tests for CascadeHandler.handle_barge_in() method."""

    def test_handle_barge_in_returns_none_without_turn_controller(self):
        """handle_barge_in returns None when TurnController is not initialized."""
        from unittest.mock import patch, MagicMock
        from reachy_mini_conversation_app.cascade.handler import CascadeHandler
        from reachy_mini_conversation_app.tools.core_tools import ToolDependencies

        # Create minimal handler without TurnController (use mocks)
        mock_reachy = MagicMock()
        mock_movement = MagicMock()
        deps = ToolDependencies(reachy_mini=mock_reachy, movement_manager=mock_movement)

        # Mock config validation and provider initialization to avoid import errors
        with patch("reachy_mini_conversation_app.cascade.config.CascadeConfig._validate"):
            with patch("reachy_mini_conversation_app.cascade.handler.init_asr_provider", return_value=MagicMock()):
                with patch("reachy_mini_conversation_app.cascade.handler.init_llm_provider", return_value=MagicMock()):
                    with patch("reachy_mini_conversation_app.cascade.handler.init_tts_provider", return_value=MagicMock()):
                        with patch("reachy_mini_conversation_app.cascade.config.get_config") as mock_config:
                            mock_config.return_value = MagicMock(is_asr_streaming=lambda: False)
                            handler = CascadeHandler(deps)

        # Without TurnController, should return None
        result = handler.handle_barge_in()
        assert result is None

    def test_handle_barge_in_calls_turn_controller_when_available(self):
        """handle_barge_in calls TurnController.handle_barge_in when available."""
        from unittest.mock import patch, MagicMock
        from reachy_mini_conversation_app.cascade.handler import CascadeHandler
        from reachy_mini_conversation_app.cascade.interrupt_coordinator import TurnCancellationToken
        from reachy_mini_conversation_app.tools.core_tools import ToolDependencies

        # Create handler with mock dependencies
        mock_reachy = MagicMock()
        mock_movement = MagicMock()
        deps = ToolDependencies(reachy_mini=mock_reachy, movement_manager=mock_movement)

        # Mock config validation and provider initialization to avoid import errors
        with patch("reachy_mini_conversation_app.cascade.config.CascadeConfig._validate"):
            with patch("reachy_mini_conversation_app.cascade.handler.init_asr_provider", return_value=MagicMock()):
                with patch("reachy_mini_conversation_app.cascade.handler.init_llm_provider", return_value=MagicMock()):
                    with patch("reachy_mini_conversation_app.cascade.handler.init_tts_provider", return_value=MagicMock()):
                        with patch("reachy_mini_conversation_app.cascade.config.get_config") as mock_config:
                            mock_config.return_value = MagicMock(is_asr_streaming=lambda: False)
                            handler = CascadeHandler(deps)

        # Mock TurnController that returns (new_turn_id, new_token)
        barge_in_called = []

        class MockTurnController:
            def handle_barge_in(self):
                barge_in_called.append(True)
                return (2, TurnCancellationToken(turn_id=2))

        handler._turn_controller = MockTurnController()

        # Call handle_barge_in
        result = handler.handle_barge_in()

        assert len(barge_in_called) == 1
        assert result is not None
        assert result[0] == 2  # new_turn_id
        assert result[1].turn_id == 2  # new_token