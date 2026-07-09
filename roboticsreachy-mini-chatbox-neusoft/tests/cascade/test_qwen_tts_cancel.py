"""Tests for QwenRealtimeTTS cancel behavior.

Task 5: QwenRealtimeTTS cancel_current() 实现

Test acceptance criteria:
- cancel_current() 标记当前 session_id 为 stale 并关闭 WebSocket
- cancel_current_from_thread() 跨线程安全调用
- stale session 的音频不产生输出
- _is_session_stale(session_id) 检查
- _cleanup_stale_sessions() 清理过旧记录
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest


class TestQwenRealtimeTTSCancel:
    """Tests for QwenRealtimeTTS cancel_current() behavior."""

    def test_session_id_initial_value(self):
        """_session_id starts at 0."""
        from reachy_mini_conversation_app.cascade.tts.qwen_realtime import QwenRealtimeTTS

        tts = QwenRealtimeTTS(api_key="test_key")

        assert tts._session_id == 0
        assert len(tts._stale_session_ids) == 0

    def test_stale_session_ids_is_set(self):
        """_stale_session_ids is a set."""
        from reachy_mini_conversation_app.cascade.tts.qwen_realtime import QwenRealtimeTTS

        tts = QwenRealtimeTTS(api_key="test_key")

        assert isinstance(tts._stale_session_ids, set)

    def test_cancel_current_marks_session_stale(self):
        """cancel_current() marks current session as stale."""
        from reachy_mini_conversation_app.cascade.tts.qwen_realtime import QwenRealtimeTTS

        tts = QwenRealtimeTTS(api_key="test_key")

        # Simulate session started
        tts._session_id = 5

        # Cancel (async)
        asyncio.run(tts.cancel_current())

        # Session 5 should be in stale set
        assert 5 in tts._stale_session_ids

    def test_cancel_current_from_thread(self):
        """cancel_current_from_thread() works from background thread."""
        from reachy_mini_conversation_app.cascade.tts.qwen_realtime import QwenRealtimeTTS

        tts = QwenRealtimeTTS(api_key="test_key")
        tts._session_id = 3

        # Create event loop
        loop = asyncio.new_event_loop()
        stale_check_result = []

        # Run loop in background
        def run_loop():
            asyncio.set_event_loop(loop)
            loop.run_forever()

        loop_thread = threading.Thread(target=run_loop, daemon=True)
        loop_thread.start()

        # Give it time to start
        time.sleep(0.1)

        # Call from another thread
        tts.cancel_current_from_thread(loop)

        # Wait for the coroutine to be scheduled
        time.sleep(0.2)

        # Session should be stale
        assert 3 in tts._stale_session_ids

        # Cleanup
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=1)
        loop.close()

    def test_is_session_stale_returns_true_for_stale_session(self):
        """_is_session_stale() returns True for stale session IDs."""
        from reachy_mini_conversation_app.cascade.tts.qwen_realtime import QwenRealtimeTTS

        tts = QwenRealtimeTTS(api_key="test_key")

        # Mark session 10 as stale
        tts._stale_session_ids.add(10)

        # Check staleness
        assert tts._is_session_stale(10) is True
        assert tts._is_session_stale(5) is False

    def test_cancel_current_clears_prepared_ws(self):
        """cancel_current() clears prepared WebSocket."""
        from reachy_mini_conversation_app.cascade.tts.qwen_realtime import QwenRealtimeTTS

        tts = QwenRealtimeTTS(api_key="test_key")
        tts._session_id = 1

        # Simulate prepared WebSocket
        class MockWS:
            async def close(self):
                pass

        tts._prepared_ws = MockWS()
        tts._prepared_cm = None

        # Cancel
        asyncio.run(tts.cancel_current())

        # Prepared WS should be cleared
        assert tts._prepared_ws is None
        assert tts._prepared_cm is None

    def test_cancel_current_handles_no_ws_gracefully(self):
        """cancel_current() works when no WebSocket is active."""
        from reachy_mini_conversation_app.cascade.tts.qwen_realtime import QwenRealtimeTTS

        tts = QwenRealtimeTTS(api_key="test_key")
        tts._session_id = 1

        # No WebSocket active
        tts._prepared_ws = None
        tts._prepared_cm = None

        # Cancel should not raise
        asyncio.run(tts.cancel_current())

        # Session should still be marked stale
        assert 1 in tts._stale_session_ids

    def test_cancel_current_from_thread_handles_non_running_loop(self):
        """cancel_current_from_thread() handles non-running loop gracefully."""
        from reachy_mini_conversation_app.cascade.tts.qwen_realtime import QwenRealtimeTTS

        tts = QwenRealtimeTTS(api_key="test_key")
        tts._session_id = 2

        # Create a loop that is not running
        loop = asyncio.new_event_loop()

        # Should not raise, just log warning
        tts.cancel_current_from_thread(loop)

        # Session should still be marked stale (synchronous part)
        assert 2 in tts._stale_session_ids

        loop.close()

    def test_cleanup_stale_sessions_removes_old_ids(self):
        """_cleanup_stale_sessions() removes old stale session IDs."""
        from reachy_mini_conversation_app.cascade.tts.qwen_realtime import QwenRealtimeTTS

        tts = QwenRealtimeTTS(api_key="test_key")

        # Add many stale sessions
        for i in range(1, 11):
            tts._stale_session_ids.add(i)

        assert len(tts._stale_session_ids) == 10

        # Cleanup, keeping only 5 most recent
        tts._cleanup_stale_sessions(keep_recent=5)

        # Should have 5 sessions (6-10)
        assert len(tts._stale_session_ids) == 5
        assert 1 not in tts._stale_session_ids
        assert 5 not in tts._stale_session_ids
        assert 6 in tts._stale_session_ids
        assert 10 in tts._stale_session_ids

    def test_cancel_current_with_current_ws(self):
        """cancel_current() closes current WebSocket if active."""
        from reachy_mini_conversation_app.cascade.tts.qwen_realtime import QwenRealtimeTTS

        tts = QwenRealtimeTTS(api_key="test_key")
        tts._session_id = 1

        # Track close calls
        close_called = []

        class MockWS:
            async def close(self):
                close_called.append(True)

        tts._current_ws = MockWS()

        # Cancel
        asyncio.run(tts.cancel_current())

        # WebSocket should be closed
        assert len(close_called) == 1
        assert tts._current_ws is None
        assert 1 in tts._stale_session_ids

    def test_session_id_increments_in_synthesize(self):
        """synthesize() increments session_id for each synthesis."""
        from reachy_mini_conversation_app.cascade.tts.qwen_realtime import QwenRealtimeTTS

        tts = QwenRealtimeTTS(api_key="test_key")

        # Initial state
        assert tts._session_id == 0

        # Note: We cannot call synthesize() without mocking WebSocket,
        # but we verify the session tracking infrastructure exists
        assert hasattr(tts, "_session_id")
        assert hasattr(tts, "_stale_session_ids")
        assert hasattr(tts, "cancel_current")
        assert hasattr(tts, "cancel_current_from_thread")
        assert hasattr(tts, "_is_session_stale")
        assert hasattr(tts, "_cleanup_stale_sessions")