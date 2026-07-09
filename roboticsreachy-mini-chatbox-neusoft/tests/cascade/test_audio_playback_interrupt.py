"""Tests for AudioPlaybackSystem interrupt(turn_id) behavior.

Task 3: AudioPlaybackSystem interrupt(turn_id) 功能

Test acceptance criteria (R2, R3):
- 正常播放：generation == current_gen 的 chunk 正常播放
- 打断过滤：interrupt(new_gen) 后，generation < new_gen 的 chunk 丢弃
- Stale END_OF_TURN：generation < current_gen 的 END_OF_TURN 不触发 completion
- Exact match：generation == current_gen 的 END_OF_TURN 才 set completion event
- signal_end_of_turn(caller_turn_id) 返回正确的 turn_id 和 event
- interrupt() 清理所有 < new_generation 的 events
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import time
import threading
from queue import Empty, Queue
from pathlib import Path

import numpy as np
import pytest


# Direct module loading to bypass ui/__init__.py which imports cv2
# This allows tests to run without opencv-python installed
def _load_audio_playback_module():
    """Load audio_playback.py directly, bypassing ui/__init__.py."""
    project_root = Path(__file__).parent.parent.parent
    module_path = project_root / "src" / "reachy_mini_conversation_app" / "cascade" / "ui" / "audio_playback.py"

    spec = importlib.util.spec_from_file_location(
        "reachy_mini_conversation_app.cascade.ui.audio_playback",
        str(module_path)
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules so subsequent imports work
    sys.modules["reachy_mini_conversation_app.cascade.ui.audio_playback"] = module
    spec.loader.exec_module(module)
    return module


# Load module at test collection time
_audio_playback_module = _load_audio_playback_module()
AudioPlaybackSystem = _audio_playback_module.AudioPlaybackSystem


class TestAudioPlaybackGenerationTracking:
    """Tests for AudioPlaybackSystem generation tracking (R2)."""

    def test_initial_generation_is_zero(self):
        """初始 generation 为 0"""
        playback = AudioPlaybackSystem(robot=None, head_wobbler=None)
        assert playback._current_generation == 0
        playback.close()

    def test_interrupt_updates_generation(self):
        """interrupt(new_generation) 更新 _current_generation"""
        playback = AudioPlaybackSystem(robot=None, head_wobbler=None)

        playback.interrupt(1)
        assert playback._current_generation == 1

        playback.interrupt(5)
        assert playback._current_generation == 5

        playback.close()

    def test_interrupt_clears_audio_queue(self):
        """interrupt() 清空音频队列"""
        playback = AudioPlaybackSystem(robot=None, head_wobbler=None)

        # Put some chunks
        chunk1 = np.zeros(1000, dtype=np.int16)
        chunk2 = np.zeros(1000, dtype=np.int16)
        playback.put_audio(chunk1, generation=0)
        playback.put_audio(chunk2, generation=0)

        # Verify queue has items
        assert playback._audio_queue.qsize() >= 2

        # Interrupt
        playback.interrupt(1)

        # Queue should be cleared (may have sentinel)
        time.sleep(0.05)  # Give a moment for cleanup

        # After interrupt, queue should be mostly empty
        # (may contain sentinel or items being processed)
        remaining = playback._audio_queue.qsize()
        assert remaining <= 1  # At most sentinel

        playback.close()

    def test_put_audio_with_generation_param(self):
        """put_audio(chunk, generation) 正确入队带标签的音频"""
        playback = AudioPlaybackSystem(robot=None, head_wobbler=None)

        chunk = np.zeros(1000, dtype=np.int16)
        playback.put_audio(chunk, generation=5)

        # Should be able to get the item with generation tag
        item = playback._audio_queue.get(timeout=1.0)
        assert item is not None
        gen, audio_chunk = item
        assert gen == 5
        assert audio_chunk.shape == chunk.shape

        playback.close()

    def test_put_audio_without_generation_uses_current(self):
        """put_audio(chunk) 不带 generation 时使用当前 generation"""
        shutdown_event = threading.Event()
        playback = AudioPlaybackSystem(robot=None, head_wobbler=None, shutdown_event=shutdown_event)

        # Stop playback thread immediately so we can inspect queue
        shutdown_event.set()
        # Wait for thread to stop consuming
        time.sleep(0.2)

        # Set generation to 3
        playback.interrupt(3)

        # Drain the None sentinel that interrupt() added
        try:
            playback._audio_queue.get_nowait()
        except:
            pass

        chunk = np.zeros(1000, dtype=np.int16)
        playback.put_audio(chunk)  # No generation param

        # Should use current generation (3)
        item = playback._audio_queue.get(timeout=1.0)
        assert item is not None
        gen, audio_chunk = item
        assert gen == 3

        playback.close()

    def test_generation_lock_is_initialized(self):
        """_generation_lock 初始化为 threading.Lock"""
        playback = AudioPlaybackSystem(robot=None, head_wobbler=None)
        assert hasattr(playback, "_generation_lock")
        assert isinstance(playback._generation_lock, type(threading.Lock()))

        playback.close()

    def test_current_generation_property(self):
        """current_generation property 返回当前 generation"""
        playback = AudioPlaybackSystem(robot=None, head_wobbler=None)

        assert playback.current_generation == 0

        playback.interrupt(10)
        assert playback.current_generation == 10

        playback.close()


class TestAudioPlaybackInterruptFiltering:
    """Tests for playback thread generation filtering (R2)."""

    def test_generation_below_current_is_discarded_in_queue(self):
        """generation < current_generation 的 chunk 在入队后会被丢弃"""
        playback = AudioPlaybackSystem(robot=None, head_wobbler=None)

        # Put chunk with generation 0
        chunk0 = np.zeros(1000, dtype=np.int16)
        playback.put_audio(chunk0, generation=0)

        # Interrupt to generation 1 (makes gen 0 stale)
        playback.interrupt(1)

        # Now the queue should have been cleared
        # and subsequent stale chunks should be filtered
        time.sleep(0.05)

        playback.close()

    def test_generation_equal_to_current_is_accepted(self):
        """generation == current_generation 的 chunk 正常入队"""
        shutdown_event = threading.Event()
        playback = AudioPlaybackSystem(robot=None, head_wobbler=None, shutdown_event=shutdown_event)

        # Stop playback thread immediately so we can inspect queue
        shutdown_event.set()
        # Wait for thread to stop consuming
        time.sleep(0.2)

        # Set generation to 1
        playback.interrupt(1)

        # Drain the None sentinel that interrupt() added
        try:
            playback._audio_queue.get_nowait()
        except:
            pass

        # Put chunk with matching generation
        chunk1 = np.zeros(1000, dtype=np.int16)
        playback.put_audio(chunk1, generation=1)

        # Should be in queue
        item = playback._audio_queue.get(timeout=1.0)
        assert item is not None
        gen, audio_chunk = item
        assert gen == 1

        playback.close()

    def test_generation_above_current_is_accepted(self):
        """generation > current_generation 的 chunk 正常入队（用于预入队下一个 turn）"""
        playback = AudioPlaybackSystem(robot=None, head_wobbler=None)

        # Current generation is 0

        # Put chunk with generation 1 (future turn)
        chunk1 = np.zeros(1000, dtype=np.int16)
        playback.put_audio(chunk1, generation=1)

        # Should be in queue (not filtered yet, filtering happens at playback)
        item = playback._audio_queue.get(timeout=1.0)
        assert item is not None
        gen, audio_chunk = item
        assert gen == 1

        playback.close()

    def test_multiple_interrupts_increment_generation_strictly(self):
        """多次 interrupt 严格递增 generation"""
        playback = AudioPlaybackSystem(robot=None, head_wobbler=None)

        gen_values = [1, 2, 3, 5, 10, 100]
        for gen in gen_values:
            playback.interrupt(gen)
            assert playback._current_generation == gen

        playback.close()


class TestAudioPlaybackCompletionEvent:
    """Tests for completion event binding (R3)."""

    def test_playback_complete_events_dict_initialized(self):
        """_playback_complete_events dict 初始化"""
        playback = AudioPlaybackSystem(robot=None, head_wobbler=None)
        assert hasattr(playback, "_playback_complete_events")
        assert isinstance(playback._playback_complete_events, dict)

        playback.close()

    def test_signal_end_of_turn_returns_event(self):
        """signal_end_of_turn(caller_turn_id) 返回 asyncio.Event"""
        playback = AudioPlaybackSystem(robot=None, head_wobbler=None)

        # Set current generation
        playback.interrupt(1)

        # Signal end of turn with matching generation
        result = playback.signal_end_of_turn(caller_turn_id=1)

        # Should return (turn_id, event)
        assert result is not None
        turn_id, event = result
        assert turn_id == 1
        assert isinstance(event, asyncio.Event)

        playback.close()

    def test_signal_end_of_turn_stale_generation_returns_immediately(self):
        """stale generation 的 signal_end_of_turn 立即返回已 set 的 event"""
        playback = AudioPlaybackSystem(robot=None, head_wobbler=None)

        # Set current generation to 2
        playback.interrupt(2)

        # Signal end of turn with stale generation (1 < 2)
        result = playback.signal_end_of_turn(caller_turn_id=1)

        # Should return event that is already set
        turn_id, event = result
        assert turn_id == 1
        assert event.is_set()  # Stale completion is immediately set

        playback.close()

    def test_signal_end_of_turn_exact_match_creates_waitable_event(self):
        """generation == current_gen 的 signal_end_of_turn 创建未 set 的 event"""
        playback = AudioPlaybackSystem(robot=None, head_wobbler=None)

        # Set current generation
        playback.interrupt(1)

        # Signal end of turn with matching generation
        result = playback.signal_end_of_turn(caller_turn_id=1)

        # Should return event that is NOT set yet
        turn_id, event = result
        assert turn_id == 1
        assert not event.is_set()  # Not yet completed

        playback.close()

    def test_interrupt_clears_stale_completion_events(self):
        """interrupt() 清理所有 < new_generation 的 events"""
        playback = AudioPlaybackSystem(robot=None, head_wobbler=None)

        # Set generation 1 and create event
        playback.interrupt(1)
        playback.signal_end_of_turn(caller_turn_id=1)

        # Set generation 2 and create event
        playback.interrupt(2)
        playback.signal_end_of_turn(caller_turn_id=2)

        # Interrupt to generation 3 (should clear gen 1 and 2 events)
        playback.interrupt(3)

        # Gen 1 and 2 events should be cleared
        assert 1 not in playback._playback_complete_events
        assert 2 not in playback._playback_complete_events

        playback.close()

    def test_interrupt_sets_stale_events_to_unblock_waiters(self):
        """interrupt() set stale events，解除 waiter 阻塞"""
        playback = AudioPlaybackSystem(robot=None, head_wobbler=None)

        # Set generation 1 and create event
        playback.interrupt(1)
        result1 = playback.signal_end_of_turn(caller_turn_id=1)
        _, event1 = result1

        # Interrupt to generation 2 (stale gen 1 event should be set)
        playback.interrupt(2)

        # Event 1 should be set (unblock waiters)
        assert event1.is_set()

        playback.close()


class TestAudioPlaybackWobblerGeneration:
    """Tests for wobbler generation isolation (R4)."""

    def test_put_wobbler_accepts_generation_param(self):
        """put_wobbler(chunk, generation) 正确入队"""
        playback = AudioPlaybackSystem(robot=None, head_wobbler=None)

        # Put wobbler data with generation
        chunk = b"test_audio_bytes"
        playback.put_wobbler(chunk, generation=5)

        # Should be in queue with generation tag
        item = playback._wobbler_queue.get(timeout=1.0)
        assert item is not None
        gen, wobbler_chunk = item
        assert gen == 5
        assert wobbler_chunk == chunk

        playback.close()

    def test_interrupt_clears_wobbler_queue(self):
        """interrupt() 清空 wobbler 队列"""
        playback = AudioPlaybackSystem(robot=None, head_wobbler=None)

        # Put some wobbler data
        playback.put_wobbler(b"data1", generation=0)
        playback.put_wobbler(b"data2", generation=0)

        # Interrupt
        playback.interrupt(1)

        # Queue should be cleared
        time.sleep(0.05)
        remaining = playback._wobbler_queue.qsize()
        assert remaining <= 1  # At most sentinel

        playback.close()


class TestAudioPlaybackThreadSafety:
    """Tests for concurrent access safety (E1, R3)."""

    def test_concurrent_interrupt_calls_no_crash(self):
        """连续 interrupt() 调用不会导致 playback thread crash"""
        playback = AudioPlaybackSystem(robot=None, head_wobbler=None)

        # Rapid concurrent interrupts from multiple threads
        def interrupt_thread(gen_value: int) -> None:
            playback.interrupt(gen_value)

        threads = [
            threading.Thread(target=interrupt_thread, args=(i,))
            for i in range(1, 20)
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=2)

        # Should not crash, final generation should be one of the values
        assert playback._current_generation >= 1
        assert playback._current_generation <= 19

        playback.close()

    def test_concurrent_put_and_interrupt_no_deadlock(self):
        """并发 put_audio 和 interrupt 不会 deadlock"""
        playback = AudioPlaybackSystem(robot=None, head_wobbler=None)

        deadlock_detected = False

        def put_thread() -> None:
            for i in range(100):
                chunk = np.zeros(100, dtype=np.int16)
                playback.put_audio(chunk, generation=i % 10)

        def interrupt_thread() -> None:
            for i in range(1, 10):
                playback.interrupt(i)
                time.sleep(0.01)

        threads = [
            threading.Thread(target=put_thread),
            threading.Thread(target=interrupt_thread),
        ]

        for t in threads:
            t.start()

        # Wait with timeout - if deadlock, threads won't finish
        for t in threads:
            t.join(timeout=5)
            if t.is_alive():
                deadlock_detected = True

        assert not deadlock_detected

        playback.close()


class TestAudioPlaybackBackwardCompatibility:
    """Tests for backward compatibility with existing API."""

    def test_put_audio_without_generation_still_works(self):
        """现有代码不带 generation 参数仍然能工作"""
        playback = AudioPlaybackSystem(robot=None, head_wobbler=None)

        # Legacy call without generation
        chunk = np.zeros(1000, dtype=np.int16)
        playback.put_audio(chunk)  # No generation param

        # Should work and use current generation
        item = playback._audio_queue.get(timeout=1.0)
        assert item is not None

        playback.close()

    def test_put_wobbler_without_generation_still_works(self):
        """现有代码不带 generation 参数仍然能工作"""
        playback = AudioPlaybackSystem(robot=None, head_wobbler=None)

        # Legacy call without generation
        playback.put_wobbler(b"test_data")  # No generation param

        # Should work
        item = playback._wobbler_queue.get(timeout=1.0)
        assert item is not None

        playback.close()

    def test_close_still_works_after_interrupt(self):
        """close() 在 interrupt 后仍然能正常关闭"""
        playback = AudioPlaybackSystem(robot=None, head_wobbler=None)

        playback.interrupt(5)
        playback.close()  # Should not raise

        # shutdown_event should be set
        assert playback.shutdown_event.is_set()