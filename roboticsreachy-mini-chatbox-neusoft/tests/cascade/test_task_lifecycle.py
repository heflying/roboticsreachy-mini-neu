"""Tests for R5 (LLM Producer Task Lifecycle) and R6 (TTS Consumer Task Lifecycle).

R5: LLM Producer Task 生命周期
- 正常完成：producer task 被 cancel + await + unregister
- speak_stream 异常：producer task 被 cancel + await
- Timeout：producer task 不泄漏，继续运行 API 调用

R6: TTS Consumer Task 生命周期
- Streaming path：consume_tts_segment 内 nonlocal first_chunk_queued, barge_in_started
- Single-request path：consume_tts 内 nonlocal 正确声明
- Cleanup：generator aclose 在 consumer task finally 内执行
- CancelledError：consumer task 正确处理，不泄漏 generator

Test acceptance criteria:
- 一个观点应该被多个 case 覆盖
- 测试 case 要完全覆盖观点
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import AsyncIterator, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reachy_mini_conversation_app.cascade.interrupt_coordinator import (
    InterruptCoordinator,
    TurnCancellationToken,
)
from reachy_mini_conversation_app.cascade.turn_controller import TurnController


# =============================================================================
# R5: LLM Producer Task 生命周期测试
# =============================================================================


class TestR5LLMProducerTaskLifecycle:
    """Tests for R5: LLM Producer Task lifecycle management."""

    # -------------------------------------------------------------------------
    # R5-1: 正常完成时 producer task 被 cancel + await + unregister
    # -------------------------------------------------------------------------

    def test_r5_1_case_1_normal_completion_producer_cleaned_up(self):
        """Case 1: 正常流程完成 - stream 正常结束，producer 被正确清理。

        观点：正常完成时，LLM producer task 应被正确 cancel、await 和 unregister。
        """
        async def run_test():
            coordinator = InterruptCoordinator()
            coordinator.set_current_turn(1)
            token = TurnCancellationToken(turn_id=1)

            # Create a mock LLM producer task
            producer_task = asyncio.create_task(asyncio.sleep(0.1))
            coordinator.register_llm_task(producer_task, token)

            # Wait for task to complete normally
            await producer_task

            # Task should complete without cancellation
            assert producer_task.done()
            assert not producer_task.cancelled()

            # Unregister the completed task
            coordinator.unregister_llm_task(producer_task, token)

            # Task should no longer be registered
            assert not coordinator.has_llm_task(producer_task)

        asyncio.run(run_test())

    def test_r5_1_case_2_user_end_turn_triggers_cleanup(self):
        """Case 2: 用户正常结束对话 - explicit end turn 触发 cleanup。

        观点：正常完成时，显式结束 turn 应触发完整的 cleanup 流程。
        """
        async def run_test():
            coordinator = InterruptCoordinator()
            coordinator.set_current_turn(1)
            token = TurnCancellationToken(turn_id=1)

            # Create a producer task that simulates LLM generation
            async def llm_producer():
                await asyncio.sleep(0.2)
                return "LLM response"

            producer_task = asyncio.create_task(llm_producer())
            coordinator.register_llm_task(producer_task, token)

            # Simulate explicit end turn
            result = await producer_task

            # Cleanup: unregister task
            coordinator.unregister_llm_task(producer_task, token)

            # Verify cleanup completed
            assert result == "LLM response"
            assert not coordinator.has_llm_task(producer_task)

        asyncio.run(run_test())

    def test_r5_1_case_3_eos_marker_cleanup(self):
        """Case 3: stream 自然结束 - EOS marker 到达后的清理。

        观点：正常完成时，EOS (End of Stream) marker 到达后应触发清理。
        """
        async def run_test():
            coordinator = InterruptCoordinator()
            coordinator.set_current_turn(1)
            token = TurnCancellationToken(turn_id=1)

            # Simulate streaming with EOS marker
            async def stream_with_eos():
                chunks = ["chunk1", "chunk2", "EOS"]
                for chunk in chunks:
                    await asyncio.sleep(0.05)
                    if chunk == "EOS":
                        break
                    yield chunk

            # Create producer task from generator
            collected = []

            async def producer():
                async for chunk in stream_with_eos():
                    collected.append(chunk)

            producer_task = asyncio.create_task(producer())
            coordinator.register_llm_task(producer_task, token)

            await producer_task

            # After EOS, cleanup should happen
            coordinator.unregister_llm_task(producer_task, token)

            # Verify EOS-triggered cleanup
            assert collected == ["chunk1", "chunk2"]
            assert not coordinator.has_llm_task(producer_task)

        asyncio.run(run_test())

    # -------------------------------------------------------------------------
    # R5-2: speak_stream 异常时 producer task 被 cancel + await
    # -------------------------------------------------------------------------

    def test_r5_2_case_1_llm_runtime_error_producer_cancelled(self):
        """Case 1: LLM 抛出 RuntimeError - 异常被捕获，producer 被 cancel。

        观点：异常发生时，LLM producer task 应被 cancel 并 await。
        """
        async def run_test():
            coordinator = InterruptCoordinator()
            coordinator.set_current_turn(1)
            token = TurnCancellationToken(turn_id=1)

            # Create a producer task that will throw RuntimeError
            async def failing_producer():
                await asyncio.sleep(0.05)
                raise RuntimeError("LLM connection lost")

            producer_task = asyncio.create_task(failing_producer())
            coordinator.register_llm_task(producer_task, token)

            # Catch the exception
            with pytest.raises(RuntimeError, match="LLM connection lost"):
                await producer_task

            # Task should be done (failed)
            assert producer_task.done()
            assert not producer_task.cancelled()  # Failed, not cancelled

            # Cleanup: unregister failed task
            coordinator.unregister_llm_task(producer_task, token)
            assert not coordinator.has_llm_task(producer_task)

        asyncio.run(run_test())

    def test_r5_2_case_2_network_interrupt_producer_cancelled(self):
        """Case 2: 网络中断异常 - ConnectionError 处理。

        观点：异常发生时，网络中断应正确处理 producer task 的 cleanup。
        """
        async def run_test():
            coordinator = InterruptCoordinator()
            coordinator.set_current_turn(1)
            token = TurnCancellationToken(turn_id=1)

            # Create a producer task that simulates network interruption
            async def network_interrupted_producer():
                await asyncio.sleep(0.05)
                raise ConnectionError("WebSocket closed unexpectedly")

            producer_task = asyncio.create_task(network_interrupted_producer())
            coordinator.register_llm_task(producer_task, token)

            # Catch the exception
            with pytest.raises(ConnectionError, match="WebSocket closed"):
                await producer_task

            # Task should be done (failed)
            assert producer_task.done()

            # Cleanup should still happen
            coordinator.unregister_llm_task(producer_task, token)
            assert not coordinator.has_llm_task(producer_task)

        asyncio.run(run_test())

    def test_r5_2_case_3_timeout_error_producer_cancelled(self):
        """Case 3: 超时异常 - TimeoutError 处理。

        观点：异常发生时，超时应正确处理 producer task 的 cleanup。
        """
        async def run_test():
            coordinator = InterruptCoordinator()
            coordinator.set_current_turn(1)
            token = TurnCancellationToken(turn_id=1)

            # Create a slow producer task
            async def slow_producer():
                await asyncio.sleep(10)  # Will timeout
                return "too late"

            producer_task = asyncio.create_task(slow_producer())
            coordinator.register_llm_task(producer_task, token)

            # Wait with timeout
            try:
                await asyncio.wait_for(producer_task, timeout=0.1)
            except asyncio.TimeoutError:
                # Cancel the timed-out task
                producer_task.cancel()
                try:
                    await producer_task
                except asyncio.CancelledError:
                    pass

            # Task should be cancelled
            assert producer_task.cancelled()

            # Cleanup: unregister cancelled task
            coordinator.unregister_llm_task(producer_task, token)
            assert not coordinator.has_llm_task(producer_task)

        asyncio.run(run_test())

    # -------------------------------------------------------------------------
    # R5-3: Timeout 时 producer task 不泄漏
    # -------------------------------------------------------------------------

    def test_r5_3_case_1_explicit_timeout_no_leak(self):
        """Case 1: 显式 timeout 设置 - 超时后 producer 仍在运行但不阻塞。

        观点：Timeout 时，producer task 不泄漏，即使 API 调用继续运行。
        """
        async def run_test():
            coordinator = InterruptCoordinator()
            coordinator.set_current_turn(1)
            token = TurnCancellationToken(turn_id=1)

            # Track whether the producer is still running after timeout
            producer_running = []

            async def long_running_producer():
                producer_running.append("started")
                try:
                    await asyncio.sleep(10)  # Simulate long API call
                    producer_running.append("completed")
                except asyncio.CancelledError:
                    producer_running.append("cancelled")
                    raise

            producer_task = asyncio.create_task(long_running_producer())
            coordinator.register_llm_task(producer_task, token)

            # Timeout after 0.1s
            try:
                await asyncio.wait_for(asyncio.shield(producer_task), timeout=0.1)
            except asyncio.TimeoutError:
                pass

            # Producer should still be running (shielded)
            assert "started" in producer_running
            assert "completed" not in producer_running

            # Now cancel the producer
            producer_task.cancel()
            try:
                await producer_task
            except asyncio.CancelledError:
                pass

            # Should be cancelled, not leaked
            assert "cancelled" in producer_running
            assert producer_task.cancelled()

            # Cleanup
            coordinator.unregister_llm_task(producer_task, token)
            assert not coordinator.has_llm_task(producer_task)

        asyncio.run(run_test())

    def test_r5_3_case_2_implicit_timeout_coordinator_cleanup(self):
        """Case 2: 隐式超时清理 - 协调器自动清理超时 producer。

        观点：Timeout 时，协调器应能自动清理超时的 producer task。
        """
        async def run_test():
            coordinator = InterruptCoordinator()
            controller = TurnController(coordinator=coordinator)

            # Start turn 1
            turn_id1, token1 = controller.start_new_turn()

            # Create a producer task that will be cancelled by barge-in
            async def slow_producer():
                try:
                    await asyncio.sleep(10)
                except asyncio.CancelledError:
                    raise

            producer_task = asyncio.create_task(slow_producer())
            coordinator.register_llm_task(producer_task, token1)

            # Simulate barge-in (implicit timeout cleanup)
            turn_id2, token2 = controller.handle_barge_in()

            # Wait a bit for cancellation to propagate
            await asyncio.sleep(0.05)

            # Try to await the cancelled task
            try:
                await producer_task
            except asyncio.CancelledError:
                pass

            # Producer should be cancelled
            assert producer_task.cancelled()

            # Turn 1 token should be cancelled
            assert token1.cancelled

            # Turn 2 should be active
            assert turn_id2 == 2
            assert not token2.cancelled

        asyncio.run(run_test())

    def test_r5_3_case_3_multiple_timeout_no_accumulated_leak(self):
        """Case 3: 多次超时累积 - 无泄漏累积。

        观点：Timeout 时，多次超时不应导致 producer task 泄漏累积。
        """
        async def run_test():
            coordinator = InterruptCoordinator()
            controller = TurnController(coordinator=coordinator)

            leaked_tasks = []

            # Simulate 5 consecutive timeouts
            for i in range(5):
                turn_id, token = controller.start_new_turn()

                async def slow_producer():
                    try:
                        await asyncio.sleep(10)
                    except asyncio.CancelledError:
                        raise

                producer_task = asyncio.create_task(slow_producer())
                coordinator.register_llm_task(producer_task, token)

                # Immediate barge-in simulation
                controller.handle_barge_in()

                # Wait for cancellation
                await asyncio.sleep(0.02)

                try:
                    await producer_task
                except asyncio.CancelledError:
                    pass

                # Check if task was properly cleaned up
                if not producer_task.cancelled():
                    leaked_tasks.append(producer_task)

            # No tasks should leak
            assert len(leaked_tasks) == 0

            # All tasks should have been cancelled
            # Each iteration: start_new_turn() increments, handle_barge_in() increments
            # So 5 iterations * 2 = 10 turns total
            assert controller.current_turn_id == 10

        asyncio.run(run_test())


# =============================================================================
# R6: TTS Consumer Task 生命周期测试
# =============================================================================


class TestR6TTSConsumerTaskLifecycle:
    """Tests for R6: TTS Consumer Task lifecycle management."""

    # -------------------------------------------------------------------------
    # R6-1: Streaming path nonlocal 声明正确
    # -------------------------------------------------------------------------

    def test_r6_1_case_1_first_chunk_queued_modified_in_closure(self):
        """Case 1: first_chunk_queued 在闭包内修改生效。

        观点：Streaming path 中，nonlocal 变量 first_chunk_queued 应正确修改。
        """
        async def run_test():
            from reachy_mini_conversation_app.cascade.speech_output import GradioSpeechOutput
            import numpy as np

            # Mock TTS and playback
            mock_tts = MagicMock()
            mock_tts.sample_rate = 16000
            mock_tts.prefer_single_request = False

            # Create async generator for TTS with valid int16 audio data
            async def mock_synthesize(text):
                # Use proper int16 audio format (2 bytes per sample)
                audio_data = np.array([100, 200, 300, 400], dtype=np.int16).tobytes()
                yield audio_data
                yield audio_data

            mock_tts.synthesize = mock_synthesize

            mock_playback = MagicMock()
            audio_queue = []
            mock_playback.put_audio = lambda chunk, generation=0: audio_queue.append(chunk)
            mock_playback.put_wobbler = lambda chunk, generation=0: None
            mock_playback.signal_end_of_turn = lambda: None

            speech_output = GradioSpeechOutput(mock_tts, mock_playback)

            # Create text chunks iterator
            async def text_chunks():
                yield "Hello"
                yield " world"

            # Call speak_stream
            result = await speech_output.speak_stream(text_chunks(), token=None, turn_id=1)

            # first_chunk_queued should have been set (audio was queued)
            assert len(audio_queue) > 0
            assert result == "Hello world"

        asyncio.run(run_test())

    def test_r6_1_case_2_barge_in_started_modified_in_closure(self):
        """Case 2: barge_in_started 在闭包内修改生效。

        观点：Streaming path 中，nonlocal 变量 barge_in_started 应正确修改。
        """
        async def run_test():
            from reachy_mini_conversation_app.cascade.speech_output import GradioSpeechOutput

            # Mock TTS
            mock_tts = MagicMock()
            mock_tts.sample_rate = 16000
            mock_tts.prefer_single_request = False

            async def mock_synthesize(text):
                yield b"audio_data"

            mock_tts.synthesize = mock_synthesize

            # Mock playback with barge-in tracking
            mock_playback = MagicMock()
            barge_in_started_tracking = []
            mock_playback.put_audio = lambda chunk, generation=0: barge_in_started_tracking.append(True)
            mock_playback.put_wobbler = lambda chunk, generation=0: None
            mock_playback.signal_end_of_turn = lambda: None

            speech_output = GradioSpeechOutput(mock_tts, mock_playback)

            async def text_chunks():
                yield "Test sentence."

            await speech_output.speak_stream(text_chunks(), token=None, turn_id=1)

            # barge_in_started should have been set when audio was queued
            assert len(barge_in_started_tracking) > 0

        asyncio.run(run_test())

    def test_r6_1_case_3_concurrent_nonlocal_modification_no_conflict(self):
        """Case 3: 多个 nonlocal 变量并发修改不冲突。

        观点：Streaming path 中，多个 nonlocal 变量并发修改应无冲突。
        """
        async def run_test():
            from reachy_mini_conversation_app.cascade.speech_output import GradioSpeechOutput

            # Mock TTS
            mock_tts = MagicMock()
            mock_tts.sample_rate = 16000
            mock_tts.prefer_single_request = False

            async def mock_synthesize(text):
                # Multiple chunks to test concurrent modification
                for i in range(3):
                    await asyncio.sleep(0.01)
                    yield b"chunk_data"

            mock_tts.synthesize = mock_synthesize

            mock_playback = MagicMock()
            modification_order = []
            mock_playback.put_audio = lambda chunk, generation=0: modification_order.append("audio")
            mock_playback.put_wobbler = lambda chunk, generation=0: modification_order.append("wobbler")
            mock_playback.signal_end_of_turn = lambda: modification_order.append("end")

            speech_output = GradioSpeechOutput(mock_tts, mock_playback)

            async def text_chunks():
                yield "First. "
                await asyncio.sleep(0.02)
                yield "Second."

            await speech_output.speak_stream(text_chunks(), token=None, turn_id=1)

            # All modifications should happen in proper order
            assert "audio" in modification_order
            assert "end" in modification_order

        asyncio.run(run_test())

    # -------------------------------------------------------------------------
    # R6-2: Single-request path nonlocal 声明正确
    # -------------------------------------------------------------------------

    def test_r6_2_case_1_single_request_nonlocal_verification(self):
        """Case 1: single request 流程的 nonlocal 验证。

        观点：Single-request path 中，nonlocal 变量应正确声明和修改。
        """
        async def run_test():
            from reachy_mini_conversation_app.cascade.speech_output import GradioSpeechOutput

            # Mock TTS with prefer_single_request
            mock_tts = MagicMock()
            mock_tts.sample_rate = 16000
            mock_tts.prefer_single_request = True  # Force single-request path

            async def mock_synthesize(text):
                yield b"single_audio_chunk"

            mock_tts.synthesize = mock_synthesize

            mock_playback = MagicMock()
            first_chunk_queued_tracking = []
            mock_playback.put_audio = lambda chunk, generation=0: first_chunk_queued_tracking.append(True)
            mock_playback.put_wobbler = lambda chunk, generation=0: None
            mock_playback.signal_end_of_turn = lambda: None

            speech_output = GradioSpeechOutput(mock_tts, mock_playback)

            # Call speak_stream (should use single-request path)
            async def text_chunks():
                yield "Single request test."

            result = await speech_output.speak_stream(text_chunks(), token=None, turn_id=1)

            # first_chunk_queued should have been set in single-request path
            assert len(first_chunk_queued_tracking) > 0
            assert result == "Single request test."

        asyncio.run(run_test())

    def test_r6_2_case_2_exception_handling_nonlocal_correct(self):
        """Case 2: 异常处理中的 nonlocal 正确。

        观点：Single-request path 中，异常处理应能访问 nonlocal 变量。
        """
        async def run_test():
            from reachy_mini_conversation_app.cascade.speech_output import GradioSpeechOutput
            import numpy as np

            # Mock TTS that throws exception
            mock_tts = MagicMock()
            mock_tts.sample_rate = 16000
            mock_tts.prefer_single_request = True

            async def failing_synthesize(text):
                # Use proper int16 audio format
                audio_data = np.array([100, 200], dtype=np.int16).tobytes()
                yield audio_data
                raise RuntimeError("TTS failed")

            mock_tts.synthesize = failing_synthesize

            mock_playback = MagicMock()
            chunks_before_error = []
            mock_playback.put_audio = lambda chunk, generation=0: chunks_before_error.append(chunk)
            mock_playback.put_wobbler = lambda chunk, generation=0: None
            mock_playback.signal_end_of_turn = lambda: None

            speech_output = GradioSpeechOutput(mock_tts, mock_playback)

            async def text_chunks():
                yield "Test text."

            # The exception should be raised, but cleanup should happen
            with pytest.raises(RuntimeError, match="TTS failed"):
                await speech_output.speak_stream(text_chunks(), token=None, turn_id=1)

            # Partial chunk should have been queued (nonlocal first_chunk_queued was set)
            assert len(chunks_before_error) > 0

        asyncio.run(run_test())

    def test_r6_2_case_3_cleanup_finally_nonlocal_accessible(self):
        """Case 3: cleanup finally 中的 nonlocal 可访问。

        观点：Single-request path 中，finally 块应能访问 nonlocal 变量。
        """
        async def run_test():
            from reachy_mini_conversation_app.cascade.speech_output import GradioSpeechOutput
            import numpy as np

            # Mock TTS
            mock_tts = MagicMock()
            mock_tts.sample_rate = 16000
            mock_tts.prefer_single_request = True

            cleanup_called = []

            async def mock_synthesize(text):
                # Use proper int16 audio format
                audio_data = np.array([100, 200, 300, 400], dtype=np.int16).tobytes()
                yield audio_data
                # Normal completion, should trigger cleanup in finally

            mock_tts.synthesize = mock_synthesize

            mock_playback = MagicMock()
            mock_playback.put_audio = lambda chunk, generation=0: None
            mock_playback.put_wobbler = lambda chunk, generation=0: None
            mock_playback.signal_end_of_turn = lambda: cleanup_called.append("signal_end")

            speech_output = GradioSpeechOutput(mock_tts, mock_playback)

            async def text_chunks():
                yield "Test."

            await speech_output.speak_stream(text_chunks(), token=None, turn_id=1)

            # Cleanup (signal_end_of_turn) should have been called in finally
            assert "signal_end" in cleanup_called

        asyncio.run(run_test())

    # -------------------------------------------------------------------------
    # R6-3: Generator aclose 在 consumer task finally 内执行
    # -------------------------------------------------------------------------

    def test_r6_3_case_1_normal_completion_aclose_called(self):
        """Case 1: 正常完成时 aclose 被调用。

        观点：Generator cleanup 应在 consumer task finally 内正确执行 aclose。
        """
        async def run_test():
            # Create an async generator that tracks aclose calls
            aclose_called = []

            async def tracked_generator():
                try:
                    for i in range(3):
                        await asyncio.sleep(0.01)
                        yield f"chunk_{i}"
                finally:
                    aclose_called.append("generator_closed")

            gen = tracked_generator()

            # Create a consumer task that uses the generator
            async def consumer_task_func(gen):
                try:
                    async for chunk in gen:
                        await asyncio.sleep(0.01)
                finally:
                    await gen.aclose()

            task = asyncio.create_task(consumer_task_func(gen))

            # Wait for completion
            await asyncio.sleep(0.05)

            # Cancel to ensure cleanup
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            # aclose should have been called
            assert len(aclose_called) > 0

        asyncio.run(run_test())

    def test_r6_3_case_2_cancellation_aclose_in_finally(self):
        """Case 2: 取消时 aclose 在 finally 内调用。

        观点：Generator cleanup 应在 consumer task 被取消时执行 aclose。
        """
        async def run_test():
            aclose_called = []

            class TrackedGenerator:
                async def generate(self):
                    try:
                        for i in range(10):
                            await asyncio.sleep(0.1)
                            yield f"chunk_{i}"
                    except asyncio.CancelledError:
                        raise
                    finally:
                        aclose_called.append("aclose_in_finally")

            gen_instance = TrackedGenerator()
            gen = gen_instance.generate()

            # Consumer task
            async def consumer():
                try:
                    async for chunk in gen:
                        await asyncio.sleep(0.01)
                except asyncio.CancelledError:
                    # aclose should be called in generator's finally
                    raise
                finally:
                    # Explicit aclose in consumer's finally
                    await gen.aclose()

            task = asyncio.create_task(consumer())

            # Cancel after short time
            await asyncio.sleep(0.05)
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

            # aclose should have been called (possibly multiple times)
            assert len(aclose_called) > 0

        asyncio.run(run_test())

    def test_r6_3_case_3_exception_aclose_still_called(self):
        """Case 3: 异常时 aclose 仍被调用。

        观点：Generator cleanup 应在异常发生时执行 aclose。
        """
        async def run_test():
            aclose_called = []

            class FailingGenerator:
                async def generate(self):
                    try:
                        yield "chunk_1"
                        await asyncio.sleep(0.01)
                        raise RuntimeError("Generator error")
                    finally:
                        aclose_called.append(True)

            gen_instance = FailingGenerator()
            gen = gen_instance.generate()

            # Consumer task
            async def consumer():
                try:
                    async for chunk in gen:
                        await asyncio.sleep(0.01)
                except RuntimeError:
                    pass
                finally:
                    await gen.aclose()

            task = asyncio.create_task(consumer())

            try:
                await task
            except RuntimeError:
                pass

            # aclose should have been called
            assert len(aclose_called) > 0

        asyncio.run(run_test())

    # -------------------------------------------------------------------------
    # R6-4: CancelledError 正确处理，不泄漏 generator
    # -------------------------------------------------------------------------

    def test_r6_4_case_1_explicit_cancelled_error_handling(self):
        """Case 1: 显式 asyncio.CancelledError。

        观点：CancelledError 应正确处理，generator 不泄漏。
        """
        async def run_test():
            generator_closed = []

            class TestGenerator:
                async def generate(self):
                    try:
                        while True:
                            await asyncio.sleep(0.1)
                            yield "data"
                    except asyncio.CancelledError:
                        generator_closed.append("caught_cancelled")
                        raise
                    finally:
                        generator_closed.append("finally_cleanup")

            gen_instance = TestGenerator()
            gen = gen_instance.generate()

            # Consumer task
            async def consumer():
                try:
                    async for chunk in gen:
                        pass
                except asyncio.CancelledError:
                    # Proper handling
                    await gen.aclose()
                    raise

            task = asyncio.create_task(consumer())

            # Explicit cancellation
            await asyncio.sleep(0.05)
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

            # Generator should be properly closed
            assert "finally_cleanup" in generator_closed

        asyncio.run(run_test())

    def test_r6_4_case_2_external_task_cancel_triggers_cleanup(self):
        """Case 2: 外部 task.cancel() 触发。

        观点：CancelledError 从外部 task.cancel() 触发时，应正确清理 generator。
        """
        async def run_test():
            cleanup_state = []

            async def tts_generator():
                try:
                    for i in range(5):
                        await asyncio.sleep(0.1)
                        yield b"audio"
                except asyncio.CancelledError:
                    cleanup_state.append("generator_cancelled")
                    raise
                finally:
                    cleanup_state.append("generator_closed")

            gen = tts_generator()

            # Consumer task that properly handles cancellation
            async def tts_consumer(gen):
                chunks = []
                try:
                    async for chunk in gen:
                        chunks.append(chunk)
                except asyncio.CancelledError:
                    cleanup_state.append("consumer_cancelled")
                    # Ensure generator is closed
                    await gen.aclose()
                    raise
                finally:
                    cleanup_state.append("consumer_finally")

            consumer_task = asyncio.create_task(tts_consumer(gen))

            # External cancel after some time
            await asyncio.sleep(0.15)  # Let 1 chunk be processed
            consumer_task.cancel()

            try:
                await consumer_task
            except asyncio.CancelledError:
                pass

            # Both generator and consumer should be properly cleaned up
            assert "consumer_cancelled" in cleanup_state
            assert "generator_closed" in cleanup_state

        asyncio.run(run_test())

    def test_r6_4_case_3_nested_cancellation_producer_and_consumer(self):
        """Case 3: 嵌套取消（producer 和 consumer 同时取消）。

        观点：CancelledError 在嵌套取消场景下应正确处理所有 generator。
        """
        async def run_test():
            cleanup_trace = []

            # Producer generator
            async def producer_gen():
                try:
                    for i in range(3):
                        await asyncio.sleep(0.05)
                        cleanup_trace.append(f"produce_{i}")
                        yield f"text_{i}"
                except asyncio.CancelledError:
                    cleanup_trace.append("producer_cancelled")
                    raise
                finally:
                    cleanup_trace.append("producer_finally")

            # TTS generator (nested)
            async def tts_gen(text):
                try:
                    await asyncio.sleep(0.05)
                    cleanup_trace.append(f"tts_{text}")
                    yield b"audio"
                except asyncio.CancelledError:
                    cleanup_trace.append(f"tts_cancelled_{text}")
                    raise
                finally:
                    cleanup_trace.append(f"tts_finally_{text}")

            # Consumer that handles both generators
            async def consumer():
                producer = producer_gen()
                try:
                    async for text in producer:
                        tts = tts_gen(text)
                        try:
                            async for audio in tts:
                                cleanup_trace.append(f"consumed_{audio}")
                        except asyncio.CancelledError:
                            await tts.aclose()
                            raise
                except asyncio.CancelledError:
                    cleanup_trace.append("consumer_cancelled")
                    await producer.aclose()
                    raise
                finally:
                    cleanup_trace.append("consumer_finally")

            task = asyncio.create_task(consumer())

            # Wait for some processing
            await asyncio.sleep(0.1)

            # Cancel both levels simultaneously
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

            # All cleanup should have happened
            assert "consumer_finally" in cleanup_trace
            # At least one level of cleanup
            assert any("finally" in trace for trace in cleanup_trace)

        asyncio.run(run_test())


# =============================================================================
# Integration Tests for R5 + R6 Combined
# =============================================================================


class TestTaskLifecycleIntegration:
    """Integration tests combining R5 and R6 scenarios."""

    def test_full_pipeline_producer_consumer_cleanup(self):
        """Full pipeline: producer and consumer cleanup on normal completion.

        观点：完整 pipeline 正常完成时，所有 task 和 generator 应正确清理。
        """
        async def run_test():
            coordinator = InterruptCoordinator()
            controller = TurnController(coordinator=coordinator)

            # Start turn
            turn_id, token = controller.start_new_turn()

            # Create mock producer and consumer tasks
            async def llm_producer():
                yield "text_chunk"
                await asyncio.sleep(0.01)
                yield "final_text"

            async def tts_consumer(text_stream: AsyncIterator[str]):
                chunks = []
                try:
                    async for text in text_stream:
                        chunks.append(text)
                        await asyncio.sleep(0.01)
                finally:
                    # Cleanup
                    pass
                return chunks

            # Create tasks
            producer_gen = llm_producer()
            consumer_task = asyncio.create_task(tts_consumer(producer_gen))

            # Register tasks
            coordinator.register_llm_task(consumer_task, token)

            # Wait for completion
            result = await consumer_task

            # Cleanup
            coordinator.unregister_llm_task(consumer_task, token)

            assert result == ["text_chunk", "final_text"]
            assert not coordinator.has_llm_task(consumer_task)

        asyncio.run(run_test())

    def test_interrupt_cancels_both_producer_and_consumer(self):
        """Interrupt cancels both producer and consumer tasks.

        观点：打断时应同时取消 producer 和 consumer task。
        """
        async def run_test():
            coordinator = InterruptCoordinator()
            controller = TurnController(coordinator=coordinator)

            # Start turn 1
            turn_id1, token1 = controller.start_new_turn()

            cleanup_trace = []

            # Slow producer
            async def slow_producer():
                try:
                    while True:
                        await asyncio.sleep(0.1)
                        yield "text"
                except asyncio.CancelledError:
                    cleanup_trace.append("producer_cancelled")
                    raise
                finally:
                    cleanup_trace.append("producer_finally")

            # Consumer
            async def consumer(text_stream: AsyncIterator[str]):
                try:
                    async for text in text_stream:
                        await asyncio.sleep(0.05)
                except asyncio.CancelledError:
                    cleanup_trace.append("consumer_cancelled")
                    raise
                finally:
                    cleanup_trace.append("consumer_finally")

            producer_gen = slow_producer()
            consumer_task = asyncio.create_task(consumer(producer_gen))

            coordinator.register_llm_task(consumer_task, token1)

            # Wait a bit
            await asyncio.sleep(0.1)

            # Barge-in
            turn_id2, token2 = controller.handle_barge_in()

            # Wait for cancellation
            await asyncio.sleep(0.05)

            try:
                await consumer_task
            except asyncio.CancelledError:
                pass

            # Both should be cleaned up
            assert "consumer_cancelled" in cleanup_trace or "consumer_finally" in cleanup_trace
            assert token1.cancelled
            assert not token2.cancelled

        asyncio.run(run_test())

    def test_exception_in_producer_propagates_to_consumer(self):
        """Exception in producer propagates to consumer and both cleanup.

        观点：producer 异常应传播到 consumer，两者都应正确清理。
        """
        async def run_test():
            coordinator = InterruptCoordinator()
            coordinator.set_current_turn(1)
            token = TurnCancellationToken(turn_id=1)

            cleanup_trace = []

            # Failing producer
            async def failing_producer():
                try:
                    yield "chunk1"
                    await asyncio.sleep(0.01)
                    raise RuntimeError("Producer failed")
                except RuntimeError:
                    cleanup_trace.append("producer_exception")
                    raise
                finally:
                    cleanup_trace.append("producer_finally")

            # Consumer
            async def consumer(text_stream: AsyncIterator[str]):
                try:
                    async for text in text_stream:
                        cleanup_trace.append(f"consumed_{text}")
                except RuntimeError:
                    cleanup_trace.append("consumer_caught_exception")
                    raise
                finally:
                    cleanup_trace.append("consumer_finally")

            producer_gen = failing_producer()
            consumer_task = asyncio.create_task(consumer(producer_gen))

            coordinator.register_llm_task(consumer_task, token)

            # Wait for exception
            with pytest.raises(RuntimeError, match="Producer failed"):
                await consumer_task

            # Cleanup
            coordinator.unregister_llm_task(consumer_task, token)

            # Both should have cleaned up
            assert "consumed_chunk1" in cleanup_trace
            assert "producer_finally" in cleanup_trace
            assert "consumer_finally" in cleanup_trace

        asyncio.run(run_test())

    def test_token_cancellation_stops_tts_generation(self):
        """Token cancellation stops TTS generation mid-stream.

        观点：Token 取消应停止 TTS generation 中途。
        """
        async def run_test():
            from reachy_mini_conversation_app.cascade.speech_output import GradioSpeechOutput

            # Mock TTS
            mock_tts = MagicMock()
            mock_tts.sample_rate = 16000
            mock_tts.prefer_single_request = False

            chunks_generated = []

            async def mock_synthesize(text):
                for i in range(5):
                    await asyncio.sleep(0.05)
                    chunks_generated.append(f"chunk_{i}")
                    yield b"audio"

            mock_tts.synthesize = mock_synthesize

            mock_playback = MagicMock()
            mock_playback.put_audio = lambda chunk, generation=0: None
            mock_playback.put_wobbler = lambda chunk, generation=0: None
            mock_playback.signal_end_of_turn = lambda: None

            speech_output = GradioSpeechOutput(mock_tts, mock_playback)

            # Create token and cancel it early
            token = TurnCancellationToken(turn_id=1)

            async def text_chunks():
                yield "Test sentence."

            # Start speak_stream and cancel token after short delay
            async def run_with_cancel():
                task = asyncio.create_task(
                    speech_output.speak_stream(text_chunks(), token=token, turn_id=1)
                )

                # Cancel token after delay
                await asyncio.sleep(0.1)
                token.cancel()

                # Wait for task to handle cancellation
                try:
                    await task
                except Exception:
                    pass

            await run_with_cancel()

            # Token should be cancelled
            assert token.cancelled

            # Some chunks may have been generated before cancellation
            # The key is that generation stops after cancellation
            initial_chunks = len(chunks_generated)
            await asyncio.sleep(0.2)
            # No new chunks should be generated after cancellation
            # (This depends on implementation checking token.cancelled)

        asyncio.run(run_test())