"""Tests for TurnCancellationToken and InterruptCoordinator.

Task 1: TurnCancellationToken 基础实现

Test acceptance criteria:
- Token 初始状态为 cancelled=False
- cancel() 调用后 cancelled=True 且永久保持
- 多个 turn 的 token 对象不同，取消互不影响
- 已取消的 token 被新 turn token 取代后，旧 token 保持 cancelled
"""

from __future__ import annotations

import asyncio
import pytest

from reachy_mini_conversation_app.cascade.interrupt_coordinator import (
    InterruptCoordinator,
    TurnCancellationToken,
)


class TestTurnCancellationToken:
    """Tests for TurnCancellationToken basic behavior."""

    def test_token_initial_state_is_not_cancelled(self):
        """Token 初始状态为 cancelled=False"""
        token = TurnCancellationToken(turn_id=1)
        assert token.cancelled is False
        assert token.turn_id == 1

    def test_cancel_sets_cancelled_true(self):
        """cancel() 调用后 cancelled=True"""
        token = TurnCancellationToken(turn_id=1)
        token.cancel()
        assert token.cancelled is True

    def test_cancelled_state_is_sticky(self):
        """取消状态是 sticky 的——一旦取消，永久保持"""
        token = TurnCancellationToken(turn_id=1)
        token.cancel()
        assert token.cancelled is True

        # 多次调用 cancel 不改变状态，但也不会出错
        token.cancel()
        token.cancel()
        assert token.cancelled is True

    def test_different_turn_tokens_are_independent_objects(self):
        """多个 turn 的 token 对象不同"""
        token1 = TurnCancellationToken(turn_id=1)
        token2 = TurnCancellationToken(turn_id=2)

        # 不同对象
        assert token1 is not token2

        # 不同 turn_id
        assert token1.turn_id == 1
        assert token2.turn_id == 2

    def test_cancellation_does_not_affect_other_turns(self):
        """取消互不影响"""
        token1 = TurnCancellationToken(turn_id=1)
        token2 = TurnCancellationToken(turn_id=2)

        # 取消 token1
        token1.cancel()

        # token1 已取消
        assert token1.cancelled is True

        # token2 未受影响
        assert token2.cancelled is False

    def test_old_token_stays_cancelled_after_replacement(self):
        """已取消的 token 被新 turn token 取代后，旧 token 保持 cancelled"""
        # Turn 1 token
        token1 = TurnCancellationToken(turn_id=1)
        token1.cancel()

        # Turn 2 token (replaces Turn 1)
        token2 = TurnCancellationToken(turn_id=2)

        # Turn 1 token 保持 cancelled
        assert token1.cancelled is True

        # Turn 2 token 未受影响
        assert token2.cancelled is False

    def test_token_turn_id_is_immutable(self):
        """turn_id 是固定的，不能被修改"""
        token = TurnCancellationToken(turn_id=1)

        # turn_id 应该是只读属性
        with pytest.raises(AttributeError):
            token.turn_id = 2  # type: ignore[misc]

    def test_token_has_clear_repr(self):
        """Token 有清晰的 repr 用于调试"""
        token = TurnCancellationToken(turn_id=1)
        assert "turn_id=1" in repr(token)
        assert "cancelled=False" in repr(token)

        token.cancel()
        assert "cancelled=True" in repr(token)

    def test_cancelled_property_is_readonly(self):
        """cancelled 属性是只读的，不能直接设置"""
        token = TurnCancellationToken(turn_id=1)

        # cancelled 应该是只读属性
        with pytest.raises(AttributeError):
            token.cancelled = True  # type: ignore[misc]

    @pytest.mark.asyncio
    async def test_token_can_be_used_in_async_context(self):
        """Token 可以在异步上下文中使用"""
        token = TurnCancellationToken(turn_id=1)

        # 模拟异步任务检查取消状态
        await asyncio.sleep(0.01)
        assert token.cancelled is False

        # 取消后再检查
        token.cancel()
        await asyncio.sleep(0.01)
        assert token.cancelled is True

    def test_high_turn_ids_are_supported(self):
        """支持高 turn_id 值（压力测试）"""
        # 创建多个 turn token
        tokens = [TurnCancellationToken(turn_id=i) for i in range(1000)]

        # 取消一些
        for i in [10, 100, 500]:
            tokens[i].cancel()

        # 验证状态
        assert tokens[10].cancelled is True
        assert tokens[100].cancelled is True
        assert tokens[500].cancelled is True
        assert tokens[0].cancelled is False
        assert tokens[999].cancelled is False


class TestInterruptCoordinator:
    """Tests for InterruptCoordinator task registration and ownership.

    Task 2: InterruptCoordinator 实现
    Test acceptance criteria (R9):
    - register/unregister APIs require token ownership validation
    - Stale task unregister cannot clear new turn's registrations
    - Identity check: unregister must match exact registered object
    - cancel_all_for_turn propagates cancellation to all registered tasks
    """

    def test_coordinator_initial_state(self):
        """Coordinator 初始状态没有注册的 task"""
        coordinator = InterruptCoordinator()
        assert coordinator.current_turn_id == 0

    @pytest.mark.asyncio
    async def test_register_llm_task_with_valid_token(self):
        """register_llm_task(task, token) 成功当 token.turn_id == current_turn_id"""
        coordinator = InterruptCoordinator()
        coordinator.set_current_turn(1)

        token = TurnCancellationToken(turn_id=1)
        task = asyncio.create_task(asyncio.sleep(1))  # dummy task

        # Should succeed
        coordinator.register_llm_task(task, token)

        # Task should be registered
        assert coordinator.has_llm_task(task)

        # Cleanup
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_register_llm_task_with_stale_token_rejected(self):
        """register_llm_task(task, token) 失败当 token.turn_id != current_turn_id"""
        coordinator = InterruptCoordinator()
        coordinator.set_current_turn(2)  # Current turn is 2

        token = TurnCancellationToken(turn_id=1)  # Stale token (turn 1)
        task = asyncio.create_task(asyncio.sleep(1))

        # Should reject registration
        with pytest.raises(ValueError, match="ownership"):
            coordinator.register_llm_task(task, token)

        # Task should NOT be registered
        assert not coordinator.has_llm_task(task)

        # Cleanup
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_unregister_llm_task_with_matching_identity_and_ownership(self):
        """unregister_llm_task(task, token) 成功当 identity 和 ownership 都匹配"""
        coordinator = InterruptCoordinator()
        coordinator.set_current_turn(1)

        token = TurnCancellationToken(turn_id=1)
        task = asyncio.create_task(asyncio.sleep(1))
        coordinator.register_llm_task(task, token)

        # Should successfully unregister
        coordinator.unregister_llm_task(task, token)

        # Task should no longer be registered
        assert not coordinator.has_llm_task(task)

        # Cleanup
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_unregister_llm_task_with_wrong_identity_fails(self):
        """unregister 必须匹配注册时的 task 对象（不能 unregister 其他 task）"""
        coordinator = InterruptCoordinator()
        coordinator.set_current_turn(1)

        token = TurnCancellationToken(turn_id=1)
        task1 = asyncio.create_task(asyncio.sleep(1))
        task2 = asyncio.create_task(asyncio.sleep(1))  # Different task

        coordinator.register_llm_task(task1, token)

        # Trying to unregister task2 (which was never registered) should fail
        with pytest.raises(KeyError, match="not registered"):
            coordinator.unregister_llm_task(task2, token)

        # task1 should still be registered
        assert coordinator.has_llm_task(task1)

        # Cleanup
        for t in [task1, task2]:
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_stale_unregister_does_not_clear_new_turn_registration(self):
        """Stale unregister: Turn 1 task 在 Turn 2 已注册后 unregister，Turn 2 task 仍可被找到"""
        coordinator = InterruptCoordinator()

        # Turn 1: Register task1
        coordinator.set_current_turn(1)
        token1 = TurnCancellationToken(turn_id=1)
        task1 = asyncio.create_task(asyncio.sleep(1))
        coordinator.register_llm_task(task1, token1)

        # Turn 2: Replace current turn, register task2
        coordinator.set_current_turn(2)
        token2 = TurnCancellationToken(turn_id=2)
        task2 = asyncio.create_task(asyncio.sleep(1))
        coordinator.register_llm_task(task2, token2)

        # Stale Turn 1 tries to unregister (with stale token)
        # This should NOT clear Turn 2's registration
        coordinator.unregister_llm_task(task1, token1)

        # Turn 2 task should still be registered
        assert coordinator.has_llm_task(task2)

        # Turn 1 task should be removed (it was registered with turn 1)
        assert not coordinator.has_llm_task(task1)

        # Cleanup
        for t in [task1, task2]:
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_unregister_with_stale_token_fails(self):
        """unregister_llm_task 用 stale token（turn_id != 注册时 turn_id）应该失败"""
        coordinator = InterruptCoordinator()

        # Turn 1: Register task
        coordinator.set_current_turn(1)
        token1 = TurnCancellationToken(turn_id=1)
        task = asyncio.create_task(asyncio.sleep(1))
        coordinator.register_llm_task(task, token1)

        # Turn 2: Update current turn
        coordinator.set_current_turn(2)

        # Try to unregister with Turn 2's token (stale for this task)
        token2 = TurnCancellationToken(turn_id=2)
        with pytest.raises(ValueError, match="ownership"):
            coordinator.unregister_llm_task(task, token2)

        # Task should still be registered (unregister failed)
        assert coordinator.has_llm_task(task)

        # Cleanup
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_register_tts_consumer_task_with_valid_token(self):
        """register_tts_consumer_task(task, token) 需要 ownership 验证"""
        coordinator = InterruptCoordinator()
        coordinator.set_current_turn(1)

        token = TurnCancellationToken(turn_id=1)
        task = asyncio.create_task(asyncio.sleep(1))

        coordinator.register_tts_consumer_task(task, token)
        assert coordinator.has_tts_consumer_task(task)

        # Cleanup
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_register_tts_consumer_task_with_stale_token_rejected(self):
        """register_tts_consumer_task 用 stale token 应该被拒绝"""
        coordinator = InterruptCoordinator()
        coordinator.set_current_turn(2)

        token = TurnCancellationToken(turn_id=1)  # Stale
        task = asyncio.create_task(asyncio.sleep(1))

        with pytest.raises(ValueError, match="ownership"):
            coordinator.register_tts_consumer_task(task, token)

        assert not coordinator.has_tts_consumer_task(task)

        # Cleanup
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_unregister_tts_consumer_task_with_matching_identity_and_ownership(self):
        """unregister_tts_consumer_task 需要 identity 和 ownership 验证"""
        coordinator = InterruptCoordinator()
        coordinator.set_current_turn(1)

        token = TurnCancellationToken(turn_id=1)
        task = asyncio.create_task(asyncio.sleep(1))
        coordinator.register_tts_consumer_task(task, token)

        coordinator.unregister_tts_consumer_task(task, token)
        assert not coordinator.has_tts_consumer_task(task)

        # Cleanup
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_unregister_tts_consumer_task_with_wrong_identity_fails(self):
        """unregister_tts_consumer_task 必须匹配注册时的 task 对象"""
        coordinator = InterruptCoordinator()
        coordinator.set_current_turn(1)

        token = TurnCancellationToken(turn_id=1)
        task1 = asyncio.create_task(asyncio.sleep(1))
        task2 = asyncio.create_task(asyncio.sleep(1))

        coordinator.register_tts_consumer_task(task1, token)

        with pytest.raises(KeyError, match="not registered"):
            coordinator.unregister_tts_consumer_task(task2, token)

        assert coordinator.has_tts_consumer_task(task1)

        # Cleanup
        for t in [task1, task2]:
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass

    def test_register_tts_generator_with_valid_token(self):
        """register_tts_generator(gen, token) 需要 ownership 验证"""
        coordinator = InterruptCoordinator()
        coordinator.set_current_turn(1)

        token = TurnCancellationToken(turn_id=1)

        # Create a simple async generator
        async def dummy_gen():
            yield b"audio"
            yield b"more"

        gen = dummy_gen()

        coordinator.register_tts_generator(gen, token)
        assert coordinator.has_tts_generator(gen)

        # Cleanup
        asyncio.run(gen.aclose())

    def test_register_tts_generator_with_stale_token_rejected(self):
        """register_tts_generator 用 stale token 应该被拒绝"""
        coordinator = InterruptCoordinator()
        coordinator.set_current_turn(2)

        token = TurnCancellationToken(turn_id=1)  # Stale

        async def dummy_gen():
            yield b"audio"

        gen = dummy_gen()

        with pytest.raises(ValueError, match="ownership"):
            coordinator.register_tts_generator(gen, token)

        assert not coordinator.has_tts_generator(gen)

        # Cleanup
        asyncio.run(gen.aclose())

    def test_unregister_tts_generator_with_matching_identity_and_ownership(self):
        """unregister_tts_generator 需要 identity 和 ownership 验证"""
        coordinator = InterruptCoordinator()
        coordinator.set_current_turn(1)

        token = TurnCancellationToken(turn_id=1)

        async def dummy_gen():
            yield b"audio"

        gen = dummy_gen()
        coordinator.register_tts_generator(gen, token)

        coordinator.unregister_tts_generator(gen, token)
        assert not coordinator.has_tts_generator(gen)

        # Cleanup
        asyncio.run(gen.aclose())

    def test_unregister_tts_generator_with_wrong_identity_fails(self):
        """unregister_tts_generator 必须匹配注册时的 gen 对象"""
        coordinator = InterruptCoordinator()
        coordinator.set_current_turn(1)

        token = TurnCancellationToken(turn_id=1)

        async def dummy_gen():
            yield b"audio"

        gen1 = dummy_gen()
        gen2 = dummy_gen()  # Different generator

        coordinator.register_tts_generator(gen1, token)

        with pytest.raises(KeyError, match="not registered"):
            coordinator.unregister_tts_generator(gen2, token)

        assert coordinator.has_tts_generator(gen1)

        # Cleanup
        asyncio.run(gen1.aclose())
        asyncio.run(gen2.aclose())

    @pytest.mark.asyncio
    async def test_cancel_all_for_turn_cancels_matching_tasks(self):
        """cancel_all_for_turn(token) 取消所有匹配 turn_id 的 tasks"""
        coordinator = InterruptCoordinator()

        # Register Turn 1 tasks
        coordinator.set_current_turn(1)
        token1 = TurnCancellationToken(turn_id=1)
        llm_task1 = asyncio.create_task(asyncio.sleep(10))
        tts_task1 = asyncio.create_task(asyncio.sleep(10))
        coordinator.register_llm_task(llm_task1, token1)
        coordinator.register_tts_consumer_task(tts_task1, token1)

        # Register Turn 2 tasks
        coordinator.set_current_turn(2)
        token2 = TurnCancellationToken(turn_id=2)
        llm_task2 = asyncio.create_task(asyncio.sleep(10))
        tts_task2 = asyncio.create_task(asyncio.sleep(10))
        coordinator.register_llm_task(llm_task2, token2)
        coordinator.register_tts_consumer_task(tts_task2, token2)

        # Cancel all Turn 1 tasks
        coordinator.cancel_all_for_turn(token1)

        # Give tasks a moment to process cancellation
        await asyncio.sleep(0.01)

        # Turn 1 tasks should be cancelled (or cancelling)
        assert llm_task1.cancelled() or llm_task1.cancelling()
        assert tts_task1.cancelled() or tts_task1.cancelling()

        # Turn 2 tasks should NOT be cancelled
        assert not llm_task2.cancelled()
        assert not tts_task2.cancelled()

        # Cleanup remaining tasks
        for t in [llm_task2, tts_task2]:
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_cancel_all_for_turn_also_cancels_generators(self):
        """cancel_all_for_turn 也取消 TTS generators"""
        coordinator = InterruptCoordinator()

        # Register Turn 1 generator
        coordinator.set_current_turn(1)
        token1 = TurnCancellationToken(turn_id=1)

        async def dummy_gen():
            try:
                while True:
                    await asyncio.sleep(0.1)
                    yield b"audio"
            except asyncio.CancelledError:
                raise

        gen1 = dummy_gen()
        coordinator.register_tts_generator(gen1, token1)

        # Start consuming the generator in a task
        async def consume_gen():
            async for chunk in gen1:
                pass

        consume_task = asyncio.create_task(consume_gen())

        # Cancel Turn 1
        coordinator.cancel_all_for_turn(token1)

        # Generator should be closed (tasks that use it would get CancelledError)
        # The generator itself is closed via aclose()
        assert not coordinator.has_tts_generator(gen1)

        # Cleanup
        consume_task.cancel()
        try:
            await consume_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_cancel_all_for_turn_token_propagation(self):
        """cancel_all_for_turn 传播取消信号到 token"""
        coordinator = InterruptCoordinator()

        coordinator.set_current_turn(1)
        token1 = TurnCancellationToken(turn_id=1)
        llm_task = asyncio.create_task(asyncio.sleep(10))
        coordinator.register_llm_task(llm_task, token1)

        # Cancel via coordinator
        coordinator.cancel_all_for_turn(token1)

        # Token should also be cancelled
        assert token1.cancelled

        # Cleanup
        try:
            await llm_task
        except asyncio.CancelledError:
            pass

    def test_cancel_all_for_turn_with_wrong_turn_id_fails(self):
        """cancel_all_for_turn 用错误 turn_id 的 token 应该失败"""
        coordinator = InterruptCoordinator()

        coordinator.set_current_turn(2)
        token_wrong = TurnCancellationToken(turn_id=999)  # Non-existent turn

        with pytest.raises(ValueError, match="no tasks registered"):
            coordinator.cancel_all_for_turn(token_wrong)

    @pytest.mark.asyncio
    async def test_multiple_registrations_same_turn(self):
        """同一 turn 可以注册多个 task"""
        coordinator = InterruptCoordinator()
        coordinator.set_current_turn(1)

        token = TurnCancellationToken(turn_id=1)
        task1 = asyncio.create_task(asyncio.sleep(1))
        task2 = asyncio.create_task(asyncio.sleep(1))
        task3 = asyncio.create_task(asyncio.sleep(1))

        coordinator.register_llm_task(task1, token)
        coordinator.register_llm_task(task2, token)
        coordinator.register_tts_consumer_task(task3, token)

        assert coordinator.has_llm_task(task1)
        assert coordinator.has_llm_task(task2)
        assert coordinator.has_tts_consumer_task(task3)

        # Cleanup
        for t in [task1, task2, task3]:
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass

    def test_coordinator_has_clear_repr(self):
        """Coordinator 有清晰的 repr 用于调试"""
        coordinator = InterruptCoordinator()
        assert "current_turn_id=0" in repr(coordinator)

        coordinator.set_current_turn(5)
        assert "current_turn_id=5" in repr(coordinator)