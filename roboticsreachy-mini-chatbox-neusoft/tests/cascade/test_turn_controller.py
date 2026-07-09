"""Tests for TurnController.

Task 6: TurnController 实现

Test acceptance criteria:
- start_new_turn() returns (turn_id, token) where turn_id == token.turn_id
- handle_barge_in() cancels old token, returns new turn_id
- current_turn_id property reflects current generation
- audio generation must use token.turn_id
"""

from __future__ import annotations

import asyncio

import pytest


class TestTurnController:
    """Tests for TurnController turn lifecycle management."""

    def test_turn_controller_initial_state(self):
        """TurnController 初始状态 turn_id 为 0"""
        from reachy_mini_conversation_app.cascade.turn_controller import (
            TurnController,
        )

        controller = TurnController()
        assert controller.current_turn_id == 0

    def test_start_new_turn_returns_matching_turn_id_and_token(self):
        """start_new_turn() 返回的 turn_id == token.turn_id"""
        from reachy_mini_conversation_app.cascade.turn_controller import (
            TurnController,
        )

        controller = TurnController()
        turn_id, token = controller.start_new_turn()

        # turn_id must match token.turn_id
        assert turn_id == token.turn_id
        assert turn_id == 1  # First turn starts at 1

        # Token should not be cancelled initially
        assert token.cancelled is False

    def test_start_new_turn_increments_turn_id(self):
        """多次调用 start_new_turn() 递增 turn_id"""
        from reachy_mini_conversation_app.cascade.turn_controller import (
            TurnController,
        )

        controller = TurnController()

        turn_id1, token1 = controller.start_new_turn()
        turn_id2, token2 = controller.start_new_turn()
        turn_id3, token3 = controller.start_new_turn()

        # Each turn_id should be unique and incrementing
        assert turn_id1 == 1
        assert turn_id2 == 2
        assert turn_id3 == 3

        # Each token should have matching turn_id
        assert token1.turn_id == 1
        assert token2.turn_id == 2
        assert token3.turn_id == 3

        # Current turn_id should reflect latest
        assert controller.current_turn_id == 3

    def test_start_new_turn_updates_current_turn_id(self):
        """start_new_turn() 更新 current_turn_id"""
        from reachy_mini_conversation_app.cascade.turn_controller import (
            TurnController,
        )

        controller = TurnController()
        assert controller.current_turn_id == 0

        turn_id1, _ = controller.start_new_turn()
        assert controller.current_turn_id == turn_id1
        assert controller.current_turn_id == 1

        turn_id2, _ = controller.start_new_turn()
        assert controller.current_turn_id == turn_id2
        assert controller.current_turn_id == 2

    def test_handle_barge_in_cancels_current_token(self):
        """handle_barge_in() 取消当前 turn 的 token"""
        from reachy_mini_conversation_app.cascade.turn_controller import (
            TurnController,
        )

        controller = TurnController()

        # Start turn 1
        turn_id1, token1 = controller.start_new_turn()
        assert token1.cancelled is False

        # Barge-in cancels turn 1 and returns new turn_id and token
        new_turn_id, new_token = controller.handle_barge_in()

        # Token 1 should now be cancelled
        assert token1.cancelled is True

        # New turn should be started
        assert new_turn_id == turn_id1 + 1
        assert new_token.turn_id == new_turn_id

    def test_handle_barge_in_starts_new_turn(self):
        """handle_barge_in() 触发打断并返回新 turn_id 和 token"""
        from reachy_mini_conversation_app.cascade.turn_controller import (
            TurnController,
        )

        controller = TurnController()

        # Start turn 1
        turn_id1, token1 = controller.start_new_turn()

        # Barge-in returns new turn_id and token
        new_turn_id, new_token = controller.handle_barge_in()

        # New turn_id should be incremented
        assert new_turn_id == 2

        # New token should match new turn_id
        assert new_token.turn_id == 2

        # Current turn should be updated
        assert controller.current_turn_id == 2

    def test_handle_barge_in_returns_new_turn_id_and_token(self):
        """handle_barge_in() 返回新的 turn_id 和 token"""
        from reachy_mini_conversation_app.cascade.turn_controller import (
            TurnController,
        )

        controller = TurnController()

        # Start turn 1
        _, token1 = controller.start_new_turn()

        # Barge-in returns new turn_id and token
        new_turn_id, new_token = controller.handle_barge_in()

        # New token should have matching turn_id
        assert new_turn_id == new_token.turn_id
        assert new_turn_id == 2

        # New token should not be cancelled
        assert new_token.cancelled is False

        # Old token should be cancelled
        assert token1.cancelled is True

    def test_handle_barge_in_without_current_turn_is_noop(self):
        """handle_barge_in() 在没有当前 turn 时是 noop"""
        from reachy_mini_conversation_app.cascade.turn_controller import (
            TurnController,
        )

        controller = TurnController()

        # No turn has been started yet (current_turn_id == 0)
        # handle_barge_in should just start turn 1
        turn_id, token = controller.handle_barge_in()

        assert turn_id == 1
        assert token.turn_id == 1
        assert controller.current_turn_id == 1

    def test_multiple_barge_ins_increment_turn_id(self):
        """多次打断正确递增 turn_id"""
        from reachy_mini_conversation_app.cascade.turn_controller import (
            TurnController,
        )

        controller = TurnController()

        # Start turn 1
        turn_id1, token1 = controller.start_new_turn()

        # First barge-in
        turn_id2, token2 = controller.handle_barge_in()
        assert token1.cancelled is True
        assert turn_id2 == 2

        # Second barge-in
        turn_id3, token3 = controller.handle_barge_in()
        assert token2.cancelled is True
        assert turn_id3 == 3

        # Third barge-in
        turn_id4, token4 = controller.handle_barge_in()
        assert token3.cancelled is True
        assert turn_id4 == 4

        # Current turn should be 4
        assert controller.current_turn_id == 4

    def test_turn_tokens_are_independent(self):
        """每个 turn 的 token 独立，取消互不影响"""
        from reachy_mini_conversation_app.cascade.turn_controller import (
            TurnController,
        )

        controller = TurnController()

        # Start multiple turns
        _, token1 = controller.start_new_turn()
        _, token2 = controller.start_new_turn()
        _, token3 = controller.start_new_turn()

        # Cancel token1 (simulate stale turn cancellation)
        token1.cancel()

        # Other tokens should not be affected
        assert token1.cancelled is True
        assert token2.cancelled is False
        assert token3.cancelled is False

    def test_turn_controller_has_coordinator_integration(self):
        """TurnController 与 InterruptCoordinator 配合使用"""
        import asyncio

        from reachy_mini_conversation_app.cascade.interrupt_coordinator import (
            InterruptCoordinator,
        )
        from reachy_mini_conversation_app.cascade.turn_controller import (
            TurnController,
        )

        async def run_test():
            coordinator = InterruptCoordinator()
            controller = TurnController(coordinator=coordinator)

            # Start turn 1
            turn_id, token = controller.start_new_turn()

            # Coordinator's current_turn_id should be updated
            assert coordinator.current_turn_id == turn_id

            # Register a task with the token
            task = asyncio.create_task(asyncio.sleep(1))
            coordinator.register_llm_task(task, token)

            # Barge-in should cancel the task
            controller.handle_barge_in()

            # Give a moment for cancellation to propagate
            await asyncio.sleep(0.01)

            # Task should be cancelled
            assert task.cancelled()

            # Token should be cancelled
            assert token.cancelled is True

        asyncio.run(run_test())

    def test_turn_controller_without_coordinator_still_works(self):
        """TurnController 可以不传 coordinator 独立使用"""
        from reachy_mini_conversation_app.cascade.turn_controller import (
            TurnController,
        )

        # Create without coordinator
        controller = TurnController()

        turn_id, token = controller.start_new_turn()
        assert turn_id == 1
        assert token.turn_id == 1

        new_turn_id, new_token = controller.handle_barge_in()
        assert new_turn_id == 2
        assert token.cancelled is True

    def test_turn_controller_has_clear_repr(self):
        """TurnController 有清晰的 repr 用于调试"""
        from reachy_mini_conversation_app.cascade.turn_controller import (
            TurnController,
        )

        controller = TurnController()
        assert "current_turn_id=0" in repr(controller)

        controller.start_new_turn()
        assert "current_turn_id=1" in repr(controller)

    def test_turn_controller_tracks_current_token(self):
        """TurnController 追踪当前活跃的 token"""
        from reachy_mini_conversation_app.cascade.turn_controller import (
            TurnController,
        )

        controller = TurnController()

        # Start turn 1
        _, token1 = controller.start_new_turn()

        # Current token should be token1
        assert controller.current_token is not None
        assert controller.current_token.turn_id == 1

        # Start turn 2
        _, token2 = controller.start_new_turn()

        # Current token should now be token2
        assert controller.current_token is not None
        assert controller.current_token.turn_id == 2

        # token1 should not be current
        assert controller.current_token is not token1

    def test_start_new_turn_can_register_tasks_with_coordinator(self):
        """start_new_turn 返回的 token 可用于注册 tasks"""
        import asyncio

        from reachy_mini_conversation_app.cascade.interrupt_coordinator import (
            InterruptCoordinator,
        )
        from reachy_mini_conversation_app.cascade.turn_controller import (
            TurnController,
        )

        async def run_test():
            coordinator = InterruptCoordinator()
            controller = TurnController(coordinator=coordinator)

            # Start turn and get token
            turn_id, token = controller.start_new_turn()

            # Use token to register tasks
            llm_task = asyncio.create_task(asyncio.sleep(1))
            tts_task = asyncio.create_task(asyncio.sleep(1))

            coordinator.register_llm_task(llm_task, token)
            coordinator.register_tts_consumer_task(tts_task, token)

            # Both tasks should be registered
            assert coordinator.has_llm_task(llm_task)
            assert coordinator.has_tts_consumer_task(tts_task)

            # Cleanup
            llm_task.cancel()
            tts_task.cancel()
            try:
                await llm_task
            except asyncio.CancelledError:
                pass
            try:
                await tts_task
            except asyncio.CancelledError:
                pass

        asyncio.run(run_test())

    def test_generation_id_equals_turn_id(self):
        """generation ID = turn_id"""
        from reachy_mini_conversation_app.cascade.turn_controller import (
            TurnController,
        )

        controller = TurnController()

        turn_id, token = controller.start_new_turn()

        # generation_id (if exposed) should equal turn_id
        # In this design, token.turn_id serves as the generation ID
        assert token.turn_id == turn_id
        assert controller.current_turn_id == turn_id

        # For audio generation, use token.turn_id
        generation_id = token.turn_id
        assert generation_id == 1

    def test_concurrent_start_new_turn_is_safe(self):
        """并发调用 start_new_turn 是安全的（turn_id 递增）"""
        import asyncio

        from reachy_mini_conversation_app.cascade.turn_controller import (
            TurnController,
        )

        controller = TurnController()

        results: list[tuple[int, object]] = []

        async def start_turn():
            turn_id, token = controller.start_new_turn()
            results.append((turn_id, token))

        async def run_concurrent():
            # Start multiple turns concurrently
            await asyncio.gather(
                start_turn(),
                start_turn(),
                start_turn(),
            )

        asyncio.run(run_concurrent())

        # All turn_ids should be unique
        turn_ids = [r[0] for r in results]
        assert len(turn_ids) == 3
        assert len(set(turn_ids)) == 3  # All unique

        # All tokens should have matching turn_ids
        for turn_id, token in results:
            assert token.turn_id == turn_id

    def test_negative_turn_id_is_invalid(self):
        """turn_id 不能为负数"""
        from reachy_mini_conversation_app.cascade.turn_controller import (
            TurnController,
        )

        controller = TurnController()

        # This shouldn't happen in normal usage, but validate if exposed
        # The internal _next_turn_id should always be positive
        for _ in range(100):
            turn_id, _ = controller.start_new_turn()
            assert turn_id > 0