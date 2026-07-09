"""Interrupt coordination for cascade pipeline.

This module provides the TurnCancellationToken class for turn-level
cancellation signal propagation in the cascade pipeline.

Architecture Decision A1: Each turn creates an independent token object.
- Reason: Avoid cross-turn state pollution
- Sticky cancelled state ensures old coroutines safely terminate
- Excluded: Single token + reset() approach - reset may miss cancellation propagation
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any


logger = logging.getLogger(__name__)


class TurnCancellationToken:
    """Turn-level cancellation token with sticky cancelled state.

    Each turn has an independent token object. Once cancelled, the state
    remains True permanently (sticky). This ensures that old coroutines
    can safely detect cancellation and terminate without affecting new turns.

    Attributes:
        turn_id: The unique identifier for this turn (immutable).
        cancelled: Whether this turn has been cancelled (read-only, sticky).

    Example:
        token = TurnCancellationToken(turn_id=1)
        assert token.cancelled is False
        token.cancel()
        assert token.cancelled is True
        # cancelled state is sticky - stays True forever
        token.cancel()  # Safe to call multiple times
        assert token.cancelled is True
    """

    __slots__ = ("_turn_id", "_cancelled")

    def __init__(self, turn_id: int) -> None:
        """Initialize a new cancellation token for the given turn.

        Args:
            turn_id: Unique identifier for this turn. Must be non-negative.
        """
        if turn_id < 0:
            raise ValueError(f"turn_id must be non-negative, got {turn_id}")
        self._turn_id: int = turn_id
        self._cancelled: bool = False

    @property
    def turn_id(self) -> int:
        """The turn ID for this token (immutable)."""
        return self._turn_id

    @property
    def cancelled(self) -> bool:
        """Whether this turn has been cancelled (read-only, sticky)."""
        return self._cancelled

    def cancel(self) -> None:
        """Mark this turn as cancelled.

        Once cancelled, the state is sticky and cannot be reversed.
        Safe to call multiple times - subsequent calls are no-op.
        """
        self._cancelled = True

    def __repr__(self) -> str:
        """Return a clear representation for debugging."""
        return f"TurnCancellationToken(turn_id={self._turn_id}, cancelled={self._cancelled})"

    def __eq__(self, other: object) -> bool:
        """Check equality based on turn_id."""
        if not isinstance(other, TurnCancellationToken):
            return NotImplemented
        return self._turn_id == other._turn_id and self._cancelled == other._cancelled

    def __hash__(self) -> int:
        """Hash based on turn_id for use in sets/dicts."""
        return hash(self._turn_id)


class InterruptCoordinator:
    """Coordinator for interrupt-aware task management with ownership protection.

    Manages registration of LLM tasks, TTS consumer tasks, and TTS generators
    with strict ownership validation. Only the current turn's token can
    register/unregister resources.

    Architecture Decision A5: signal_end_of_turn accepts caller_turn_id parameter.
    - Reason: Prevent stale turn completion from affecting new turn
    - Excluded: signal_end_of_turn only reads current_generation - stale turn would incorrectly mark new turn complete

    Attributes:
        current_turn_id: The current turn ID (generation ID).

    Example:
        coordinator = InterruptCoordinator()
        coordinator.set_current_turn(1)
        token = TurnCancellationToken(turn_id=1)
        task = asyncio.create_task(some_work())
        coordinator.register_llm_task(task, token)  # Succeeds
        coordinator.cancel_all_for_turn(token)  # Cancels task and marks token cancelled
    """

    __slots__ = (
        "_current_turn_id",
        "_llm_tasks",
        "_tts_consumer_tasks",
        "_tts_generators",
        "_lock",
    )

    def __init__(self) -> None:
        """Initialize coordinator with no registered tasks."""
        self._current_turn_id: int = 0
        # Dict[task, turn_id] for ownership tracking
        self._llm_tasks: dict[asyncio.Task[Any], int] = {}
        self._tts_consumer_tasks: dict[asyncio.Task[Any], int] = {}
        # Dict[generator, turn_id] for ownership tracking
        self._tts_generators: dict[Any, int] = {}
        self._lock: asyncio.Lock = asyncio.Lock()

    @property
    def current_turn_id(self) -> int:
        """The current turn ID (generation ID)."""
        return self._current_turn_id

    def set_current_turn(self, turn_id: int) -> None:
        """Set the current turn ID.

        Args:
            turn_id: The new current turn ID.

        Note:
            This should be called when starting a new turn or after
            an interrupt to update the generation boundary.
        """
        if turn_id < 0:
            raise ValueError(f"turn_id must be non-negative, got {turn_id}")
        self._current_turn_id = turn_id
        logger.debug(f"Current turn set to {turn_id}")

    def _validate_ownership(self, token: TurnCancellationToken, operation: str) -> None:
        """Validate that token.turn_id matches current_turn_id.

        Args:
            token: The token to validate.
            operation: Description of the operation for error message.

        Raises:
            ValueError: If token.turn_id != current_turn_id.
        """
        if token.turn_id != self._current_turn_id:
            raise ValueError(
                f"{operation} failed: ownership mismatch - "
                f"token.turn_id={token.turn_id} != current_turn_id={self._current_turn_id}"
            )

    def register_llm_task(self, task: asyncio.Task[Any], token: TurnCancellationToken) -> None:
        """Register an LLM task for the current turn.

        Args:
            task: The asyncio Task to register.
            token: The cancellation token for ownership validation.

        Raises:
            ValueError: If token.turn_id != current_turn_id (ownership mismatch).
        """
        self._validate_ownership(token, "register_llm_task")
        self._llm_tasks[task] = token.turn_id
        logger.debug(f"Registered LLM task for turn {token.turn_id}")

    def register_tts_consumer_task(
        self, task: asyncio.Task[Any], token: TurnCancellationToken
    ) -> None:
        """Register a TTS consumer task for the current turn.

        Args:
            task: The asyncio Task to register.
            token: The cancellation token for ownership validation.

        Raises:
            ValueError: If token.turn_id != current_turn_id (ownership mismatch).
        """
        self._validate_ownership(token, "register_tts_consumer_task")
        self._tts_consumer_tasks[task] = token.turn_id
        logger.debug(f"Registered TTS consumer task for turn {token.turn_id}")

    def register_tts_generator(self, gen: Any, token: TurnCancellationToken) -> None:
        """Register a TTS async generator for the current turn.

        Args:
            gen: The async generator to register.
            token: The cancellation token for ownership validation.

        Raises:
            ValueError: If token.turn_id != current_turn_id (ownership mismatch).
        """
        self._validate_ownership(token, "register_tts_generator")
        self._tts_generators[gen] = token.turn_id
        logger.debug(f"Registered TTS generator for turn {token.turn_id}")

    def has_llm_task(self, task: asyncio.Task[Any]) -> bool:
        """Check if an LLM task is registered.

        Args:
            task: The task to check.

        Returns:
            True if the task is registered, False otherwise.
        """
        return task in self._llm_tasks

    def has_tts_consumer_task(self, task: asyncio.Task[Any]) -> bool:
        """Check if a TTS consumer task is registered.

        Args:
            task: The task to check.

        Returns:
            True if the task is registered, False otherwise.
        """
        return task in self._tts_consumer_tasks

    def has_tts_generator(self, gen: Any) -> bool:
        """Check if a TTS generator is registered.

        Args:
            gen: The generator to check.

        Returns:
            True if the generator is registered, False otherwise.
        """
        return gen in self._tts_generators

    def _validate_task_registered(
        self, task: asyncio.Task[Any], registry: dict[asyncio.Task[Any], int], operation: str
    ) -> int:
        """Validate that a task is registered and return its turn_id.

        Args:
            task: The task to validate.
            registry: The task registry dict.
            operation: Description of the operation for error message.

        Returns:
            The turn_id the task was registered with.

        Raises:
            KeyError: If the task is not registered.
        """
        if task not in registry:
            raise KeyError(f"{operation} failed: task not registered")
        return registry[task]

    def _validate_gen_registered(
        self, gen: Any, registry: dict[Any, int], operation: str
    ) -> int:
        """Validate that a generator is registered and return its turn_id.

        Args:
            gen: The generator to validate.
            registry: The generator registry dict.
            operation: Description of the operation for error message.

        Returns:
            The turn_id the generator was registered with.

        Raises:
            KeyError: If the generator is not registered.
        """
        if gen not in registry:
            raise KeyError(f"{operation} failed: generator not registered")
        return registry[gen]

    def unregister_llm_task(
        self, task: asyncio.Task[Any], token: TurnCancellationToken
    ) -> None:
        """Unregister an LLM task.

        Args:
            task: The task to unregister.
            token: The cancellation token for ownership validation.

        Raises:
            KeyError: If the task is not registered.
            ValueError: If token.turn_id != task's registered turn_id.
        """
        registered_turn_id = self._validate_task_registered(
            task, self._llm_tasks, "unregister_llm_task"
        )
        if token.turn_id != registered_turn_id:
            raise ValueError(
                f"unregister_llm_task failed: ownership mismatch - "
                f"token.turn_id={token.turn_id} != registered_turn_id={registered_turn_id}"
            )
        del self._llm_tasks[task]
        logger.debug(f"Unregistered LLM task for turn {token.turn_id}")

    def unregister_tts_consumer_task(
        self, task: asyncio.Task[Any], token: TurnCancellationToken
    ) -> None:
        """Unregister a TTS consumer task.

        Args:
            task: The task to unregister.
            token: The cancellation token for ownership validation.

        Raises:
            KeyError: If the task is not registered.
            ValueError: If token.turn_id != task's registered turn_id.
        """
        registered_turn_id = self._validate_task_registered(
            task, self._tts_consumer_tasks, "unregister_tts_consumer_task"
        )
        if token.turn_id != registered_turn_id:
            raise ValueError(
                f"unregister_tts_consumer_task failed: ownership mismatch - "
                f"token.turn_id={token.turn_id} != registered_turn_id={registered_turn_id}"
            )
        del self._tts_consumer_tasks[task]
        logger.debug(f"Unregistered TTS consumer task for turn {token.turn_id}")

    def unregister_tts_generator(self, gen: Any, token: TurnCancellationToken) -> None:
        """Unregister a TTS generator.

        Args:
            gen: The generator to unregister.
            token: The cancellation token for ownership validation.

        Raises:
            KeyError: If the generator is not registered.
            ValueError: If token.turn_id != generator's registered turn_id.
        """
        registered_turn_id = self._validate_gen_registered(
            gen, self._tts_generators, "unregister_tts_generator"
        )
        if token.turn_id != registered_turn_id:
            raise ValueError(
                f"unregister_tts_generator failed: ownership mismatch - "
                f"token.turn_id={token.turn_id} != registered_turn_id={registered_turn_id}"
            )
        del self._tts_generators[gen]
        logger.debug(f"Unregistered TTS generator for turn {token.turn_id}")

    def cancel_all_for_turn(self, token: TurnCancellationToken) -> None:
        """Cancel all tasks and generators registered for the given turn.

        Args:
            token: The cancellation token specifying which turn to cancel.

        Raises:
            ValueError: If no tasks are registered for the given turn_id.
        """
        turn_id = token.turn_id

        # Find tasks for this turn
        llm_tasks_to_cancel = [
            t for t, tid in self._llm_tasks.items() if tid == turn_id
        ]
        tts_tasks_to_cancel = [
            t for t, tid in self._tts_consumer_tasks.items() if tid == turn_id
        ]
        gens_to_cancel = [
            g for g, tid in self._tts_generators.items() if tid == turn_id
        ]

        if (
            not llm_tasks_to_cancel
            and not tts_tasks_to_cancel
            and not gens_to_cancel
        ):
            raise ValueError(
                f"cancel_all_for_turn failed: no tasks registered for turn_id={turn_id}"
            )

        # Cancel LLM tasks
        for task in llm_tasks_to_cancel:
            task.cancel()
            del self._llm_tasks[task]
            logger.info(f"Cancelled LLM task for turn {turn_id}")

        # Cancel TTS consumer tasks
        for task in tts_tasks_to_cancel:
            task.cancel()
            del self._tts_consumer_tasks[task]
            logger.info(f"Cancelled TTS consumer task for turn {turn_id}")

        # Close TTS generators
        for gen in gens_to_cancel:
            # Generators are closed via aclose() in their consuming tasks
            del self._tts_generators[gen]
            logger.info(f"Removed TTS generator for turn {turn_id}")

        # Mark token as cancelled
        token.cancel()
        logger.info(f"Turn {turn_id} cancellation complete")

    def __repr__(self) -> str:
        """Return a clear representation for debugging."""
        return (
            f"InterruptCoordinator(current_turn_id={self._current_turn_id}, "
            f"llm_tasks={len(self._llm_tasks)}, "
            f"tts_consumer_tasks={len(self._tts_consumer_tasks)}, "
            f"tts_generators={len(self._tts_generators)})"
        )


__all__ = ["TurnCancellationToken", "InterruptCoordinator"]