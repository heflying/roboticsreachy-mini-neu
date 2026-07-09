"""Turn-level lifecycle management for cascade pipeline.

Task 6: TurnController 实现

This module provides the TurnController class for managing turn-level
lifecycle in the cascade pipeline.

Architecture Decision:
- TurnController is the single source of turn_id (generation ID)
- Each turn creates an independent TurnCancellationToken
- TurnController uses InterruptCoordinator for task registration/cancellation

Key Design:
- start_new_turn() -> (turn_id, token) where turn_id == token.turn_id
- handle_barge_in() -> (new_turn_id, new_token) (increments turn_id)
- All audio generation uses token.turn_id as generation ID

Acceptance Criteria:
- start_new_turn() returns turn_id == token.turn_id
- handle_barge_in() cancels old token, returns new turn_id
- audio generation must use token.turn_id
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, TYPE_CHECKING

from .interrupt_coordinator import TurnCancellationToken, InterruptCoordinator


if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


class TurnController:
    """Manage turn-level lifecycle for cascade pipeline.

    **Core Design: turn_id is the single source of generation ID**

    - start_new_turn() creates new token and increments turn_id
    - handle_barge_in() cancels current turn and creates new turn_id + new_token
    - All audio generation uses token.turn_id as generation ID

    Responsibilities:
    - Maintain global turn_id counter
    - Create TurnCancellationToken for each turn
    - Coordinate interrupt events via InterruptCoordinator
    - Provide token for LLM/TTS/SentenceChunker to check cancellation

    Usage:
    - Each time user starts speaking: start_new_turn() -> (turn_id, token)
    - User barge-in: handle_barge_in() -> (new_turn_id, new_token)
    - LLM/TTS/Playback use token.turn_id for isolation

    Example:
        controller = TurnController()
        turn_id, token = controller.start_new_turn()
        # Audio generation uses turn_id = token.turn_id
        # Check token.cancelled during LLM/TTS generation
        # Barge-in: controller.handle_barge_in() -> (new_turn_id, new_token)
    """

    __slots__ = (
        "_turn_id_counter",
        "_current_token",
        "_coordinator",
        "_handler",
        "_audio_playback",
        "_event_loop",
    )

    def __init__(
        self,
        handler: Any | None = None,
        audio_playback: Any | None = None,
        coordinator: InterruptCoordinator | None = None,
    ) -> None:
        """Initialize TurnController.

        Args:
            handler: CascadeHandler instance (optional, for future integration)
            audio_playback: AudioPlaybackSystem instance (optional, for interrupt)
            coordinator: InterruptCoordinator instance (optional, creates new if None)
        """
        self._turn_id_counter: int = 0
        self._current_token: TurnCancellationToken | None = None
        self._coordinator: InterruptCoordinator = coordinator if coordinator is not None else InterruptCoordinator()
        self._handler = handler
        self._audio_playback = audio_playback
        self._event_loop: asyncio.AbstractEventLoop | None = None

        logger.info("TurnController initialized")

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set event loop for cross-thread interrupt support.

        Args:
            loop: The asyncio event loop running in the main thread.

        Note:
            This should be called when the handler starts its event loop.
            Required for VAD thread to safely trigger interrupts.
        """
        self._event_loop = loop
        logger.debug("Event loop set for TurnController")

    def set_audio_playback(self, audio_playback: Any) -> None:
        """Set audio playback system (called after Gradio UI initialization).

        Args:
            audio_playback: AudioPlaybackSystem instance.
        """
        self._audio_playback = audio_playback
        logger.info("AudioPlayback set for TurnController")

    def start_new_turn(self) -> tuple[int, TurnCancellationToken]:
        """Start a new turn.

        **Critical: turn_id comes from the token**

        Flow:
        1. Increment turn_id counter
        2. Create new TurnCancellationToken with turn_id
        3. Update coordinator's current_turn_id
        4. Return (turn_id, token) where turn_id == token.turn_id

        Returns:
            (turn_id, token) - turn_id for audio isolation, token for cancellation check

        Note:
            Audio generation must use token.turn_id as generation ID.
            This ensures generation == turn_id for all audio in this turn.
        """
        # Increment counter
        self._turn_id_counter += 1
        new_turn_id = self._turn_id_counter

        # Create new token
        new_token = TurnCancellationToken(turn_id=new_turn_id)
        self._current_token = new_token

        # Update coordinator
        self._coordinator.set_current_turn(new_turn_id)

        logger.info(f"Turn {new_turn_id} started (generation={new_turn_id})")
        return new_turn_id, new_token

    def handle_barge_in(self) -> tuple[int, TurnCancellationToken]:
        """Handle user barge-in: interrupt current turn, start new turn.

        **Cross-thread safe: can be called from VAD thread**

        Execution:
        1. Cancel current token (mark as cancelled)
        2. Cancel registered tasks via coordinator
        3. Interrupt audio playback (if available)
        4. Start new turn (increment turn_id counter and create new token)
        5. Return (new_turn_id, new_token)

        Returns:
            (new_turn_id, new_token) - new turn_id for audio isolation,
                                        new_token for cancellation check

        Note:
            Old token is cancelled (sticky state). New token is fresh.
            If no current turn exists, just starts turn 1.
        """
        # Cancel current token if exists
        if self._current_token is not None and not self._current_token.cancelled:
            self._current_token.cancel()
            logger.info(f"Turn {self._current_token.turn_id} cancelled by barge-in")

            # Mark the timing tracker as cancelled for report labeling
            from .timing import tracker as _tracker
            _tracker.mark_cancelled()

            # Cancel registered tasks for this turn
            try:
                self._coordinator.cancel_all_for_turn(self._current_token)
            except ValueError as e:
                # No tasks registered for this turn - OK
                logger.debug(f"No tasks to cancel for turn {self._current_token.turn_id}: {e}")

        # Interrupt audio playback (synchronous, cross-thread safe)
        # First increment counter for new generation
        self._turn_id_counter += 1
        new_turn_id = self._turn_id_counter

        if self._audio_playback is not None:
            try:
                # playback.interrupt(new_generation) discards stale audio
                if hasattr(self._audio_playback, "interrupt"):
                    self._audio_playback.interrupt(new_turn_id)
                    logger.info(f"AudioPlayback interrupted, new generation={new_turn_id}")
                else:
                    logger.warning("AudioPlayback has no interrupt method")
            except Exception as e:
                logger.warning(f"Failed to interrupt AudioPlayback: {e}")

        # Create new token for new turn
        new_token = TurnCancellationToken(turn_id=new_turn_id)
        self._current_token = new_token

        # Update coordinator for new turn
        self._coordinator.set_current_turn(new_turn_id)

        logger.info(f"Barge-in handled: new turn_id={new_turn_id}")
        return new_turn_id, new_token

    @property
    def current_turn_id(self) -> int:
        """Current turn ID (also the current generation ID).

        Returns 0 if no turn has been started yet.
        """
        return self._turn_id_counter

    @property
    def current_token(self) -> TurnCancellationToken | None:
        """Current turn's cancellation token (if any)."""
        return self._current_token

    @property
    def coordinator(self) -> InterruptCoordinator:
        """Get InterruptCoordinator (for registering tasks/generators)."""
        return self._coordinator

    def is_current_turn(self, turn_id: int) -> bool:
        """Check if given turn_id is the current turn.

        Args:
            turn_id: The turn_id to check.

        Returns:
            True if turn_id == current_turn_id, False otherwise.
        """
        return turn_id == self._turn_id_counter

    def get_token_for_turn(self, turn_id: int) -> TurnCancellationToken | None:
        """Get token for a specific turn_id (for compatibility).

        Args:
            turn_id: The turn_id to get token for.

        Returns:
            The current token if turn_id matches, None otherwise.

        Note:
            This is a convenience method. In normal usage, the token
            returned by start_new_turn() should be passed through
            the pipeline.
        """
        if self._current_token is not None and self._current_token.turn_id == turn_id:
            return self._current_token
        return None

    def __repr__(self) -> str:
        """Return a clear representation for debugging."""
        token_info = ""
        if self._current_token is not None:
            token_info = f", token.turn_id={self._current_token.turn_id}, cancelled={self._current_token.cancelled}"
        return f"TurnController(current_turn_id={self._turn_id_counter}{token_info})"


__all__ = ["TurnController"]