"""Utilities for converting streamed LLM text into TTS-friendly segments."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reachy_mini_conversation_app.cascade.interrupt_coordinator import (
        TurnCancellationToken,
    )


class SentenceChunker:
    """Incrementally split text deltas into short, speakable segments.

    Supports interrupt-aware behavior:
    - interrupt() marks the chunker as interrupted
    - push() checks token.cancelled and returns empty list if cancelled
    - flush_on_interrupt() discards incomplete buffer and returns None
    - reset() clears buffer and interrupted state
    """

    def __init__(
        self,
        min_chars: int = 4,
        max_chars: int = 36,
        soft_punctuation: str = "\u3001",
        hard_punctuation: str = ".!?,;\u3002\uFF01\uFF1F\uFF0C\uFF1B\n",
    ) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.soft_punctuation = soft_punctuation
        self.hard_punctuation = hard_punctuation
        self._buffer = ""
        self._interrupted = False

    @property
    def is_interrupted(self) -> bool:
        """Whether the chunker has been interrupted."""
        return self._interrupted

    def interrupt(self) -> None:
        """Mark the chunker as interrupted.

        Once interrupted:
        - push() returns empty list
        - flush() returns None
        - flush_on_interrupt() returns None and discards buffer
        """
        self._interrupted = True

    def reset(self) -> None:
        """Clear buffer and reset interrupted state.

        After reset(), the chunker can be used normally for a new turn.
        """
        self._buffer = ""
        self._interrupted = False

    def push(
        self,
        text_delta: str,
        token: "TurnCancellationToken | None" = None,
    ) -> list[str]:
        """Add text and return any complete segments.

        Args:
            text_delta: The text to add.
            token: Optional cancellation token. If provided and cancelled,
                   returns empty list without processing.

        Returns:
            List of complete segments, or empty list if interrupted or
            token is cancelled.
        """
        # Check token cancellation first
        if token is not None and token.cancelled:
            return []

        # Check internal interrupt state
        if self._interrupted:
            return []

        if not text_delta:
            return []

        self._buffer += text_delta
        segments: list[str] = []

        while True:
            split_index = self._find_split_index()
            if split_index is None:
                break

            segment = self._buffer[:split_index].strip()
            self._buffer = self._buffer[split_index:].lstrip()
            if segment:
                segments.append(segment)

        return segments

    def flush(self) -> str | None:
        """Return the remaining buffered text.

        Returns:
            Buffered text if not interrupted, None if interrupted.
        """
        if self._interrupted:
            return None

        segment = self._buffer.strip()
        self._buffer = ""
        return segment or None

    def flush_on_interrupt(self) -> str | None:
        """Discard incomplete buffer and return None.

        This method is called when an interrupt occurs to discard
        any incomplete text in the buffer. Always returns None and
        clears the buffer regardless of interrupt state.

        Returns:
            Always None (buffer is discarded).
        """
        self._buffer = ""
        return None

    def _find_split_index(self) -> int | None:
        if len(self._buffer.strip()) < self.min_chars:
            return None

        for idx, char in enumerate(self._buffer):
            if char in self.hard_punctuation and idx + 1 >= self.min_chars:
                return idx + 1

        if len(self._buffer) >= self.max_chars:
            for idx in range(min(len(self._buffer), self.max_chars) - 1, self.min_chars - 1, -1):
                if self._buffer[idx] in self.soft_punctuation:
                    return idx + 1
            return self.max_chars

        return None
