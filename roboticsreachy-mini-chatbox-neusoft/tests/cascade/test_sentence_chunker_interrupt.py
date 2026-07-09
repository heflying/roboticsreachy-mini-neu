"""Tests for SentenceChunker interrupt-aware behavior."""

from __future__ import annotations

import pytest

from reachy_mini_conversation_app.cascade.interrupt_coordinator import (
    TurnCancellationToken,
)
from reachy_mini_conversation_app.cascade.streaming_text import SentenceChunker


class TestSentenceChunkerInterrupt:
    """Tests for interrupt-aware SentenceChunker."""

    def test_interrupt_marks_interrupted_state(self) -> None:
        """interrupt() should mark the chunker as interrupted."""
        chunker = SentenceChunker()
        assert not chunker.is_interrupted
        chunker.interrupt()
        assert chunker.is_interrupted

    def test_push_returns_empty_list_after_interrupt(self) -> None:
        """After interrupt(), push() should return empty list."""
        chunker = SentenceChunker()
        # Push some text first
        segments = chunker.push("Hello world, this is a test.")
        assert len(segments) > 0  # Should get some segments

        # Interrupt and push more
        chunker.interrupt()
        segments = chunker.push("More text here.")
        assert segments == []

    def test_push_returns_empty_when_token_cancelled(self) -> None:
        """push() with cancelled token should return empty list."""
        chunker = SentenceChunker()
        token = TurnCancellationToken(turn_id=1)
        token.cancel()

        segments = chunker.push("Hello world.", token=token)
        assert segments == []

    def test_push_without_token_works_normally(self) -> None:
        """push() without token should work normally."""
        chunker = SentenceChunker()
        segments = chunker.push("Hello world, this is a test.")
        assert len(segments) > 0

    def test_flush_on_interrupt_returns_none(self) -> None:
        """After interrupt(), flush_on_interrupt() should return None."""
        chunker = SentenceChunker()
        chunker.push("Incomplete text")
        chunker.interrupt()
        result = chunker.flush_on_interrupt()
        assert result is None

    def test_flush_on_interrupt_discards_buffer(self) -> None:
        """flush_on_interrupt() should discard the buffer."""
        chunker = SentenceChunker()
        chunker.push("Incomplete text")
        chunker.interrupt()
        chunker.flush_on_interrupt()
        # Buffer should be empty after flush_on_interrupt
        assert chunker._buffer == ""

    def test_normal_flush_returns_buffered_text(self) -> None:
        """Normal flush() should return buffered text."""
        chunker = SentenceChunker()
        chunker.push("Incomplete text")
        result = chunker.flush()
        assert result == "Incomplete text"

    def test_flush_after_interrupt_returns_none(self) -> None:
        """flush() after interrupt should return None (buffer discarded)."""
        chunker = SentenceChunker()
        chunker.push("Incomplete text")
        chunker.interrupt()
        result = chunker.flush()
        assert result is None

    def test_reset_clears_interrupt_state(self) -> None:
        """reset() should clear the interrupted state."""
        chunker = SentenceChunker()
        chunker.interrupt()
        assert chunker.is_interrupted
        chunker.reset()
        assert not chunker.is_interrupted

    def test_reset_clears_buffer(self) -> None:
        """reset() should clear the buffer."""
        chunker = SentenceChunker()
        chunker.push("Some text")
        chunker.reset()
        assert chunker._buffer == ""

    def test_push_after_reset_works_normally(self) -> None:
        """After reset(), push() should work normally."""
        chunker = SentenceChunker()
        chunker.push("First text, ")
        chunker.interrupt()
        chunker.reset()

        # Now push should work again
        segments = chunker.push("Hello world, this is new.")
        assert len(segments) > 0

    def test_interrupt_is_idempotent(self) -> None:
        """Calling interrupt() multiple times should be safe."""
        chunker = SentenceChunker()
        chunker.interrupt()
        chunker.interrupt()
        assert chunker.is_interrupted

        # Push should still return empty
        segments = chunker.push("More text.")
        assert segments == []

    def test_flush_on_interrupt_without_interrupt_returns_buffer(self) -> None:
        """flush_on_interrupt() without prior interrupt should still discard."""
        chunker = SentenceChunker()
        chunker.push("Incomplete text")
        result = chunker.flush_on_interrupt()
        assert result is None
        assert chunker._buffer == ""

    def test_push_with_valid_token_processes_normally(self) -> None:
        """push() with non-cancelled token should process normally."""
        chunker = SentenceChunker()
        token = TurnCancellationToken(turn_id=1)
        # Token not cancelled

        segments = chunker.push("Hello world, this is a test.", token=token)
        assert len(segments) > 0