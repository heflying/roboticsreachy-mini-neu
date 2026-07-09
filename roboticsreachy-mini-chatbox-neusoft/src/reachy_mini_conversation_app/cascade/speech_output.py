"""SpeechOutput protocol and implementations for cascade TTS playback."""

from __future__ import annotations
import re
import time
import base64
import asyncio
import logging
import os
from typing import TYPE_CHECKING, Callable, Protocol, Awaitable, AsyncIterator

import numpy as np


if TYPE_CHECKING:
    import numpy.typing as npt

    from reachy_mini_conversation_app.cascade.tts import TTSProvider
    from reachy_mini_conversation_app.audio.head_wobbler import HeadWobbler
    from reachy_mini_conversation_app.cascade.ui.audio_playback import AudioPlaybackSystem
    from reachy_mini_conversation_app.cascade.interrupt_coordinator import TurnCancellationToken


logger = logging.getLogger(__name__)


class SpeechOutput(Protocol):
    """Protocol for TTS playback backends."""

    async def speak(
        self,
        text: str,
        *,
        token: TurnCancellationToken | None = None,
        turn_id: int = 0,
    ) -> None:
        """Synthesize and play speech."""
        ...

    async def speak_stream(self, text_chunks: AsyncIterator[str]) -> str:
        """Synthesize and play streamed text, returning the full spoken text."""
        ...


# ---------------------------------------------------------------------------
# Console mode
# ---------------------------------------------------------------------------


class ConsoleSpeechOutput:
    """TTS playback for console mode — streams chunks to a playback callback with rate limiting."""

    def __init__(
        self,
        tts: TTSProvider,
        head_wobbler: HeadWobbler | None,
        playback_callback: Callable[[bytes], Awaitable[None]],
    ) -> None:
        """Initialize with TTS provider, optional head wobbler, and playback callback."""
        self.tts = tts
        self.head_wobbler = head_wobbler
        self.playback_callback = playback_callback

    async def speak(
        self,
        text: str,
        *,
        token: TurnCancellationToken | None = None,
        turn_id: int = 0,
    ) -> None:
        """Stream TTS chunks to playback callback with rate limiting and head wobble."""
        from reachy_mini_conversation_app.cascade.timing import tracker

        if self.head_wobbler:
            self.head_wobbler.reset()

        logger.debug(f"[TTS] Speaking: {text}")

        audio_chunks: list[bytes] = []
        first_chunk_sent = False

        async for chunk in self.tts.synthesize(text):
            if token and token.cancelled:
                logger.info("Console TTS cancelled at turn %s", turn_id)
                break
            audio_chunks.append(chunk)

            if self.head_wobbler:
                self.head_wobbler.feed(base64.b64encode(chunk).decode("utf-8"))

            await self.playback_callback(chunk)
            if not first_chunk_sent:
                tracker.mark("audio_playback_started")
                first_chunk_sent = True

            # Rate limiting: match audio generation speed
            chunk_duration = len(chunk) / (2 * self.tts.sample_rate)
            await asyncio.sleep(chunk_duration * 0.95)

        logger.info(f"Generated {len(audio_chunks)} audio chunks for head animation")

        # Small buffer to let remaining audio drain
        await asyncio.sleep(0.5)

        if self.head_wobbler:
            self.head_wobbler.reset()

    async def speak_stream(self, text_chunks: AsyncIterator[str]) -> str:
        """Speak streamed text by flushing sentence-sized segments to TTS."""
        from reachy_mini_conversation_app.cascade.streaming_text import SentenceChunker

        chunker = SentenceChunker()
        full_text = ""
        async for text_delta in text_chunks:
            full_text += text_delta
            for segment in chunker.push(text_delta):
                await self.speak(segment)

        final_segment = chunker.flush()
        if final_segment:
            await self.speak(final_segment)
        return full_text


# ---------------------------------------------------------------------------
# Gradio mode
# ---------------------------------------------------------------------------


class GradioSpeechOutput:
    """TTS playback for Gradio mode — parallel sentence generation with ordered queuing."""

    def __init__(
        self,
        tts: TTSProvider,
        playback: AudioPlaybackSystem,
        barge_in_start_callback: Callable[[], None] | None = None,
        barge_in_stop_callback: Callable[[], None] | None = None,
    ) -> None:
        """Initialize with TTS provider and Gradio audio playback system.

        Args:
            tts: TTS provider for speech synthesis.
            playback: Pre-warmed audio playback system for low-latency output.
            barge_in_start_callback: Called when first audio chunk is queued (starts VAD monitor).
            barge_in_stop_callback: Called when playback completes or is interrupted (stops VAD monitor).

        """
        self.tts = tts
        self.playback = playback
        self.barge_in_start_callback = barge_in_start_callback
        self.barge_in_stop_callback = barge_in_stop_callback
        self.return_after_tts_queued = os.getenv(
            "CASCADE_RETURN_AFTER_TTS_QUEUED",
            "0",
        ).strip().lower() not in {"0", "false", "no", "off"}

    async def speak(
        self,
        text: str,
        *,
        token: TurnCancellationToken | None = None,
        turn_id: int = 0,
    ) -> None:
        """Split text into sentences, generate TTS in parallel, and queue for ordered playback."""
        from reachy_mini_conversation_app.cascade.timing import tracker

        logger.info(f"Synthesizing speech: '{text[:50]}...'")
        generation = turn_id or self.playback.current_generation

        if getattr(self.tts, "prefer_single_request", False):
            await self._speak_single_request(text, turn_id=generation, token=token)
            return

        audio_chunks: list[npt.NDArray[np.int16]] = []
        first_chunk_queued = False

        sentences = split_into_sentences(text)
        logger.debug(f"Split text into {len(sentences)} sentence chunks for streaming TTS")
        for i, s in enumerate(sentences):
            logger.debug(f"  Sentence {i + 1}: '{s}'")

        logger.debug("Using pre-warmed audio playback system")

        total_chunks = 0

        # Gate events ensure chunks queue in sentence order even if TTS responses arrive out of order
        queue_events = [asyncio.Event() for _ in sentences]
        queue_events[0].set()  # Sentence 0 can queue immediately

        async def generate_and_queue_sentence(idx: int, sentence: str) -> list[npt.NDArray[np.int16]]:
            """Generate TTS for one sentence and queue chunks in order."""
            nonlocal total_chunks, first_chunk_queued

            logger.debug(f"TTS sentence {idx + 1}/{len(sentences)}: '{sentence}' (PARALLEL)")
            sentence_chunks: list[npt.NDArray[np.int16]] = []
            sentence_start = time.time()

            gate_is_open = queue_events[idx].is_set()

            if gate_is_open:
                async for chunk in self.tts.synthesize(sentence):
                    if token and token.cancelled:
                        logger.info("TTS synthesis cancelled at turn %s", generation)
                        break
                    total_chunks += 1
                    audio_array = np.frombuffer(chunk, dtype=np.int16)
                    sentence_chunks.append(audio_array)
                    audio_chunks.append(audio_array)
                    self.playback.put_audio(audio_array, generation=generation)
                    self.playback.put_wobbler(chunk, generation=generation)
                    if not first_chunk_queued:
                        first_chunk_queued = True
                        tracker.mark("audio_playback_started")
                        logger.info("First audio chunk playing - playback started while TTS continues in background")
                        # Start barge-in monitor when first audio chunk is queued (R7)
                        if self.barge_in_start_callback is not None:
                            self.barge_in_start_callback()
            else:
                raw_chunks: list[bytes] = []
                async for chunk in self.tts.synthesize(sentence):
                    if token and token.cancelled:
                        logger.info("Buffered TTS synthesis cancelled at turn %s", generation)
                        break
                    total_chunks += 1
                    audio_array = np.frombuffer(chunk, dtype=np.int16)
                    sentence_chunks.append(audio_array)
                    raw_chunks.append(chunk)

                await queue_events[idx].wait()

                for audio_array, raw_chunk in zip(sentence_chunks, raw_chunks):
                    audio_chunks.append(audio_array)
                    self.playback.put_audio(audio_array, generation=generation)
                    self.playback.put_wobbler(raw_chunk, generation=generation)
                    if not first_chunk_queued:
                        first_chunk_queued = True
                        tracker.mark("audio_playback_started")
                        logger.info("First audio chunk playing - playback started while TTS continues in background")
                        # Start barge-in monitor when first audio chunk is queued (R7)
                        if self.barge_in_start_callback is not None:
                            self.barge_in_start_callback()

            gen_duration = time.time() - sentence_start
            if sentence_chunks:
                total_samples = sum(len(c) for c in sentence_chunks)
                logger.debug(
                    f"Sentence {idx + 1} generated: {len(sentence_chunks)} chunks "
                    f"({total_samples} samples, {total_samples / self.tts.sample_rate:.2f}s) "
                    f"in {gen_duration:.2f}s"
                )

            if idx + 1 < len(queue_events):
                queue_events[idx + 1].set()

            logger.debug(f"Sentence {idx + 1} queued for playback")
            return sentence_chunks

        # Generate sentences with intelligent overlap
        tasks: list[asyncio.Task[list[npt.NDArray[np.int16]]]] = []
        for idx, sentence in enumerate(sentences):
            if idx == 0:
                task = asyncio.create_task(generate_and_queue_sentence(idx, sentence))
                tasks.append(task)
            elif idx == 1:
                await asyncio.sleep(0.3)
                task = asyncio.create_task(generate_and_queue_sentence(idx, sentence))
                tasks.append(task)
            else:
                if idx >= 2 and tasks:
                    await tasks[idx - 1]
                task = asyncio.create_task(generate_and_queue_sentence(idx, sentence))
                tasks.append(task)

        await asyncio.gather(*tasks)
        logger.info(f"Parallel TTS complete: All {len(sentences)} sentences generated")
        logger.info(f"Generated {total_chunks} total audio chunks from {len(sentences)} sentences")

        completion = self.playback.signal_end_of_turn(caller_turn_id=generation)

        # Wait for audio to finish playing
        if audio_chunks:
            total_samples = sum(len(chunk) for chunk in audio_chunks)
            duration_seconds = total_samples / self.tts.sample_rate
            tracker.mark("tts_audio_queued", {"duration_s": round(duration_seconds, 2)})
            if self.return_after_tts_queued:
                logger.info(
                    "Audio queued for playback (audio=%.1fs); returning before local playback drain",
                    duration_seconds,
                )
                return
            logger.info(f"Waiting {duration_seconds:.1f}s for playback to complete...")
            await self._wait_for_playback_completion(completion, duration_seconds)

        logger.info("Playback complete (using pre-warmed system)")
        tracker.mark("playback_complete")
        # Stop barge-in monitor when playback completes (R7)
        if self.barge_in_stop_callback is not None:
            self.barge_in_stop_callback()

        # 注意：print_summary() 在 gradio_app.py 统一调用，包含 transcript_show

    async def _speak_single_request(
        self,
        text: str,
        streaming_dialog: bool = False,
        turn_id: int = 0,
        token: TurnCancellationToken | None = None,
    ) -> None:
        """Stream one TTS request directly to playback for providers that dislike sentence splitting.

        Args:
            text: Text to synthesize.
            streaming_dialog: Whether this is part of a streaming dialog.
            turn_id: Current turn ID for audio generation isolation.
            token: Optional cancellation token for interrupt detection.

        """
        from reachy_mini_conversation_app.cascade.timing import tracker

        audio_chunks: list[npt.NDArray[np.int16]] = []
        first_chunk_queued = False
        barge_in_started = False
        generation = turn_id or self.playback.current_generation

        try:
            async for chunk in self.tts.synthesize(text):
                # Check for cancellation during synthesis
                if token and token.cancelled:
                    logger.info(f"TTS synthesis cancelled at turn {turn_id}")
                    break

                audio_array = np.frombuffer(chunk, dtype=np.int16)
                audio_chunks.append(audio_array)

                # Put with generation=turn_id for playback isolation (R2)
                self.playback.put_audio(audio_array, generation=generation)
                self.playback.put_wobbler(chunk, generation=generation)

                if not first_chunk_queued:
                    first_chunk_queued = True
                    barge_in_started = True
                    tracker.mark("audio_playback_started")
                    if streaming_dialog:
                        tracker.mark("streaming_dialog_first_audio")
                    logger.info(
                        "First audio chunk playing (turn_id=%s) - playback started while TTS continues",
                        generation,
                    )
                    # Start barge-in monitor when first audio chunk is queued (R7)
                    if self.barge_in_start_callback is not None:
                        self.barge_in_start_callback()
        except Exception:
            # On error, stop barge-in monitor immediately
            if barge_in_started and self.barge_in_stop_callback is not None:
                self.barge_in_stop_callback()
            raise

        logger.info("Single-request TTS complete: generated %s audio chunks", len(audio_chunks))
        completion = self.playback.signal_end_of_turn(caller_turn_id=generation)

        if audio_chunks:
            total_samples = sum(len(chunk) for chunk in audio_chunks)
            duration_seconds = total_samples / self.tts.sample_rate
            tracker.mark("tts_audio_queued", {"duration_s": round(duration_seconds, 2)})
            if self.return_after_tts_queued:
                logger.info(
                    "Audio queued for playback (audio=%.1fs); returning before local playback drain",
                    duration_seconds,
                )
                # Stop barge-in monitor before returning (caller will handle playback)
                if barge_in_started and self.barge_in_stop_callback is not None:
                    self.barge_in_stop_callback()
                return
            logger.info(f"Waiting {duration_seconds:.1f}s for playback to complete...")
            await self._wait_for_playback_completion(completion, duration_seconds)

        logger.info("Playback complete (using pre-warmed system)")
        tracker.mark("playback_complete")
        # Stop barge-in monitor when playback completes (R7)
        if barge_in_started and self.barge_in_stop_callback is not None:
            self.barge_in_stop_callback()

        # 注意：print_summary() 在 gradio_app.py 统一调用

    async def speak_stream(
        self,
        text_chunks: AsyncIterator[str],
        token: TurnCancellationToken | None = None,
        turn_id: int = 0,
    ) -> str:
        """Stream LLM text deltas into TTS while the model is still generating.

        Args:
            text_chunks: AsyncIterator of text deltas from LLM.
            token: Optional TurnCancellationToken for interruption check.
            turn_id: Current turn ID for audio isolation (equals generation ID).

        Returns:
            The full text that was synthesized.

        """
        from reachy_mini_conversation_app.cascade.timing import tracker
        from reachy_mini_conversation_app.cascade.streaming_text import SentenceChunker

        logger.info(f"Synthesizing streamed speech from LLM text deltas (turn_id={turn_id})")
        generation = turn_id or self.playback.current_generation

        if getattr(self.tts, "prefer_single_request", False):
            full_text = ""
            tts_preconnect_triggered = False
            # Collect text while checking for cancellation
            async for text_delta in text_chunks:
                if token and token.cancelled:
                    logger.info(f"TTS collection cancelled at turn {turn_id}")
                    break
                full_text += text_delta

                # Delayed pre-connect: trigger on first text delta to reduce stale risk
                if not tts_preconnect_triggered and hasattr(self.tts, "prepare_stream"):
                    tts_preconnect_triggered = True
                    # Mark as preparing BEFORE starting task so synthesize can wait
                    self.tts._preparing = True  # type: ignore[attr-defined]
                    # Save task reference so synthesize can wait for it
                    task = asyncio.create_task(self.tts.prepare_stream())  # type: ignore[attr-defined]
                    self.tts._prepare_task = task  # type: ignore[attr-defined]
                    logger.info("Delayed TTS pre-connect triggered on first LLM text delta")

            full_text = full_text.strip()
            if not full_text:
                return ""

            # Check token again before TTS synthesis
            if token and token.cancelled:
                logger.info(f"TTS synthesis skipped for cancelled turn {turn_id}")
                return full_text

            tracker.mark("tts_first_segment_start", {"text_len": len(full_text), "mode": "single_request"})
            logger.info(
                "Streaming dialog collected full text for single TTS request (%s chars): %r",
                len(full_text),
                full_text[:80],
            )
            await self._speak_single_request(full_text, streaming_dialog=True, turn_id=turn_id, token=token)
            return full_text

        full_text = ""
        audio_chunks: list[npt.NDArray[np.int16]] = []
        first_chunk_queued = False
        first_segment_started = False
        barge_in_started = False
        segment_queue: asyncio.Queue[str | None] = asyncio.Queue()
        chunker = SentenceChunker()

        # Delayed pre-connect for segment mode: trigger now since produce/consume run in parallel
        if hasattr(self.tts, "prepare_stream"):
            # Mark as preparing BEFORE starting task so synthesize can wait
            self.tts._preparing = True  # type: ignore[attr-defined]
            task = asyncio.create_task(self.tts.prepare_stream())  # type: ignore[attr-defined]
            self.tts._prepare_task = task  # type: ignore[attr-defined]
            logger.info("Delayed TTS pre-connect triggered for segment mode")

        async def produce_segments() -> None:
            """Produce segments from text deltas, checking for cancellation."""
            nonlocal full_text
            try:
                async for text_delta in text_chunks:
                    # Check for cancellation during text collection
                    if token and token.cancelled:
                        chunker.interrupt()
                        logger.info(f"Text chunker interrupted at turn {turn_id}")
                        break
                    full_text += text_delta
                    for segment in chunker.push(text_delta):
                        await segment_queue.put(segment)

                # Only flush if not cancelled
                if not (token and token.cancelled):
                    final_segment = chunker.flush()
                    if final_segment:
                        await segment_queue.put(final_segment)
            finally:
                await segment_queue.put(None)

        async def consume_segments() -> None:
            """Consume segments and synthesize TTS, checking for cancellation."""
            nonlocal first_chunk_queued, first_segment_started, barge_in_started
            segment_index = 0
            try:
                while True:
                    segment = await segment_queue.get()
                    if segment is None:
                        break

                    # Check token before starting TTS for this segment
                    if token and token.cancelled:
                        logger.info(f"TTS synthesis cancelled at turn {turn_id}, segment {segment_index}")
                        continue  # Skip this segment

                    segment_index += 1
                    if not first_segment_started:
                        first_segment_started = True
                        tracker.mark("tts_first_segment_start", {"text_len": len(segment)})

                    logger.debug("Streaming TTS segment %s: %r", segment_index, segment)

                    # Synthesize this segment
                    async for chunk in self.tts.synthesize(segment):
                        # Check token during synthesis
                        if token and token.cancelled:
                            logger.debug(f"TTS chunk dropped for cancelled turn {turn_id}")
                            continue

                        audio_array = np.frombuffer(chunk, dtype=np.int16)
                        audio_chunks.append(audio_array)

                        # Put with generation=turn_id for playback isolation (R2)
                        self.playback.put_audio(audio_array, generation=turn_id)
                        self.playback.put_wobbler(chunk, generation=turn_id)

                        if not first_chunk_queued:
                            first_chunk_queued = True
                            barge_in_started = True
                            tracker.mark("audio_playback_started")
                            tracker.mark("streaming_dialog_first_audio")
                            logger.info(
                                "First streamed audio chunk playing (turn_id=%s) while LLM/TTS continue",
                                generation,
                            )
                            # Start barge-in monitor when first audio chunk is queued (R7)
                            if self.barge_in_start_callback is not None:
                                self.barge_in_start_callback()
            except Exception:
                # On error, stop barge-in monitor immediately
                if barge_in_started and self.barge_in_stop_callback is not None:
                    self.barge_in_stop_callback()
                raise

        await asyncio.gather(produce_segments(), consume_segments())

        logger.info("Streaming speech complete: generated %s audio chunks", len(audio_chunks))
        completion = self.playback.signal_end_of_turn(caller_turn_id=generation)

        if audio_chunks:
            total_samples = sum(len(chunk) for chunk in audio_chunks)
            duration_seconds = total_samples / self.tts.sample_rate
            tracker.mark("tts_audio_queued", {"duration_s": round(duration_seconds, 2)})
            if self.return_after_tts_queued:
                logger.info(
                    "Audio queued for playback (audio=%.1fs); returning before local playback drain",
                    duration_seconds,
                )
                # Stop barge-in monitor before returning (caller will handle playback)
                if barge_in_started and self.barge_in_stop_callback is not None:
                    self.barge_in_stop_callback()
                return full_text
            logger.info(f"Waiting {duration_seconds:.1f}s for playback to complete...")
            await self._wait_for_playback_completion(completion, duration_seconds)

        logger.info("Playback complete (streaming speech)")
        tracker.mark("playback_complete")
        # Stop barge-in monitor when playback completes (R7)
        if barge_in_started and self.barge_in_stop_callback is not None:
            self.barge_in_stop_callback()
        # 注意：print_summary() 在 gradio_app.py 统一调用
        return full_text

    async def _wait_for_playback_completion(
        self,
        completion: tuple[int, threading.Event] | None,
        duration_seconds: float,
    ) -> None:
        """Wait for playback drain, but unblock immediately if interrupted."""
        if completion is None:
            await asyncio.sleep(duration_seconds + 0.5)
            return

        _turn_id, event = completion
        timeout = max(duration_seconds + 1.0, 2.0)
        try:
            await asyncio.wait_for(asyncio.to_thread(event.wait), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("Timed out waiting for playback completion after %.1fs", timeout)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def split_into_sentences(text: str, min_length: int = 8) -> list[str]:
    """Split text into sentence-like chunks for streaming TTS.

    Splits on: . ! ? , ; — (but keeps punctuation with the sentence)

    Args:
        text: Text to split
        min_length: Minimum characters per segment (default 8)

    Returns:
        List of text segments, each at least min_length characters (except possibly the last)

    """
    pattern = r"([.!?,;—]\s+)"
    parts = re.split(pattern, text)

    raw_sentences: list[str] = []
    current = ""
    for part in parts:
        current += part
        if re.match(pattern, part):
            if current.strip():
                raw_sentences.append(current.strip())
            current = ""

    if current.strip():
        raw_sentences.append(current.strip())

    if not raw_sentences:
        return [text]

    merged_sentences: list[str] = []
    accumulator = ""

    for sentence in raw_sentences:
        if accumulator:
            accumulator += " " + sentence
        else:
            accumulator = sentence

        if len(accumulator) >= min_length:
            merged_sentences.append(accumulator)
            accumulator = ""

    if accumulator:
        if merged_sentences and len(merged_sentences[-1]) < min_length * 2:
            merged_sentences[-1] += " " + accumulator
        else:
            merged_sentences.append(accumulator)

    return merged_sentences if merged_sentences else [text]
