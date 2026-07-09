"""Test file mode for cascade pipeline — feeds text utterances through TTS→ASR→LLM→TTS.

Reads utterances from a text file, synthesizes them to audio via TTS, and feeds
the audio through the full cascade pipeline without human interaction.
"""

from __future__ import annotations
import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Any
from pathlib import Path

import numpy as np
import sounddevice as sd
import numpy.typing as npt

from reachy_mini_conversation_app.cascade.timing import tracker
from reachy_mini_conversation_app.cascade.speech_output import ConsoleSpeechOutput
from reachy_mini_conversation_app.cascade.asr.audio_utils import pcm_to_wav
from reachy_mini_conversation_app.cascade.provider_factory import init_tts_provider


if TYPE_CHECKING:
    from reachy_mini import ReachyMini
    from reachy_mini_conversation_app.cascade.handler import CascadeHandler
    from reachy_mini_conversation_app.cascade.turn_result import TurnResult

logger = logging.getLogger(__name__)

DEFAULT_DELAY_BETWEEN_TURNS = 2.0
MOVE_WAIT_POLL_INTERVAL = 0.1
MOVE_WAIT_TIMEOUT = 30.0
INPUT_VOICE = "af_heart"

# Chunk size for simulated real-time streaming (512 samples @ 16kHz = 32ms,
# matches the VAD chunk size used in console mode).
STREAM_CHUNK_SAMPLES = 512
STREAM_SAMPLE_RATE = 16000


class CascadeTestStream:
    """Feeds text utterances from a file through the full cascade pipeline."""

    def __init__(self, handler: CascadeHandler, robot: ReachyMini, test_file: str) -> None:
        """Initialize test stream with a separate TTS for input audio generation."""
        self.handler = handler
        self._robot = robot
        self._test_file = Path(test_file)
        self._delay = DEFAULT_DELAY_BETWEEN_TURNS

        # Separate TTS instance for generating "user voice" audio
        self._input_tts = init_tts_provider()
        sample_rate = self._input_tts.sample_rate

        # Audio buffer for callback-based sounddevice playback.
        # The callback pulls from this buffer at a steady rate, avoiding
        # gaps that occur when pushing chunks with blocking write().
        self._audio_buffer = bytearray()
        self._buffer_lock = threading.Lock()

        # Sounddevice output stream in callback mode — pulls audio smoothly
        self._sd_stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            callback=self._audio_callback,
        )

        # Wire speech output so handler plays audio through computer speakers
        self.handler.speech_output = ConsoleSpeechOutput(
            tts=self.handler.tts,
            head_wobbler=self.handler.deps.head_wobbler,
            playback_callback=self._queue_audio_chunk,
        )

        logger.info(f"CascadeTestStream initialized (file={self._test_file}, rate={sample_rate}Hz)")

    def _audio_callback(
        self, outdata: npt.NDArray[Any], frames: int, time_info: object, status: sd.CallbackFlags
    ) -> None:
        """Sounddevice callback — fills output buffer from our audio buffer."""
        bytes_needed = frames * 2  # int16 = 2 bytes per frame
        with self._buffer_lock:
            available = len(self._audio_buffer)
            if available >= bytes_needed:
                data = bytes(self._audio_buffer[:bytes_needed])
                del self._audio_buffer[:bytes_needed]
                outdata[:] = np.frombuffer(data, dtype=np.int16).reshape(-1, 1)
            elif available > 0:
                data = bytes(self._audio_buffer)
                self._audio_buffer.clear()
                frames_available = available // 2
                outdata[:frames_available] = np.frombuffer(data, dtype=np.int16).reshape(-1, 1)
                outdata[frames_available:] = 0
            else:
                outdata[:] = 0

    async def _queue_audio_chunk(self, audio_bytes: bytes) -> None:
        """Append audio chunk to the playback buffer (non-blocking)."""
        with self._buffer_lock:
            self._audio_buffer.extend(audio_bytes)

    def launch(self) -> None:
        """Start the test stream (blocking)."""
        self._sd_stream.start()
        logger.info("Test file mode ready. Running utterances...")
        asyncio.run(self._main_loop())

    async def _main_loop(self) -> None:
        """Process each utterance through the full pipeline."""
        utterances = self._load_utterances()
        logger.info(f"Loaded {len(utterances)} utterances from {self._test_file}")
        streaming = self.handler.is_streaming_asr

        for i, text in enumerate(utterances):
            logger.info(f"\n{'='*60}")
            logger.info(f"Utterance {i+1}/{len(utterances)}: \"{text}\"")
            logger.info(f"{'='*60}")

            # Generate PCM audio from text (before latency tracking starts)
            pcm_data = await self._synthesize_pcm(text)
            logger.info(f"Generated input audio ({len(pcm_data)} bytes PCM)")

            if streaming:
                turn = await self._process_streaming(pcm_data)
            else:
                turn = await self._process_manual(pcm_data)

            # Log results
            if turn.transcript:
                logger.info(f"ASR heard: \"{turn.transcript}\"")
            for item in turn.items:
                logger.info(f"  {item}")
            if turn.cost > 0:
                logger.info(f"Turn cost: ${turn.cost:.4f}")

            tracker.print_summary()

            # Wait for any queued movements to finish before next utterance
            await self._wait_for_moves()

            # Small delay between turns for natural pacing
            await asyncio.sleep(self._delay)

        logger.info(f"\nAll {len(utterances)} utterances processed.")

    async def _wait_for_moves(self) -> None:
        """Wait for the robot to finish queued movements (with timeout)."""
        mm = self.handler.deps.movement_manager
        if mm is None or not mm.has_pending_moves():
            return
        logger.info("Waiting for movements to complete...")
        elapsed = 0.0
        while mm.has_pending_moves() and elapsed < MOVE_WAIT_TIMEOUT:
            await asyncio.sleep(MOVE_WAIT_POLL_INTERVAL)
            elapsed += MOVE_WAIT_POLL_INTERVAL
        if elapsed >= MOVE_WAIT_TIMEOUT:
            logger.warning("Timed out waiting for movements after %.0fs", MOVE_WAIT_TIMEOUT)
        else:
            logger.info("Movements completed (waited %.1fs)", elapsed)

    async def _synthesize_pcm(self, text: str) -> bytes:
        """Synthesize text to raw PCM bytes."""
        pcm_chunks: list[bytes] = []
        async for chunk in self._input_tts.synthesize(text, voice=INPUT_VOICE):
            pcm_chunks.append(chunk)
        return b"".join(pcm_chunks)

    async def _play_and_wait(self, pcm_data: bytes) -> None:
        """Queue PCM for speaker playback and wait for it to finish."""
        with self._buffer_lock:
            self._audio_buffer.extend(pcm_data)
        duration = len(pcm_data) / (2 * self._input_tts.sample_rate)
        await asyncio.sleep(duration)

    async def _process_manual(self, pcm_data: bytes) -> TurnResult:
        """Non-streaming path: play user audio, then send full WAV to pipeline."""
        await self._play_and_wait(pcm_data)

        # Reset tracker after playback — simulates "user stopped speaking"
        duration = len(pcm_data) / (2 * self._input_tts.sample_rate)
        tracker.reset("vad_speech_end")
        tracker.mark("vad_speech_end")
        tracker.mark("recording_captured", {"duration_s": round(duration, 2)})
        wav_bytes = pcm_to_wav(pcm_data, self._input_tts.sample_rate)
        return await self.handler.process_audio_manual(wav_bytes)

    async def _process_streaming(self, pcm_data: bytes) -> TurnResult:
        """Streaming path: play user audio while feeding chunks to ASR in real time."""
        from scipy.signal import resample as scipy_resample

        # Resample from TTS rate to 16 kHz for streaming ASR
        tts_rate = self._input_tts.sample_rate
        audio = np.frombuffer(pcm_data, dtype=np.int16)
        if tts_rate != STREAM_SAMPLE_RATE:
            n_out = int(len(audio) * STREAM_SAMPLE_RATE / tts_rate)
            audio_16k = scipy_resample(audio, n_out).astype(np.int16)
        else:
            audio_16k = audio

        # Start speaker playback (non-blocking — sounddevice callback drains it)
        with self._buffer_lock:
            self._audio_buffer.extend(pcm_data)

        # Start streaming ASR session
        await self.handler.process_audio_streaming_start()

        # Feed chunks at real-time pace
        chunk_duration = STREAM_CHUNK_SAMPLES / STREAM_SAMPLE_RATE
        offset = 0
        while offset < len(audio_16k):
            chunk = audio_16k[offset : offset + STREAM_CHUNK_SAMPLES]
            wav_chunk = pcm_to_wav(chunk.tobytes(), STREAM_SAMPLE_RATE)
            await self.handler.process_audio_streaming_chunk(wav_chunk)
            offset += STREAM_CHUNK_SAMPLES
            await asyncio.sleep(chunk_duration)

        # Wait for speaker playback to finish (may already be done)
        total_duration = len(pcm_data) / (2 * tts_rate)
        elapsed_streaming = len(audio_16k) / STREAM_SAMPLE_RATE
        remaining = total_duration - elapsed_streaming
        if remaining > 0:
            await asyncio.sleep(remaining)

        # Reset tracker now — simulates "speech ended" (same as VAD flow)
        tracker.reset("vad_speech_end")
        tracker.mark("vad_speech_end")
        tracker.mark("recording_captured", {"duration_s": round(total_duration, 2)})

        # Finalize streaming ASR and run LLM pipeline
        turn = await self.handler.process_audio_streaming_end()
        if turn.transcript:
            logger.info(f"User: {turn.transcript}")
        return turn

    def _load_utterances(self) -> list[str]:
        """Load utterances from text file, stripping comments and blank lines."""
        lines = self._test_file.read_text(encoding="utf-8").splitlines()
        utterances = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                utterances.append(stripped)
        return utterances

    def close(self) -> None:
        """Stop sounddevice stream."""
        logger.info("Stopping CascadeTestStream...")
        self._sd_stream.stop()
        self._sd_stream.close()
        logger.info("CascadeTestStream stopped")
