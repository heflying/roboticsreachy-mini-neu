"""Console mode for cascade pipeline using VAD-based speech detection."""

from __future__ import annotations
import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import sounddevice as sd

from reachy_mini_conversation_app.cascade.asr.audio_utils import pcm_to_wav
from reachy_mini_conversation_app.cascade.timing import tracker
from reachy_mini_conversation_app.cascade.ui.audio_playback import AudioPlaybackSystem
from reachy_mini_conversation_app.cascade.vad import (
    VAD_CHUNK_SIZE,
    SILERO_SAMPLE_RATE,
    VADEvent,
    VADState,
    SileroVAD,
    VADStateMachine,
)


if TYPE_CHECKING:
    from reachy_mini import ReachyMini
    from reachy_mini_conversation_app.cascade.handler import CascadeHandler

logger = logging.getLogger(__name__)


class CascadeLocalStream:
    """Console stream for cascade pipeline using VAD-based speech detection."""

    def __init__(self, handler: CascadeHandler, robot: ReachyMini) -> None:
        """Initialize the console stream."""
        self.handler = handler
        self._robot = robot

        # VAD state machine
        from reachy_mini_conversation_app.cascade.config import get_config

        cfg = get_config()
        vad = SileroVAD(
            backend=cfg.vad_backend,
            threshold=cfg.vad_threshold,
            min_speech_duration_ms=cfg.vad_min_speech_duration_ms,
            min_silence_duration_ms=700,  # console uses longer silence threshold
        )
        self._vad_sm = VADStateMachine(vad)

        # State
        self._stop_event = asyncio.Event()
        self._tasks: list[asyncio.Task[Any]] = []
        self._reply_tasks: set[asyncio.Task[None]] = set()
        self._playback_active = False
        self._reply_processing_active = False

        self._playback = AudioPlaybackSystem(
            robot=robot,
            head_wobbler=self.handler.deps.head_wobbler,
            tts_sample_rate=self.handler.tts.sample_rate,
        )
        self.handler.init_turn_controller(self._playback)

        # Wire speech output so handler plays audio through robot speaker
        from reachy_mini_conversation_app.cascade.speech_output import GradioSpeechOutput

        self.handler.speech_output = GradioSpeechOutput(
            tts=self.handler.tts,
            playback=self._playback,
            barge_in_start_callback=self._start_barge_in_monitor,
            barge_in_stop_callback=self._stop_barge_in_monitor,
        )
        logger.info("CascadeLocalStream initialized")

    def launch(self) -> None:
        """Start the console stream and run the async processing loops."""
        self._stop_event.clear()

        logger.info("Starting media recording...")
        self._robot.media.start_recording()
        time.sleep(1)  # Give pipelines time to start

        # Log which mic we'll use (system default, not robot USB device)
        default_dev = sd.query_devices(kind="input")
        logger.info(
            f"Mic input: '{default_dev['name']}' (system default, "
            f"{default_dev['default_samplerate']:.0f} Hz)"
        )
        if self._playback.use_robot_media:
            output_sr = self._robot.media.get_output_audio_samplerate()
            logger.info(f"Speaker output: {output_sr} Hz (robot)")
        else:
            output_dev = sd.query_devices(kind="output")
            logger.info(
                f"Speaker output: '{output_dev['name']}' (system default, "
                f"{output_dev['default_samplerate']:.0f} Hz)"
            )

        logger.info("Console mode ready. Speak to interact with the robot. Press Ctrl+C to stop.")
        asyncio.run(self._main_loop())

    async def _main_loop(self) -> None:
        """Run record and play loops concurrently."""
        self._tasks = [
            asyncio.create_task(self._record_loop(), name="cascade-record-loop"),
        ]
        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            logger.info("Tasks cancelled during shutdown")

    async def _record_loop(self) -> None:
        """Read mic audio from system default device and process through VAD."""
        logger.info(f"Recording from system default mic at {SILERO_SAMPLE_RATE} Hz, listening...")

        stream = sd.InputStream(
            samplerate=SILERO_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=VAD_CHUNK_SIZE,
        )
        stream.start()

        try:
            while not self._stop_event.is_set():
                # Read exactly one VAD chunk (512 samples = 32ms at 16kHz)
                audio_frame, overflowed = stream.read(VAD_CHUNK_SIZE)
                if overflowed:
                    logger.debug("Audio input overflowed")

                audio_int16 = audio_frame[:, 0].astype(np.int16)  # (samples, 1) → (samples,)
                await self._process_vad(audio_int16)
                await asyncio.sleep(0)
        finally:
            stream.stop()
            stream.close()

    async def _process_vad(self, audio_chunk: npt.NDArray[np.int16]) -> None:
        """Process audio chunk through VAD state machine."""
        streaming = self.handler.is_streaming_asr
        event = self._vad_sm.process_chunk(audio_chunk)

        if event == VADEvent.SPEECH_STARTED:
            turn_controller = self.handler.turn_controller
            current_token = turn_controller.current_token if turn_controller is not None else None
            should_interrupt = (
                (self._playback_active or self._reply_processing_active)
                and current_token is not None
                and not current_token.cancelled
            )
            if should_interrupt:
                self._playback_active = False
                self.handler.handle_barge_in()
            if streaming:
                await self.handler.process_audio_streaming_start()
                for chunk in self._vad_sm.speech_chunks:
                    wav_bytes = pcm_to_wav(chunk.tobytes(), SILERO_SAMPLE_RATE)
                    await self.handler.process_audio_streaming_chunk(wav_bytes)

        elif event == VADEvent.SPEECH_ENDED:
            # Start latency tracking from speech end
            tracker.reset("vad_speech_end")
            tracker.mark("vad_speech_end")

            if streaming:
                audio_data = np.concatenate(self._vad_sm.speech_chunks)
                duration = len(audio_data) / SILERO_SAMPLE_RATE
                tracker.mark("recording_captured", {"duration_s": round(duration, 2)})
                self._schedule_reply_task(
                    self._finalize_streaming_turn(),
                    label="streaming-turn",
                )
            else:
                await self._schedule_manual_turn()

            self._vad_sm.finish_processing()
            logger.info("Listening...")

        elif event == VADEvent.SENTENCE_PAUSE and streaming:
            self.handler.on_sentence_pause()

        elif self._vad_sm.state == VADState.RECORDING and streaming:
            # Mid-recording: stream current chunk to ASR
            wav_bytes = pcm_to_wav(audio_chunk.tobytes(), SILERO_SAMPLE_RATE)
            await self.handler.process_audio_streaming_chunk(wav_bytes)

    async def _schedule_manual_turn(self) -> None:
        """Copy the current utterance and process it in the background."""
        if not self._vad_sm.speech_chunks:
            logger.warning("Empty audio buffer, skipping")
            return

        audio_data = np.concatenate(self._vad_sm.speech_chunks)
        duration = len(audio_data) / SILERO_SAMPLE_RATE
        logger.info(f"Processing {len(audio_data)} samples ({duration:.2f}s)")
        tracker.mark("recording_captured", {"duration_s": round(duration, 2)})

        wav_bytes = pcm_to_wav(audio_data.tobytes(), SILERO_SAMPLE_RATE)
        self._schedule_reply_task(
            self._process_recorded_audio(wav_bytes),
            label="manual-turn",
        )

    async def _process_recorded_audio(self, wav_bytes: bytes) -> None:
        """Process recorded audio through the cascade pipeline."""
        self._reply_processing_active = True
        try:
            logger.info("Transcribing...")
            turn = await self.handler.process_audio_manual(wav_bytes)
            transcript = turn.transcript
            if transcript:
                logger.info(f"User: {transcript}")
            else:
                logger.info("No speech detected")

            # Print latency summary for this turn
            tracker.print_summary()
            tracker.next_turn()
        finally:
            self._reply_processing_active = False

    async def _finalize_streaming_turn(self) -> None:
        """Finalize a streaming-ASR turn in the background."""
        self._reply_processing_active = True
        try:
            turn = await self.handler.process_audio_streaming_end()
            transcript = turn.transcript
            if transcript:
                logger.info(f"User: {transcript}")
            else:
                logger.info("No speech detected")
            tracker.print_summary()
            tracker.next_turn()
        finally:
            self._reply_processing_active = False

    def _schedule_reply_task(self, coro: Any, *, label: str) -> None:
        """Run one reply-processing coroutine in the background."""
        task = asyncio.create_task(coro, name=label)
        self._reply_tasks.add(task)

        def _cleanup(done_task: asyncio.Task[None]) -> None:
            self._reply_tasks.discard(done_task)
            if done_task.cancelled():
                return
            exc = done_task.exception()
            if exc is not None:
                logger.exception("Reply task failed: %s", label, exc_info=exc)

        task.add_done_callback(_cleanup)

    def _start_barge_in_monitor(self) -> None:
        """Mark playback as active so new user speech can interrupt it."""
        self._playback_active = True

    def _stop_barge_in_monitor(self) -> None:
        """Mark playback as inactive after drain or interruption."""
        self._playback_active = False

    def close(self) -> None:
        """Stop the stream and cleanup."""
        logger.info("Stopping CascadeLocalStream...")

        # Stop media first
        try:
            self._robot.media.stop_recording()
        except Exception as e:
            logger.debug(f"Error stopping recording: {e}")

        # Signal async loops to stop
        self._stop_event.set()

        # Cancel tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        for task in list(self._reply_tasks):
            if not task.done():
                task.cancel()

        self._playback.close()
        logger.info("CascadeLocalStream stopped")
