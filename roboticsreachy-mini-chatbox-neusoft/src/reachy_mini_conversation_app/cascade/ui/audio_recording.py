"""Audio recording for VAD-based continuous recording."""

from __future__ import annotations
import os
import logging
import threading
import time
from enum import Enum
from queue import Full, Empty, Queue
from typing import TYPE_CHECKING, Any, Callable
from dataclasses import dataclass, field

import numpy as np
import sounddevice as sd

from reachy_mini_conversation_app.cascade.vad import VADEvent, VADStateMachine
from reachy_mini_conversation_app.cascade.asr.audio_utils import pcm_to_wav


if TYPE_CHECKING:
    from reachy_mini_conversation_app.cascade.vad import SileroVAD

logger = logging.getLogger(__name__)


class ContinuousState(Enum):
    """State machine for continuous VAD-based recording."""

    IDLE = "idle"
    LISTENING = "listening"
    RECORDING = "recording"
    PROCESSING = "processing"


@dataclass
class StreamingASRCallbacks:
    """Callbacks for streaming ASR integration.

    Allows decoupling recording from ASR provider by injecting callbacks.
    """

    on_start: Callable[[], None]
    """Called when recording starts to initialize streaming session."""

    on_chunk: Callable[[bytes], None]
    """Called for each audio chunk (as WAV bytes) during recording."""

    # Track whether on_start succeeded
    _session_started: bool = field(default=False, init=False)

    def mark_started(self) -> None:
        """Mark the streaming session as successfully started."""
        self._session_started = True

    def mark_failed(self) -> None:
        """Mark the streaming session as failed to start."""
        self._session_started = False

    def is_ready(self) -> bool:
        """Check if the session is ready to receive chunks."""
        return self._session_started



class ContinuousVADRecorder:
    """VAD-based continuous recording mode.

    Automatically detects speech start/end using Silero VAD.

    Barge-in Detection:
    - When playback is active, VAD can detect user speech and trigger interruption
    - Uses debounce to prevent rapid repeated triggering
    - Lifecycle: enable when playback starts, disable when playback ends
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        streaming_asr_callbacks: StreamingASRCallbacks | None = None,
        on_speech_captured: Callable[[bytes], Any] | None = None,
        on_sentence_pause: Callable[[], Any] | None = None,
        vad_threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 700,
        sentence_pause_threshold_ms: int = 200,
        is_dialogue_active: Callable[[], bool] | None = None,
        on_audio_frame: Callable[[npt.NDArray[np.int16]], None] | None = None,
    ) -> None:
        """Initialize VAD recorder.

        Args:
            sample_rate: Recording sample rate (default 16kHz)
            streaming_asr_callbacks: Optional callbacks for streaming ASR
            on_speech_captured: Callback when complete utterance is captured (receives WAV bytes)
            on_sentence_pause: Callback when sentence pause is detected (for LLM warmup)
            vad_threshold: VAD detection threshold (0-1)
            min_speech_duration_ms: Minimum speech duration to trigger detection
            min_silence_duration_ms: Silence duration to end speech segment
            sentence_pause_threshold_ms: Silence duration to trigger sentence pause
            is_dialogue_active: Callback that returns True when robot is in DIALOGUE state.
                When provided, VAD processing only runs in DIALOGUE state; in non-dialogue
                states, audio frames are dispatched to on_audio_frame instead.
            on_audio_frame: Callback for raw audio frames during non-dialogue states.
                Receives numpy int16 array at sample_rate Hz. Used by proactive modules
                for audio analysis (fall detection, cough detection, etc.).

        """
        self.sample_rate = sample_rate
        self.streaming_callbacks = streaming_asr_callbacks
        self.on_speech_captured = on_speech_captured
        self.on_sentence_pause = on_sentence_pause
        self.vad_threshold = vad_threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.sentence_pause_threshold_ms = sentence_pause_threshold_ms
        self.is_dialogue_active = is_dialogue_active
        self.on_audio_frame = on_audio_frame

        self._active = False
        self._vad_sm: VADStateMachine | None = None
        self._continuous_thread: threading.Thread | None = None
        self._vad: SileroVAD | None = None

        # Barge-in detection state (Task 9: VAD Barge-in Trigger)
        self._barge_in_callback: Callable[[], Any] | None = None
        self._barge_in_detection_enabled: bool = False
        self._barge_in_last_fire_time: float = 0.0
        self._barge_in_debounce_s: float = 2.0  # Minimum interval between fires

        # Barge-in mode: use lower threshold during playback to better detect user voice
        # over speaker output (echo problem workaround)
        self._normal_vad_threshold: float = vad_threshold
        self._barge_in_vad_threshold: float = max(0.5, vad_threshold - 0.1)  # Conservative during playback

    def set_barge_in_callback(self, callback: Callable[[], Any] | None) -> None:
        """Set the barge-in callback.

        Args:
            callback: Function to call when barge-in is detected (user starts speaking during playback).
                      Pass None to clear the callback.

        """
        self._barge_in_callback = callback
        logger.debug(f"Barge-in callback set: {callback is not None}")

    def enable_barge_in_detection(self, enabled: bool) -> None:
        """Enable or disable barge-in detection.

        When enabled, uses a lower VAD threshold to better detect user speech
        over speaker output (workaround for echo problem during playback).

        Args:
            enabled: True to enable detection during playback, False to disable.

        """
        self._barge_in_detection_enabled = enabled
        state_str = "enabled" if enabled else "disabled"

        # Dynamically adjust VAD threshold for barge-in detection
        # Lower threshold during playback to detect user voice over speaker echo
        if self._vad is not None:
            if enabled:
                old_threshold = self._vad.threshold
                self._vad.threshold = self._barge_in_vad_threshold
                logger.info(f"Barge-in detection {state_str} (VAD threshold: {old_threshold:.2f} → {self._vad.threshold:.2f})")
            else:
                self._vad.threshold = self._normal_vad_threshold
                logger.info(f"Barge-in detection {state_str} (VAD threshold restored: {self._vad.threshold:.2f})")
        else:
            logger.info(f"Barge-in detection {state_str}")

    def _on_speech_start_detected(self) -> None:
        """Called when VAD detects speech start.

        If barge-in detection is enabled, triggers the callback.
        Includes debounce to prevent rapid repeated triggering.

        This method is called from the VAD processing loop (potentially from a
        background thread), so the callback should be thread-safe.
        """
        if not self._barge_in_detection_enabled:
            return

        if self._barge_in_callback is None:
            return

        # Debounce check
        now = time.monotonic()
        if now - self._barge_in_last_fire_time < self._barge_in_debounce_s:
            logger.debug("Barge-in debounced, skipping")
            return

        self._barge_in_last_fire_time = now
        logger.info("🔊 [BARGE-IN][gradio] user speech detected during playback - interrupting")

        # Fire callback (may be called from VAD thread)
        try:
            self._barge_in_callback()
        except Exception as e:
            logger.warning(f"Barge-in callback error: {e}")

    @staticmethod
    def _hostapi_name(device: dict[str, Any], hostapis: list[dict[str, Any]]) -> str:
        """Return the host API name for a sounddevice device dictionary."""
        hostapi_index = int(device.get("hostapi", -1))
        if 0 <= hostapi_index < len(hostapis):
            return str(hostapis[hostapi_index].get("name", ""))
        return ""

    def _input_device_candidates(self) -> list[int | None]:
        """Return input devices to try, ordered for Windows microphone reliability."""
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        candidates: list[int | None] = []

        env_device = os.getenv("CASCADE_INPUT_DEVICE")
        if env_device:
            for idx, dev in enumerate(devices):
                if int(dev.get("max_input_channels", 0)) <= 0:
                    continue
                if env_device.isdigit() and idx == int(env_device):
                    candidates.append(idx)
                elif env_device.lower() in str(dev.get("name", "")).lower():
                    candidates.append(idx)

        default_index = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else None
        if isinstance(default_index, int) and default_index >= 0:
            candidates.append(default_index)
        candidates.append(None)

        host_priority = {
            "wasapi": 0,
            "windows wdm-ks": 1,
            "wdm-ks": 1,
            "directsound": 2,
            "mme": 3,
        }

        def sort_key(item: tuple[int, dict[str, Any]]) -> tuple[int, str]:
            _, dev = item
            host = self._hostapi_name(dev, hostapis).lower()
            priority = 10
            for marker, value in host_priority.items():
                if marker in host:
                    priority = value
                    break
            return priority, str(dev.get("name", ""))

        input_devices = [
            (idx, dev)
            for idx, dev in enumerate(devices)
            if int(dev.get("max_input_channels", 0)) > 0
        ]
        for idx, _dev in sorted(input_devices, key=sort_key):
            candidates.append(idx)

        deduped: list[int | None] = []
        for candidate in candidates:
            if candidate not in deduped:
                deduped.append(candidate)
        return deduped

    def _open_input_stream(
        self,
        preferred_rate: int,
        vad_chunk_size: int,
        silero_sample_rate: int,
        callback: Callable[[Any, int, Any, Any], None],
    ) -> tuple[sd.InputStream, int, int]:
        """Open a microphone stream, trying alternate Windows host APIs when needed."""
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        errors: list[str] = []

        for device_index in self._input_device_candidates():
            device = sd.query_devices(kind="input") if device_index is None else devices[device_index]
            device_name = str(device.get("name", "system default"))
            host_name = self._hostapi_name(device, hostapis) if device_index is not None else "system default"
            default_rate = int(device.get("default_samplerate") or preferred_rate)
            rates: list[int] = []
            for rate in (preferred_rate, default_rate):
                if rate not in rates:
                    rates.append(rate)

            for record_rate in rates:
                sized_block = max(1, int(vad_chunk_size * record_rate / silero_sample_rate))
                for blocksize in (sized_block, 0):
                    try:
                        stream = sd.InputStream(
                            device=device_index,
                            channels=1,
                            samplerate=record_rate,
                            dtype=np.int16,
                            blocksize=blocksize,
                            latency="high",
                            callback=callback,
                        )
                        stream.start()
                        logger.info(
                            "Mic stream opened: device=%s host=%s rate=%sHz blocksize=%s",
                            device_name,
                            host_name,
                            record_rate,
                            blocksize,
                        )
                        return stream, record_rate, sized_block
                    except sd.PortAudioError as exc:
                        errors.append(
                            f"device={device_index} name={device_name!r} host={host_name!r} "
                            f"rate={record_rate} blocksize={blocksize}: {exc}"
                        )

        logger.error("Failed to open any microphone input. Tried:\n%s", "\n".join(errors[-20:]))
        raise RuntimeError("Could not open a microphone input stream; see logs for tried devices.")

    @property
    def state(self) -> ContinuousState:
        """Current VAD state."""
        if not self._active:
            return ContinuousState.IDLE
        if self._vad_sm is None:
            return ContinuousState.IDLE
        return ContinuousState(self._vad_sm.state.value)

    @property
    def is_active(self) -> bool:
        """Whether continuous mode is active."""
        return self._active

    def start(self) -> str:
        """Start continuous VAD-based recording mode.

        Returns:
            Status message

        """
        if self._active:
            return "Already in continuous mode"

        # Initialize VAD lazily (avoids ~1-2s model load at startup)
        if self._vad is None:
            logger.info("Initializing Silero VAD...")
            from reachy_mini_conversation_app.cascade.vad import SileroVAD
            from reachy_mini_conversation_app.cascade.config import get_config

            cfg = get_config()
            self._vad = SileroVAD(
                backend=cfg.vad_backend,
                threshold=cfg.vad_threshold,
                min_speech_duration_ms=cfg.vad_min_speech_duration_ms,
                min_silence_duration_ms=cfg.vad_min_silence_duration_ms,
                sentence_pause_threshold_ms=cfg.sentence_pause_threshold_ms,
            )

        self._vad_sm = VADStateMachine(self._vad)
        self._active = True

        # Start continuous recording thread
        self._continuous_thread = threading.Thread(target=self._continuous_record_loop, daemon=True)
        self._continuous_thread.start()

        logger.info("Continuous mode started")
        return "Listening... (VAD active)"

    def reset_vad(self) -> None:
        """Reset VAD state machine for entering DIALOGUE state.

        Clears speech_chunks buffer, resets Silero hidden states, and
        resets the VADStateMachine back to LISTENING. Called by the
        state machine loop when transitioning into DIALOGUE state.

        Thread-safe: can be called from any thread (VAD thread reads
        _vad_sm on next iteration; Python object assignment is atomic).
        """
        if self._vad_sm is not None and self._vad is not None:
            self._vad.reset()
            self._vad_sm = VADStateMachine(self._vad)
            logger.info("[VAD] State machine reset for DIALOGUE entry")

    def stop(self) -> str:
        """Stop continuous VAD-based recording mode.

        Returns:
            Status message

        """
        if not self._active:
            return "Not in continuous mode"

        self._active = False

        # Wait for thread to finish
        if self._continuous_thread:
            self._continuous_thread.join(timeout=2.0)
            self._continuous_thread = None

        # Reset VAD state
        if self._vad:
            self._vad.reset()
        self._vad_sm = None

        logger.info("Continuous mode stopped")
        return "Continuous mode stopped"

    def _continuous_record_loop(self) -> None:
        """Continuous recording loop with frame dispatch based on robot state.

        When is_dialogue_active callback is provided:
        - DIALOGUE state: audio frames flow through VAD → ASR → LLM pipeline
        - Non-dialogue state: audio frames are dispatched to on_audio_frame
          callback (for proactive module audio analysis).

        When is_dialogue_active is NOT provided (legacy behavior):
        - All frames always flow through VAD regardless of state.
        """
        from reachy_mini_conversation_app.cascade.vad import VAD_CHUNK_SIZE, SILERO_SAMPLE_RATE
        from reachy_mini_conversation_app.cascade.timing import tracker

        assert self._vad_sm is not None

        # Log which mic we'll use (system default)
        default_dev = sd.query_devices(kind="input")
        logger.info(
            f"Mic input: '{default_dev['name']}' (system default, "
            f"{default_dev['default_samplerate']:.0f} Hz)"
        )

        logger.info("Continuous mode started - listening for speech...")
        audio_queue: Queue[tuple[npt.NDArray[np.int16], bool]] = Queue(maxsize=100)
        processing_done = threading.Event()

        def audio_callback(indata: npt.NDArray[np.int16], frames: int, time_info: Any, status: Any) -> None:
            """Collect microphone chunks from PortAudio's callback thread."""
            item = (indata.copy(), bool(status))
            try:
                audio_queue.put_nowait(item)
            except Full:
                try:
                    audio_queue.get_nowait()
                except Empty:
                    pass
                try:
                    audio_queue.put_nowait(item)
                except Full:
                    pass

        try:
            stream, record_rate, record_chunk_samples = self._open_input_stream(
                preferred_rate=self.sample_rate,
                vad_chunk_size=VAD_CHUNK_SIZE,
                silero_sample_rate=SILERO_SAMPLE_RATE,
                callback=audio_callback,
            )
            try:
                while self._active:
                    # ── Frame dispatch: decide where audio goes ──
                    in_dialogue = self.is_dialogue_active is not None and self.is_dialogue_active()

                    # Check processing_done transition (only relevant in dialogue mode)
                    if in_dialogue and self._vad_sm.state.value == ContinuousState.PROCESSING.value and processing_done.is_set():
                        processing_done.clear()
                        self._vad_sm.finish_processing()
                        tracker.mark("vad_ready_for_next_utterance")
                        speech_end_to_ready = tracker.get_duration("vad_speech_end", "vad_ready_for_next_utterance")
                        if speech_end_to_ready is not None:
                            logger.info("PERCEIVED: Speech End -> VAD Ready %.1fms", speech_end_to_ready)
                        logger.info("VAD: Ready for next utterance")

                    try:
                        data, overflowed = audio_queue.get(timeout=0.2)
                    except Empty:
                        continue
                    if overflowed:
                        logger.warning("Audio buffer overflowed in continuous mode")

                    # Resample to 16kHz for VAD if needed
                    if record_rate != SILERO_SAMPLE_RATE:
                        import librosa

                        audio_float = data.flatten().astype(np.float32) / 32768.0
                        audio_resampled = librosa.resample(
                            audio_float,
                            orig_sr=record_rate,
                            target_sr=SILERO_SAMPLE_RATE,
                        )
                        vad_audio = (audio_resampled * 32768).astype(np.int16)
                    else:
                        vad_audio = data.flatten()

                    # ── Non-dialogue mode: dispatch raw frames to proactive ──
                    if not in_dialogue:
                        if self.on_audio_frame:
                            try:
                                self.on_audio_frame(vad_audio)
                            except Exception as e:
                                logger.warning(f"on_audio_frame callback error: {e}")
                        continue  # Skip VAD processing entirely

                    # ── Dialogue mode: feed frames through VAD state machine ──
                    event = self._vad_sm.process_chunk(vad_audio)

                    if event == VADEvent.SPEECH_STARTED:
                        tracker.mark("vad_speech_start")

                        # Trigger barge-in callback if detection is enabled
                        self._on_speech_start_detected()

                        if self.streaming_callbacks:
                            try:
                                self.streaming_callbacks.on_start()
                                self.streaming_callbacks.mark_started()
                                merged = np.concatenate(self._vad_sm.speech_chunks)
                                self.streaming_callbacks.on_chunk(
                                    pcm_to_wav(merged.tobytes(), SILERO_SAMPLE_RATE)
                                )
                            except Exception as e:
                                logger.error(f"Streaming ASR on_start failed: {e}")
                                self.streaming_callbacks.mark_failed()

                    elif event == VADEvent.SENTENCE_PAUSE:
                        logger.info("[VAD] Sentence pause event received")
                        if self.on_sentence_pause:
                            try:
                                self.on_sentence_pause()
                            except Exception as e:
                                logger.warning(f"Sentence pause callback failed: {e}")

                    elif event == VADEvent.SPEECH_ENDED:
                        tracker.mark("vad_speech_end")

                        audio_data = np.concatenate(self._vad_sm.speech_chunks)
                        duration = len(audio_data) / SILERO_SAMPLE_RATE
                        logger.info(f"[VAD] Speech ended - captured {duration:.2f}s, {len(self._vad_sm.speech_chunks)} chunks")
                        tracker.mark("recording_captured", {"duration_s": round(duration, 2)})

                        wav_bytes = pcm_to_wav(audio_data.tobytes(), SILERO_SAMPLE_RATE)
                        if self.on_speech_captured:
                            callback_result = self.on_speech_captured(wav_bytes)
                            if hasattr(callback_result, "add_done_callback"):
                                callback_result.add_done_callback(
                                    lambda _future: logger.info("[BARGE-IN][gradio] background reply task completed")
                                )
                            logger.info("[BARGE-IN][gradio] speech dispatched; VAD immediately returns to listening")
                            processing_done.set()
                        else:
                            processing_done.set()

                    elif self._vad_sm.state.value == ContinuousState.RECORDING.value:
                        if self.streaming_callbacks and self.streaming_callbacks.is_ready():
                            self.streaming_callbacks.on_chunk(
                                pcm_to_wav(vad_audio.tobytes(), SILERO_SAMPLE_RATE)
                            )
            finally:
                stream.stop()
                stream.close()

        except Exception as e:
            logger.exception(f"Error in continuous recording loop: {e}")
        finally:
            logger.info("Continuous mode stopped")
