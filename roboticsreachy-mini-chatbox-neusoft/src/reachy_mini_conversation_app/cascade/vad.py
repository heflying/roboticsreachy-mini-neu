"""Voice Activity Detection using Silero VAD.

Supports multiple inference backends (PyTorch, ONNX Runtime) via VADBackend Protocol.
SileroVAD facade delegates inference and adds timing logs.
"""

import logging
from enum import Enum, auto
from time import perf_counter
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


logger = logging.getLogger(__name__)

SILERO_SAMPLE_RATE = 16000
VAD_CHUNK_SIZE = 512


@runtime_checkable
class VADBackend(Protocol):
    """Protocol for VAD inference backends."""

    def get_speech_prob(self, audio: npt.NDArray[Any], sample_rate: int) -> float: ...
    def reset_states(self) -> None: ...


class SileroVADPyTorch:
    """Silero VAD inference using PyTorch."""

    def __init__(self) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Silero VAD PyTorch backend requires torch. Install with: "
                "uv sync --extra cascade_silero_vad"
            ) from exc
        self._torch = torch
        logger.info("Loading Silero VAD model (PyTorch)...")
        self.model, _utils = self._torch.hub.load(  # type: ignore[no-untyped-call]
            "snakers4/silero-vad",
            "silero_vad",
            trust_repo=True,
        )
        self.model.eval()

    def get_speech_prob(self, audio: npt.NDArray[Any], sample_rate: int = SILERO_SAMPLE_RATE) -> float:
        if sample_rate != SILERO_SAMPLE_RATE:
            raise ValueError(f"Silero VAD requires {SILERO_SAMPLE_RATE}Hz audio, got {sample_rate}Hz")

        if audio.dtype == np.int16:
            audio_float = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.float32:
            audio_float = audio
        else:
            audio_float = audio.astype(np.float32)

        tensor = self._torch.from_numpy(audio_float)
        with self._torch.no_grad():
            return self.model(tensor, sample_rate).item()  # type: ignore[no-any-return]

    def reset_states(self) -> None:
        self.model.reset_states()


class SileroVAD:
    """Voice Activity Detection facade with pluggable backends.

    Delegates inference to a VADBackend and adds timing logs
    for A/B performance comparison.
    """

    def __init__(
        self,
        backend: str = "onnx",
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 500,
        sentence_pause_threshold_ms: int = 200,
    ) -> None:
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.sentence_pause_threshold_ms = sentence_pause_threshold_ms
        self._backend_name = backend

        if backend == "onnx":
            from reachy_mini_conversation_app.cascade.vad_onnx import SileroVADOnnx

            self._backend: VADBackend = SileroVADOnnx()
        elif backend == "pytorch":
            self._backend = SileroVADPyTorch()
        else:
            raise ValueError(f"Unknown VAD backend: {backend!r}. Use 'onnx' or 'pytorch'.")

        self._speech_frames: float = 0
        self._silence_frames: float = 0
        self._is_speaking = False
        self._prob_log_count = 0
        logger.info(f"Silero VAD initialized (backend={backend}, threshold={threshold})")

    def get_speech_prob(self, audio: npt.NDArray[Any], sample_rate: int = SILERO_SAMPLE_RATE) -> float:
        t0 = perf_counter()
        prob = self._backend.get_speech_prob(audio, sample_rate)
        elapsed_ms = (perf_counter() - t0) * 1000
        self._prob_log_count += 1
        if self._prob_log_count % 50 == 0:
            logger.debug(f"VAD backend={self._backend_name} inference={elapsed_ms:.3f}ms prob={prob:.3f}")
        return prob

    def is_speech(self, audio: npt.NDArray[Any], sample_rate: int = SILERO_SAMPLE_RATE) -> bool:
        prob = self.get_speech_prob(audio, sample_rate)
        return prob >= self.threshold

    def process_chunk(self, audio: npt.NDArray[Any], sample_rate: int = SILERO_SAMPLE_RATE) -> tuple[bool, bool, bool]:
        """Process audio chunk and return VAD events.

        Returns:
            Tuple of (speech_started, speech_ended, sentence_pause)
        """
        chunk_duration_ms = len(audio) / sample_rate * 1000
        prob = self.get_speech_prob(audio, sample_rate)
        is_speech = prob >= self.threshold

        speech_started = False
        speech_ended = False
        sentence_pause = False

        if is_speech:
            self._silence_frames = 0
            self._speech_frames += chunk_duration_ms
            if not self._is_speaking and self._speech_frames >= self.min_speech_duration_ms:
                self._is_speaking = True
                speech_started = True
                logger.debug(f"Speech started (accumulated {self._speech_frames:.0f}ms)")
        else:
            self._speech_frames = 0
            if self._is_speaking:
                self._silence_frames += chunk_duration_ms
                # Check for sentence pause (shorter threshold)
                if self._silence_frames >= self.sentence_pause_threshold_ms:
                    sentence_pause = True
                    logger.debug(f"Sentence pause (silence {self._silence_frames:.0f}ms)")
                # Check for speech end (longer threshold)
                if self._silence_frames >= self.min_silence_duration_ms:
                    self._is_speaking = False
                    speech_ended = True
                    logger.debug(f"Speech ended (silence {self._silence_frames:.0f}ms)")

        return speech_started, speech_ended, sentence_pause

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    def reset(self) -> None:
        self._speech_frames = 0.0
        self._silence_frames = 0.0
        self._is_speaking = False
        self._prob_log_count = 0
        self._backend.reset_states()
        logger.debug("VAD state reset")


class VADEvent(Enum):
    NOTHING = auto()
    SPEECH_STARTED = auto()
    SENTENCE_PAUSE = auto()
    SPEECH_ENDED = auto()


class VADState(Enum):
    LISTENING = "listening"
    RECORDING = "recording"
    PROCESSING = "processing"


class VADStateMachine:
    """Shared VAD state machine for pre-roll buffering and speech detection.

    Callers feed audio chunks and react to returned events.
    Does NOT own audio sources, callbacks, or async/threading.
    """

    def __init__(
        self,
        vad: SileroVAD,
        chunk_size: int = VAD_CHUNK_SIZE,
        preroll_duration_s: float = 0.5,
    ) -> None:
        self._vad = vad
        self._state = VADState.LISTENING
        self._chunk_size = chunk_size
        self._max_preroll = int(preroll_duration_s * SILERO_SAMPLE_RATE / chunk_size)
        self._preroll_chunks: list[npt.NDArray[np.int16]] = []
        self.speech_chunks: list[npt.NDArray[np.int16]] = []
        self._last_sentence_pause_time: float = 0.0  # Debounce sentence pause events

    @property
    def state(self) -> VADState:
        return self._state

    def process_chunk(self, audio_chunk: npt.NDArray[np.int16]) -> VADEvent:
        if self._state == VADState.PROCESSING:
            return VADEvent.NOTHING

        if self._state == VADState.LISTENING:
            speech_started, _, _ = self._vad.process_chunk(audio_chunk, SILERO_SAMPLE_RATE)
            self._preroll_chunks.append(audio_chunk)
            if len(self._preroll_chunks) > self._max_preroll:
                self._preroll_chunks = self._preroll_chunks[-self._max_preroll :]
            if speech_started:
                self._state = VADState.RECORDING
                self.speech_chunks = list(self._preroll_chunks)
                self._preroll_chunks = []
                logger.info("Speech detected, recording...")
                return VADEvent.SPEECH_STARTED
            return VADEvent.NOTHING

        self.speech_chunks.append(audio_chunk)
        _, speech_ended, sentence_pause = self._vad.process_chunk(audio_chunk, SILERO_SAMPLE_RATE)

        if speech_ended:
            self._state = VADState.PROCESSING
            logger.info(f"Speech ended, {len(self.speech_chunks)} chunks")
            return VADEvent.SPEECH_ENDED

        if sentence_pause:
            # Debounce: only trigger if enough time has passed since last pause
            now = perf_counter()
            if now - self._last_sentence_pause_time > 0.5:  # 500ms debounce
                self._last_sentence_pause_time = now
                logger.info("[VAD] Sentence pause detected")
                return VADEvent.SENTENCE_PAUSE

        return VADEvent.NOTHING

    def finish_processing(self) -> None:
        self.speech_chunks = []
        self._preroll_chunks = []
        self._vad.reset()
        self._state = VADState.LISTENING
        self._last_sentence_pause_time = 0.0
