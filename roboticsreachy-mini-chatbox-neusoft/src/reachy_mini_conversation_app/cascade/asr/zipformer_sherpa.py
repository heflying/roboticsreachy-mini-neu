"""Zipformer ASR provider via sherpa-onnx (local, CPU streaming)."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional
import os

import numpy as np
import numpy.typing as npt

from .audio_utils import wav_to_float32
from .base_streaming import StreamingASRProvider

logger = logging.getLogger(__name__)


class ZipformerSherpaASR(StreamingASRProvider):
    """Local streaming ASR using sherpa-onnx Zipformer transducer model.

    Pure CPU inference, ~160MB INT8 model, RTF 0.15.
    Model is loaded eagerly in __init__ for best first-turn performance.
    """

    # Expected model files
    _MODEL_FILES = ("encoder.int8.onnx", "decoder.onnx", "joiner.int8.onnx", "tokens.txt")

    def __init__(
        self,
        model_id: str = "csukuangfj/sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30",
        model_dir: str = os.path.join(StreamingASRProvider.base_model_dir, "zipformer-zh"),
        num_threads: int = 1,
        sample_rate: int = 16000,
        decoding_method: str = "greedy_search",
        enable_endpoint: bool = True,
        rule1_min_trailing_silence: float = 2.4,
        rule2_min_trailing_silence: float = 1.2,
    ) -> None:
        self._model_id = model_id
        self._model_dir = Path(model_dir)
        self._num_threads = num_threads
        self._sample_rate = sample_rate
        self._decoding_method = decoding_method
        self._enable_endpoint = enable_endpoint
        self._rule1_min_trailing_silence = rule1_min_trailing_silence
        self._rule2_min_trailing_silence = rule2_min_trailing_silence

        # Recognizer (singleton, shared across streams)
        self._recognizer: Any = None
        self._model_loaded = False

        # Streaming state (per-stream)
        self._stream: Any = None
        self._partial_text: str = ""

        # Minimum chunk size to avoid sherpa-onnx GetFrames assertion.
        # ~0.3s at 16kHz = 4800 samples. Smaller chunks are buffered.
        self._min_feed_samples = int(self._sample_rate * 0.3)
        self._audio_buffer: list[float] = []

        # Duration of silence padding before input_finished().
        # Gives the transducer enough context to commit the final token.
        self._padding_duration_s: float = 0.3

        # Serialize native sherpa-onnx calls — OnlineStream is not thread-safe
        # and on_chunk callbacks can schedule overlapping coroutines.
        self._native_lock = threading.Lock()

        # Eager load
        self._ensure_model()

    def _ensure_model(self) -> None:
        """Download model if missing, then create OnlineRecognizer."""
        if self._model_loaded:
            return

        self._download_if_missing()
        self._create_recognizer()
        self._model_loaded = True

    def _download_if_missing(self) -> None:
        """Download model from HuggingFace if local files are missing."""
        if self._all_model_files_present():
            logger.info(f"Zipformer model files found in {self._model_dir}")
            return

        logger.info(f"Downloading Zipformer model from {self._model_id}...")
        from huggingface_hub import snapshot_download

        t0 = time.perf_counter()
        if not self._model_dir.is_dir():
            self._model_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=self._model_id, local_dir=str(self._model_dir))
        elapsed = time.perf_counter() - t0
        logger.info(f"Model downloaded in {elapsed:.1f}s")

    def _all_model_files_present(self) -> bool:
        """Check if all required model files exist and are non-empty."""
        if not self._model_dir.exists():
            return False
        for fname in self._MODEL_FILES:
            fpath = self._model_dir / fname
            if not fpath.exists() or fpath.stat().st_size == 0:
                return False
        return True

    def _create_recognizer(self) -> None:
        """Create sherpa_onnx OnlineRecognizer from local model files."""
        import sherpa_onnx

        encoder = str(self._model_dir / "encoder.int8.onnx")
        decoder = str(self._model_dir / "decoder.onnx")
        joiner = str(self._model_dir / "joiner.int8.onnx")
        tokens = str(self._model_dir / "tokens.txt")

        t0 = time.perf_counter()
        self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            num_threads=self._num_threads,
            sample_rate=self._sample_rate,
            feature_dim=80,
            enable_endpoint_detection=self._enable_endpoint,
            rule1_min_trailing_silence=self._rule1_min_trailing_silence,
            rule2_min_trailing_silence=self._rule2_min_trailing_silence,
            decoding_method=self._decoding_method,
            provider="cpu",
        )
        elapsed = time.perf_counter() - t0
        logger.info(f"Zipformer recognizer created in {elapsed:.2f}s")

    # ------------------------------------------------------------------
    # StreamingASRProvider abstract methods
    # ------------------------------------------------------------------

    async def start_stream(self) -> None:
        """Create a new OnlineStream for this session."""
        from reachy_mini_conversation_app.cascade.timing import tracker

        self._stream = self._recognizer.create_stream()
        self._partial_text = ""
        self._audio_buffer = []

        tracker.mark("asr_local_ready")
        tracker.mark("asr_local_stream_start")
        logger.debug("Zipformer streaming session started")

    async def send_audio_chunk(self, audio_chunk: bytes) -> None:
        """Feed audio chunk to the recognizer."""
        if not audio_chunk:
            return

        audio = wav_to_float32(audio_chunk, self._sample_rate)
        if len(audio) == 0:
            return

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._sync_feed, audio)

    def _sync_feed(self, audio: npt.NDArray[np.float32]) -> None:
        """Synchronous: feed audio and decode one step.

        Buffers small chunks to avoid sherpa-onnx GetFrames assertion
        when insufficient audio is available for feature extraction.
        """
        with self._native_lock:
            if self._stream is None:
                return
            self._audio_buffer.extend(audio.tolist())
            if len(self._audio_buffer) < self._min_feed_samples:
                return

            buf = np.array(self._audio_buffer, dtype=np.float32)
            self._stream.accept_waveform(self._sample_rate, buf)
            self._audio_buffer = []
            if self._recognizer.is_ready(self._stream):
                self._recognizer.decode_stream(self._stream)

    async def get_partial_transcript(self) -> Optional[str]:
        """Return current partial transcript, or None."""
        from reachy_mini_conversation_app.cascade.timing import tracker

        loop = asyncio.get_event_loop()
        try:
            text = await loop.run_in_executor(None, self._get_result_locked)
        except ValueError:
            return self._partial_text or None
        text = text.strip() if text else ""
        if text:
            self._partial_text = text
            tracker.mark("asr_local_chunk_decode")
            return self._partial_text
        return None if not self._partial_text else self._partial_text

    def _get_result_locked(self) -> str:
        with self._native_lock:
            if self._stream is None:
                return ""
            return self._recognizer.get_result(self._stream)

    def _flush_and_end_stream(self) -> None:
        with self._native_lock:
            if self._stream is None:
                return
            # Flush buffered audio
            buf = np.array(self._audio_buffer, dtype=np.float32)
            self._stream.accept_waveform(self._sample_rate, buf)
            self._audio_buffer = []
            # Pad silence to help transducer commit final token
            padding = np.zeros(int(self._sample_rate * self._padding_duration_s), dtype=np.float32)
            self._stream.accept_waveform(self._sample_rate, padding)
            # Signal end of input
            self._stream.input_finished()
            # Decode until fully consumed
            while self._recognizer.is_ready(self._stream):
                self._recognizer.decode_stream(self._stream)

    def _end_stream_locked(self) -> None:
        with self._native_lock:
            if self._stream is None:
                return
            # Pad silence to help transducer commit final token
            padding = np.zeros(int(self._sample_rate * self._padding_duration_s), dtype=np.float32)
            self._stream.accept_waveform(self._sample_rate, padding)
            # Signal end of input
            self._stream.input_finished()
            # Decode until fully consumed
            while self._recognizer.is_ready(self._stream):
                self._recognizer.decode_stream(self._stream)

    async def end_stream(self) -> str:
        """Signal end of audio and return final transcript."""
        from reachy_mini_conversation_app.cascade.timing import tracker

        tracker.mark("asr_local_final_decode")

        # Flush buffered audio before signaling end
        if self._audio_buffer:
            await asyncio.get_event_loop().run_in_executor(
                None, self._flush_and_end_stream
            )
        else:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._end_stream_locked)

        loop = asyncio.get_event_loop()
        try:
            text = await loop.run_in_executor(None, self._get_result_locked)
        except ValueError as e:
            logger.warning(f"get_result failed ({e}), using partial transcript")
            text = ""

        result = text.strip() if text else ""
        if not result:
            result = self._partial_text

        tracker.mark("asr_result_delivered", {"transcript_len": len(result)})
        logger.info(f"Zipformer final transcript: '{result}'")

        self._stream = None
        return result
