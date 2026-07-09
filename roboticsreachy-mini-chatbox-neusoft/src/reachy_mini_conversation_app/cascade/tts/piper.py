"""Piper TTS provider for cascade pipeline - lightweight local TTS with Chinese support."""

from __future__ import annotations

import time
import asyncio
import logging
from pathlib import Path
from typing import AsyncIterator, Optional

import numpy as np
import sys
import os

# g2pw uses open() without encoding; on Windows the default is GBK,
# which fails on UTF-8 data files.  Patch builtins.open so it defaults
# to UTF-8 when no encoding is specified and mode is not binary.
if sys.platform == "win32":
    import builtins as _builtins

    _orig_open = _builtins.open

    def _open_utf8(file, mode="r", *args, **kwargs):
        if "encoding" not in kwargs and "b" not in str(mode):
            kwargs["encoding"] = "utf-8"
        return _orig_open(file, mode, *args, **kwargs)

    _builtins.open = _open_utf8

from .base import TTSProvider
from .utils import trim_leading_silence


logger = logging.getLogger(__name__)

# Piper generates per-sentence chunks (typically 1-3 seconds each).
# We further split into sub-chunks to keep barge-in responsive even when
# Piper yields a whole sentence at once. 4096 samples ≈ 186ms at 22kHz:
# short enough to limit stale audio after interrupt, still large enough
# for smooth playback.
SUB_CHUNK_SIZE = 4096


def _preprocess_mixed_text(text: str) -> str:
    char_map = {
        "b": "弼",
        "c": "s伊",
        "d": "棣",
        "e": "邑", 
        "f": "癌fu ",
        "g": "暨", 
        "h": "ah ",
        "i": "埃",
        "k": "剋",
        "l": "癌o ",
        "m": "埃mu ",
        "n": "嗯",
        "o": "讴",
        "p": "砒",
        "q": "k呦",
        "r": "錒r ",
        "s": "埃s ",
        "t": "倜",
        "u": "滺",
        "v": "u一",
        "w": "哒bbw ",
        "x": "挨ks s ",
        "y": "顡",
    }
    result = []
    for ch in text:
        if "A" <= ch <= "Z":
            ch = ch.lower()
        # 再应用字符映射
        ch = char_map.get(ch, ch)
        result.append(ch)
    return "".join(result)


class PiperTTS(TTSProvider):
    """Piper TTS implementation - lightweight, supports Chinese models.

    Model resolution:
    - If ``model`` is an absolute path to a .onnx file, use it directly.
    - Otherwise, look for ``{model}.onnx`` in the ``models/`` directory.
    """

    def __init__(
        self,
        model_onnx_path: str = os.path.join(TTSProvider.base_model_dir, "piper-zh_CN-chaowen-medium", "zh_CN-chaowen-medium.onnx"),
        noise_scale: float = 0.667,
        length_scale: float = 1.0,
        noise_w: float = 0.8,
    ) -> None:
        """Initialize Piper TTS.

        Args:
            model_onnx_path: Model path to .onnx file.
            noise_scale: Reserved for future use.
            length_scale: Reserved for future use.
            noise_w: Reserved for future use.

        """
        from piper import PiperVoice

        model_path = Path(model_onnx_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Piper model not found: {model_path}. "
                "Download from https://huggingface.co/rhasspy/piper-voices"
            )

        self._model_path = model_path
        self._noise_scale = noise_scale
        self._length_scale = length_scale
        self._noise_w = noise_w

        logger.info(f"Loading Piper model from {model_path}...")
        self._voice = PiperVoice.load(str(model_path), download_dir=self.base_model_dir)
        self._sample_rate: int = self._voice.config.sample_rate

        logger.info(f"Piper TTS initialized: model={model_path.name}, sample_rate={self._sample_rate}")

    @property
    def sample_rate(self) -> int:
        """Audio sample rate from model config."""
        return self._sample_rate

    async def warmup(self) -> None:
        """Warm up Piper by synthesizing a short text to pre-load the model."""
        logger.info("Piper TTS: warming up with 'Hi'...")
        async for _ in self.synthesize("Hi"):
            pass
        logger.info("Piper TTS: warmup complete")

    async def synthesize(self, text: str, voice: Optional[str] = None) -> AsyncIterator[bytes]:
        """Synthesize text using Piper TTS with streaming via producer thread.

        Yields:
            Audio bytes (PCM 16-bit, model sample rate, mono, up to SUB_CHUNK_SIZE-sample chunks)

        """
        from reachy_mini_conversation_app.cascade.timing import tracker

        if not text.strip():
            logger.warning("Empty text provided for synthesis")
            return

        logger.info(f"Piper TTS: Starting synthesis for '{text[:50]}...'")
        tracker.mark("tts_start", {"text_len": len(text)})
        # Preprocess: lowercase English letters + space after each, for Chinese Piper model
        processed_text = _preprocess_mixed_text(text)

        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _producer() -> None:
            """Iterate Piper synthesize (sync generator) and push sub-chunks to queue."""
            tracker.mark("tts_model_generation_start")
            generation_start = time.perf_counter()
            is_first_chunk = True

            try:
                for chunk in self._voice.synthesize(processed_text):
                    audio_float = chunk.audio_float_array

                    if is_first_chunk:
                        tracker.mark("tts_model_first_chunk")
                        audio_float = trim_leading_silence(
                            audio_float,
                            sample_rate=self._sample_rate,
                            provider_name="Piper TTS",
                        )
                        is_first_chunk = False

                    audio_int16 = np.clip(audio_float * 32767, -32767, 32767).astype(np.int16)
                    for i in range(0, len(audio_int16), SUB_CHUNK_SIZE):
                        chunk_bytes = audio_int16[i : i + SUB_CHUNK_SIZE].tobytes()
                        loop.call_soon_threadsafe(queue.put_nowait, chunk_bytes)

                generation_time_ms = (time.perf_counter() - generation_start) * 1000
                tracker.mark("tts_model_generation_complete", {"generation_ms": round(generation_time_ms, 1)})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        future = asyncio.ensure_future(asyncio.to_thread(_producer))

        try:
            first_yielded = True
            chunk_count = 0
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                chunk_count += 1
                if first_yielded:
                    tracker.mark("tts_first_chunk_ready")
                    logger.info("Piper TTS: First chunk ready (can start playback now!)")
                    first_yielded = False
                yield chunk

            logger.info(f"Piper TTS: Generated {chunk_count} audio chunks for '{text[:50]}...'")
        finally:
            await future
