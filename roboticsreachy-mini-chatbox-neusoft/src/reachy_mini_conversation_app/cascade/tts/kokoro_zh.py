"""Kokoro-82M-v1.1-zh Chinese TTS provider for cascade pipeline.

Separate from the English KokoroTTS (kokoro.py) to avoid any interference.
Uses the Chinese-specific KModel + KPipeline(lang_code='z') API with:
- en_callable for mixed Chinese-English reading
- speed_callable to mitigate rushing on longer text
"""

from __future__ import annotations

import time
import asyncio
import logging
from pathlib import Path
from typing import AsyncIterator, Callable, Optional

import numpy as np

from .base import TTSProvider
from .utils import trim_leading_silence


logger = logging.getLogger(__name__)

REPO_ID = "hexgrad/Kokoro-82M-v1.1-zh"
LOCAL_MODEL_DIR = Path("models/kokoro-zh")

CHUNK_SIZE = 4096  # samples per sub-chunk (~170ms at 24kHz)


def make_speed_callable() -> Callable[[int], float]:
    """Create a speed callable that slows down for longer Chinese text.

    Kokoro Chinese tends to rush on longer texts (>83 phonemes).
    This applies a piecewise linear slowdown:
    - 0–83 phonemes: speed = 1.0 (normal)
    - 83–310 phonemes: linear ramp from 1.0 → 0.85
    - >310 phonemes: speed = 0.85 (floor)

    Returns:
        Callable that takes phoneme count and returns speed multiplier.
    """
    def _speed(n_phonemes: int) -> float:
        if n_phonemes <= 83:
            return 1.0
        if n_phonemes >= 310:
            return 0.85
        return 1.0 - 0.15 * (n_phonemes - 83) / (310 - 83)

    return _speed


def make_en_callable(repo_id: str = REPO_ID) -> Callable[[str], str]:
    """Create an en_callable for mixed Chinese-English reading.

    The Chinese Kokoro pipeline needs an English phonemizer for
    handling English words/sentences embedded in Chinese text.

    Args:
        repo_id: HuggingFace repo ID for the model.

    Returns:
        Callable that takes English text and returns phoneme string.
    """
    from kokoro import KPipeline

    en_pipeline = KPipeline(lang_code="a", repo_id=repo_id, model=False)

    def _en_callable(text: str) -> str:
        return next(en_pipeline(text)).phonemes

    return _en_callable


class KokoroZHTTS(TTSProvider):
    """Kokoro-82M-v1.1-zh Chinese TTS implementation.

    Separate from the English KokoroTTS — uses KModel + KPipeline(lang_code='z')
    with en_callable for mixed reading and speed_callable for long text.
    """

    def __init__(
        self,
        voice: str = "zf_001",
        repo_id: str = REPO_ID,
        device: str = "cpu",
    ) -> None:
        """Initialize Chinese Kokoro TTS.

        Args:
            voice: Chinese voice name (zf_001~zf_085 female, zm_010 male).
            repo_id: HuggingFace repo ID (fallback if local files missing).
            device: Torch device for model inference ('cpu', 'cuda', 'mps').
        """
        from kokoro import KModel, KPipeline

        self.default_voice = voice
        self.repo_id = repo_id

        # Resolve local model paths
        local_config = LOCAL_MODEL_DIR / "config.json"
        local_weights = LOCAL_MODEL_DIR / "kokoro-v1_1-zh.pth"
        local_voice = LOCAL_MODEL_DIR / "voices" / f"{voice}.pt"

        use_local = local_config.exists() and local_weights.exists()
        if use_local:
            logger.info(f"Loading Kokoro-zh KModel from local: {LOCAL_MODEL_DIR}")
            self._model = KModel(
                config=str(local_config),
                model=str(local_weights),
            ).to(device).eval()
        else:
            logger.info(f"Loading Kokoro-zh KModel from HuggingFace: {repo_id}")
            self._model = KModel(repo_id=repo_id).to(device).eval()

        # Resolve voice path for local loading
        if local_voice.exists():
            self._voice_path = str(local_voice)
        else:
            self._voice_path = voice

        logger.info("Creating English phonemizer pipeline (en_callable)...")
        self._en_pipeline = KPipeline(lang_code="a", repo_id=repo_id, model=False)

        def _en_callable(text: str) -> str:
            return next(self._en_pipeline(text)).phonemes

        logger.info("Creating Chinese TTS pipeline (lang_code='z')...")
        self._zh_pipeline = KPipeline(
            lang_code="z",
            repo_id=repo_id,
            model=self._model,
            en_callable=_en_callable,
        )

        self._speed_callable = make_speed_callable()

        # Preload voice
        logger.info(f"Preloading voice: {voice}...")
        try:
            for _ in self._zh_pipeline(".", voice=self._voice_path, speed=1.0):
                break
            logger.info(f"Voice {voice} preloaded successfully")
        except Exception as e:
            logger.warning(f"Failed to preload voice {voice}: {e}")

        logger.info(f"Kokoro-zh TTS initialized: voice={voice}, local={use_local}, device={device}")

    async def synthesize(self, text: str, voice: Optional[str] = None) -> AsyncIterator[bytes]:
        """Synthesize Chinese text using Kokoro-82M-v1.1-zh with streaming.

        Yields:
            Audio bytes (PCM 16-bit, 24kHz mono, up to CHUNK_SIZE-sample sub-chunks)
        """
        from reachy_mini_conversation_app.cascade.timing import tracker

        if not text.strip():
            logger.warning("Empty text provided for synthesis")
            return

        voice_to_use = voice or self._voice_path
        logger.info(f"Kokoro-zh TTS: Starting synthesis for '{text[:50]}...'")

        tracker.mark("tts_start", {"text_len": len(text)})

        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _producer() -> None:
            """Iterate zh_pipeline in a thread, push sub-chunks to the async queue."""
            tracker.mark("tts_model_generation_start")
            generation_start = time.perf_counter()
            is_first_chunk = True

            try:
                for result in self._zh_pipeline(
                    text, voice=voice_to_use, speed=self._speed_callable
                ):
                    audio_data = result.audio if hasattr(result, "audio") else result
                    if hasattr(audio_data, "numpy"):
                        audio_data = audio_data.numpy()

                    if is_first_chunk:
                        tracker.mark("tts_model_first_chunk")
                        audio_data = trim_leading_silence(
                            audio_data, sample_rate=self.sample_rate, provider_name="Kokoro-zh TTS"
                        )
                        is_first_chunk = False

                    audio_int16 = (audio_data * 32767).astype(np.int16)

                    for i in range(0, len(audio_int16), CHUNK_SIZE):
                        sub_chunk = audio_int16[i : i + CHUNK_SIZE].tobytes()
                        loop.call_soon_threadsafe(queue.put_nowait, sub_chunk)

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
                    logger.info("Kokoro-zh TTS: First chunk ready (can start playback now!)")
                    first_yielded = False
                yield chunk

            logger.info(f"Kokoro-zh TTS: Generated {chunk_count} audio chunks for '{text[:50]}...'")
        finally:
            await future
