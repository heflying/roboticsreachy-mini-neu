"""Kokoro-82M-v1.1-zh ONNX Chinese TTS provider for cascade pipeline.

Pipeline: Chinese text → KPipeline(lang_code='z') phonemize (Bopomofo) → ONNX Runtime inference.
Direct onnxruntime usage for full control over input shapes.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import AsyncIterator, Optional

import numpy as np

from .base import TTSProvider

logger = logging.getLogger(__name__)

MODEL_DIR = Path("models/kokoro-zh-onnx")
REPO_ID = "hexgrad/Kokoro-82M-v1.1-zh"

CHUNK_SIZE = 4096  # samples per sub-chunk (~170ms at 24kHz)
SAMPLE_RATE = 24000
MAX_PHONEME_LENGTH = 510

# Common Chinese opening phrases for speculative pre-synthesis (D3 bypass)
WARMUP_PHRASES = [
    "好的",
    "好的，",
    "嗯",
    "当然",
    "是的",
    "没问题",
]


class KokoroZhOnnxTTS(TTSProvider):
    """ONNX-based Kokoro-82M-v1.1-zh Chinese TTS.

    Uses kokoro KPipeline for Chinese phonemization + onnxruntime for inference.
    """

    def __init__(
        self,
        voice: str = "zf_001",
        model_variant: str = "quantized",
        device: str = "cpu",
    ) -> None:
        import onnxruntime as rt
        from kokoro import KPipeline

        self.default_voice = voice
        self.model_variant = model_variant

        if model_variant in ("normal", "fp32", "standard"):
            model_path = MODEL_DIR / "onnx" / "model.onnx"
            variant_label = "标准模型 (FP32, ~339MB)"
        elif model_variant in ("quantized", "int8", "quantized_int8"):
            model_path = MODEL_DIR / "onnx" / "model_quantized.onnx"
            variant_label = "量化模型 (INT8, ~127MB)"
        else:
            raise ValueError(f"Unknown model_variant: {model_variant}")

        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        model_size_mb = model_path.stat().st_size / (1024 * 1024)
        logger.info(f"Loading Kokoro-zh ONNX [{variant_label}] ({model_size_mb:.1f} MB)")

        # Load ONNX session with full performance optimizations
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
        sess_options.intra_op_num_threads = 0  # 0 = all CPU cores
        sess_options.inter_op_num_threads = 0
        sess_options.enable_cpu_mem_arena = True  # Memory pool for faster allocation
        sess_options.enable_mem_pattern = True  # Reuse memory patterns
        sess_options.enable_mem_reuse = True
        self._session = rt.InferenceSession(
            str(model_path), sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        self._input_names = [inp.name for inp in self._session.get_inputs()]
        for inp in self._session.get_inputs():
            logger.info(f"  ONNX input: {inp.name} shape={inp.shape} type={inp.type}")

        # Load voice embedding
        voice_path = MODEL_DIR / "voices" / f"{voice}.bin"
        if not voice_path.exists():
            raise FileNotFoundError(f"Voice file not found: {voice_path}")
        self._voice = np.fromfile(str(voice_path), dtype=np.float32).reshape(-1, 256)
        logger.info(f"  Voice: {voice}, shape={self._voice.shape}")

        # Load vocab from tokenizer.json for tokenization
        tokenizer_path = MODEL_DIR / "tokenizer.json"
        with open(tokenizer_path, encoding="utf-8") as f:
            tokenizer_data = json.load(f)
        self._vocab = tokenizer_data["model"]["vocab"]

        # Build allowed chars regex from normalizer pattern
        norm_pattern = tokenizer_data.get("normalizer", {}).get("pattern", {}).get("Regex", "")
        if norm_pattern:
            self._allowed_chars = set(re.findall(r"(?:\\u[0-9a-fA-F]{4}|[^\\\[\]^])", norm_pattern))
        else:
            self._allowed_chars = None

        # ONNX session warm-up: run a dummy inference to trigger graph compilation & memory allocation
        logger.info("Warming up ONNX session...")
        warmup_tokens = [0, 16, 0]  # minimal: pad + space + pad
        warmup_inputs = {}
        for inp in self._session.get_inputs():
            if inp.name in ("input_ids", "tokens"):
                warmup_inputs[inp.name] = np.array([warmup_tokens], dtype=np.int64)
            elif inp.name == "style":
                warmup_inputs[inp.name] = self._voice[1].reshape(1, -1).astype(np.float32)
            elif inp.name == "speed":
                warmup_inputs[inp.name] = np.array([1.0], dtype=np.float32)
        self._session.run(None, warmup_inputs)
        logger.info("  ONNX session warm-up complete")

        # Chinese phonemizer via PyTorch kokoro (no model, text processing only)
        logger.info("Initializing Chinese phonemizer (KPipeline lang_code='z')...")
        en_pipeline = KPipeline(lang_code="a", repo_id=REPO_ID, model=False)

        def _en_callable(text: str) -> str:
            return next(en_pipeline(text)).phonemes

        self._zh_phonemizer = KPipeline(
            lang_code="z",
            repo_id=REPO_ID,
            model=False,
            en_callable=_en_callable,
        )
        # Preload jieba dictionary so first conversation turn doesn't pay the cost
        logger.info("Preloading jieba dictionary...")
        for _ in self._zh_phonemizer("预加载"):
            break

        # Speculative pre-synthesis: cache common opening phrases for zero-D3 first segment
        self._phrase_cache: dict[str, list[bytes]] = {}
        logger.info(f"Pre-synthesizing {len(WARMUP_PHRASES)} common opening phrases...")
        for phrase in WARMUP_PHRASES:
            pcm_chunks = self._synthesize_to_chunks(phrase)
            if pcm_chunks:
                self._phrase_cache[phrase] = pcm_chunks
                logger.info(f"  Cached: '{phrase}' ({len(pcm_chunks)} chunks)")
        logger.info(f"Kokoro-zh ONNX TTS ready: variant={variant_label}, voice={voice}, cached phrases={len(self._phrase_cache)}")

    def _tokenize(self, phonemes: str) -> list[int]:
        """Convert phoneme string to token IDs using vocab."""
        tokens = []
        for ch in phonemes:
            if ch in self._vocab:
                tokens.append(self._vocab[ch])
        return tokens

    def _split_phonemes(self, phonemes: str) -> list[str]:
        """Split phonemes into batches at punctuation boundaries."""
        words = re.split(r"([.,!?;])", phonemes)
        batches = []
        current = ""
        for part in words:
            part = part.strip()
            if not part:
                continue
            if len(current) + len(part) + 1 >= MAX_PHONEME_LENGTH:
                if current:
                    batches.append(current.strip())
                current = part
            else:
                if part in ".,!?;":
                    current += part
                else:
                    if current:
                        current += " "
                    current += part
        if current:
            batches.append(current.strip())
        return batches

    def _create_audio(self, phonemes: str, speed: float = 1.0) -> tuple[np.ndarray, int]:
        """Run ONNX inference for one phoneme batch."""
        tokens = self._tokenize(phonemes)
        if not tokens:
            return np.array([], dtype=np.float32), SAMPLE_RATE

        if len(tokens) > MAX_PHONEME_LENGTH:
            tokens = tokens[:MAX_PHONEME_LENGTH]

        n_tokens = len(tokens)
        input_ids = np.array([[0] + tokens + [0]], dtype=np.int64)

        # Get voice style for this token count
        style_idx = min(n_tokens, self._voice.shape[0] - 1)
        style = self._voice[style_idx].reshape(1, -1).astype(np.float32)

        # Build inputs matching model signature
        inputs = {}
        for inp in self._session.get_inputs():
            if inp.name in ("input_ids", "tokens"):
                inputs[inp.name] = input_ids
            elif inp.name == "style":
                inputs[inp.name] = style
            elif inp.name == "speed":
                inputs[inp.name] = np.array([speed], dtype=np.float32)

        audio = self._session.run(None, inputs)[0]
        return audio, SAMPLE_RATE

    def _synthesize_to_chunks(self, text: str) -> list[bytes]:
        """Synchronously synthesize text to PCM chunks (for pre-caching)."""
        phoneme_parts = []
        for result in self._zh_phonemizer(text):
            if result.phonemes:
                phoneme_parts.append(result.phonemes)
        if not phoneme_parts:
            return []

        full_phonemes = " ".join(phoneme_parts)
        batches = self._split_phonemes(full_phonemes)

        chunks: list[bytes] = []
        for phonemes in batches:
            audio, _sr = self._create_audio(phonemes)
            if audio.size == 0:
                continue
            audio = np.clip(audio.flatten(), -1.0, 1.0)
            audio_int16 = (audio * 32767).astype(np.int16)
            for i in range(0, len(audio_int16), CHUNK_SIZE):
                chunks.append(audio_int16[i : i + CHUNK_SIZE].tobytes())
        return chunks

    def _find_cached_prefix(self, text: str) -> tuple[str | None, str]:
        """Check if text starts with a cached phrase. Returns (matched_phrase, remainder)."""
        for phrase in self._phrase_cache:
            if text.startswith(phrase) and len(text) > len(phrase):
                return phrase, text[len(phrase):]
        return None, text

    async def synthesize(
        self, text: str, voice: Optional[str] = None
    ) -> AsyncIterator[bytes]:
        from reachy_mini_conversation_app.cascade.timing import tracker

        if not text.strip():
            return

        logger.info(
            f"Kokoro-zh ONNX [{self.model_variant}]: synthesizing '{text[:50]}...'"
        )

        tracker.mark(
            "tts_start",
            {"text_len": len(text), "engine": "kokoro-onnx", "variant": self.model_variant},
        )

        # Check speculative pre-synthesis cache for first segment
        matched_phrase, remainder = self._find_cached_prefix(text)
        if matched_phrase and remainder.strip():
            cached_chunks = self._phrase_cache[matched_phrase]
            logger.info(
                f"  Cache HIT: '{matched_phrase}' ({len(cached_chunks)} chunks), "
                f"remainder: '{remainder[:30]}...'"
            )
            # Yield cached audio immediately (D3 ≈ 0 for prefix)
            tracker.mark("tts_cache_hit", {"phrase": matched_phrase})
            for chunk in cached_chunks:
                yield chunk
            tracker.mark("tts_first_chunk_ready")

            # Synthesize remainder in background
            async for chunk in self._synthesize_text(remainder):
                yield chunk
            return

        # No cache hit — normal synthesis path
        async for chunk in self._synthesize_text(text):
            yield chunk

    async def _synthesize_text(self, text: str) -> AsyncIterator[bytes]:
        """Core synthesis pipeline: phonemize → ONNX inference → PCM chunks."""
        from reachy_mini_conversation_app.cascade.timing import tracker

        # Step 1: Phonemize Chinese text → Bopomofo
        tracker.mark("tts_phonemize_start")
        phoneme_parts = []
        for result in self._zh_phonemizer(text):
            if result.phonemes:
                phoneme_parts.append(result.phonemes)

        if not phoneme_parts:
            logger.warning("Phonemization produced no output")
            return

        full_phonemes = " ".join(phoneme_parts)
        tracker.mark("tts_phonemize_complete", {"segments": len(phoneme_parts)})
        logger.info(f"  Phonemized: {len(phoneme_parts)} segs, {len(full_phonemes)} chars")

        # Step 2: Split and run ONNX inference
        batches = self._split_phonemes(full_phonemes)
        tracker.mark("tts_model_generation_start")
        logger.info(f"  ONNX inference: {len(batches)} batch(es)")

        loop = __import__("asyncio").get_running_loop()
        is_first_chunk = True
        chunk_count = 0

        for batch_idx, phonemes in enumerate(batches):
            audio, sr = await loop.run_in_executor(
                None, self._create_audio, phonemes, 1.0
            )

            if audio.size == 0:
                continue

            if is_first_chunk:
                tracker.mark("tts_model_first_chunk")
                logger.info(
                    f"  ONNX first batch done: {len(audio)} samples "
                    f"({len(audio)/SAMPLE_RATE:.2f}s)"
                )
                is_first_chunk = False

            audio = np.clip(audio.flatten(), -1.0, 1.0)
            audio_int16 = (audio * 32767).astype(np.int16)

            for i in range(0, len(audio_int16), CHUNK_SIZE):
                sub_chunk = audio_int16[i : i + CHUNK_SIZE].tobytes()
                chunk_count += 1
                if chunk_count == 1:
                    tracker.mark("tts_first_chunk_ready")
                yield sub_chunk

        tracker.mark("tts_model_generation_complete")
        logger.info(
            f"Kokoro-zh ONNX [{self.model_variant}]: "
            f"complete, {chunk_count} PCM chunks"
        )
