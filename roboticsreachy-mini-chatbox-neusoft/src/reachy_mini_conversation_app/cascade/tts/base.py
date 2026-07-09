"""TTS (Text-to-Speech) abstraction for cascade pipeline."""

from __future__ import annotations
import abc
from typing import Optional, AsyncIterator
import os


class TTSProvider(abc.ABC):
    """Abstract base class for TTS providers."""

    base_model_dir = os.path.join("models", "TTS")

    @property
    def sample_rate(self) -> int:
        """Audio sample rate in Hz. Override for non-24kHz providers."""
        return 24000

    async def warmup(self) -> None:
        """Warm up the TTS engine by running a short synthesis.

        Called once after initialization to pre-load model weights, JIT-compile
        inference graphs, fill caches, etc.  Default implementation is a no-op;
        providers that benefit from warmup should override this method.
        """

    @abc.abstractmethod
    def synthesize(self, text: str, voice: Optional[str] = None) -> AsyncIterator[bytes]:
        """Synthesize text to audio stream.

        Args:
            text: Text to synthesize
            voice: Optional voice identifier

        Yields:
            Audio bytes (PCM or other format depending on provider)

        """
        raise NotImplementedError
