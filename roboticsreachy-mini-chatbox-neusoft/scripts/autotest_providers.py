#!/usr/bin/env python3
"""Smoke-test each cascade provider by calling its real API.

Discovers all providers from cascade.yaml automatically.
Skips providers when API keys are missing, hardware doesn't match,
or required packages aren't installed.

Run from the project root:
    uv run python scripts/autotest_providers.py
"""

import os
import sys
import asyncio
import logging
import platform
import importlib
from typing import Any
from pathlib import Path
from unittest.mock import patch

import yaml


# Ensure we run from project root (cascade.yaml must be found)
os.chdir(Path(__file__).resolve().parent.parent)

from reachy_mini_conversation_app.cascade.config import CascadeConfig, set_config  # noqa: E402
from reachy_mini_conversation_app.cascade.provider_factory import init_provider  # noqa: E402


logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Terminal colours
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _pass(label: str) -> str:
    return f"  {_c('32', '✔ PASS')}  {label}"


def _fail(label: str, reason: str) -> str:
    return f"  {_c('31', '✘ FAIL')}  {label}  — {reason}"


def _skip(label: str, reason: str) -> str:
    return f"  {_c('33', '⊘ SKIP')}  {label}  ({reason})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
THE_TIME_HAS_COME_WAV = FIXTURES_DIR / "the_time_has_come.wav"
# Minimum duration in seconds for TTS output to be considered valid
MIN_TTS_DURATION_S = 0.3


def _has_key(env_var: str) -> bool:
    val = os.getenv(env_var, "")
    return bool(val) and val != "test-dummy"


def _load_speech_wav() -> bytes:
    """Load the bundled 'the time has come' WAV fixture."""
    return THE_TIME_HAS_COME_WAV.read_bytes()


def _load_config() -> CascadeConfig:
    with patch.object(CascadeConfig, "_validate"), patch.object(CascadeConfig, "_log_config"):
        cfg = CascadeConfig()
    set_config(cfg)
    return cfg


def _check_skip(info: dict[str, Any]) -> str | None:
    """Return a skip reason string, or None if the provider can run."""
    # Missing API keys
    for key in info.get("requires", []):
        if not _has_key(key):
            return f"{key} not set"

    # Hardware check
    hw = info.get("hardware")
    if hw == "apple_silicon":
        if platform.machine() != "arm64" or platform.system() != "Darwin":
            return f"requires Apple Silicon (got {platform.system()} {platform.machine()})"
    elif hw == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                return "requires CUDA GPU (none detected)"
        except ImportError:
            return "requires PyTorch with CUDA"

    # Missing dependency
    import_check = info.get("import_check")
    if import_check:
        try:
            importlib.import_module(import_check)
        except ImportError:
            extra = info.get("install_extra", "???")
            return f"'{import_check}' not installed (uv sync --extra {extra})"

    return None


# ---------------------------------------------------------------------------
# Per-type test runners
# ---------------------------------------------------------------------------

async def _test_asr(cfg: CascadeConfig, provider: str, wav: bytes) -> None:
    """Transcribe 'the time has come' WAV and check the transcript contains 'time'."""
    cfg.asr_provider = provider
    asr = init_provider("asr")
    result = await asr.transcribe(wav)
    assert isinstance(result, str), f"Expected str, got {type(result)}"
    assert "time" in result.lower(), "Expected 'time' in transcript"


async def _test_llm(cfg: CascadeConfig, provider: str) -> None:
    """Send a prompt and verify streaming text response."""
    cfg.llm_provider = provider
    llm = init_provider("llm", {"system_instructions": "You are a test assistant."})
    messages = [{"role": "user", "content": "Say hello in one word."}]
    chunks = []
    async for chunk in llm.generate(messages):
        chunks.append(chunk)
    text_deltas = [c for c in chunks if c.type == "text_delta"]
    assert len(text_deltas) > 0, "No text_delta chunks received"
    full_text = "".join(c.content for c in text_deltas if c.content)
    assert len(full_text) > 0, "LLM returned empty text"


async def _test_tts(cfg: CascadeConfig, provider: str) -> None:
    """Synthesize 'Hello world' and verify audio duration is reasonable."""
    cfg.tts_provider = provider
    tts = init_provider("tts")
    audio_chunks = []
    async for chunk in tts.synthesize("Hello world"):
        audio_chunks.append(chunk)
    total = b"".join(audio_chunks)
    assert len(total) > 0, "No audio bytes produced"
    # Check duration: PCM 16-bit mono at provider's sample rate
    sample_rate = getattr(tts, "sample_rate", 24000)
    duration_s = len(total) / (sample_rate * 2)  # 2 bytes per sample (16-bit)
    assert duration_s >= MIN_TTS_DURATION_S, f"Audio too short: {duration_s:.2f}s (expected >= {MIN_TTS_DURATION_S}s)"




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> int:
    """Discover and smoke-test all providers from cascade.yaml."""
    with open("cascade.yaml", encoding="utf-8") as f:
        cascade_yaml = yaml.safe_load(f)

    cfg = _load_config()
    wav = _load_speech_wav()

    passed = 0
    failed = 0
    skipped = 0

    for ptype in ("asr", "llm", "tts"):
        print(f"\n  {_c('1', ptype.upper())}")
        providers = cascade_yaml[ptype]["providers"]
        for name, info in providers.items():
            label = f"{ptype.upper():<4} {name}"

            skip_reason = _check_skip(info)
            if skip_reason:
                print(_skip(label, skip_reason))
                skipped += 1
                continue

            try:
                if ptype == "asr":
                    await _test_asr(cfg, name, wav)
                elif ptype == "llm":
                    await _test_llm(cfg, name)
                else:
                    await _test_tts(cfg, name)
                print(_pass(label))
                passed += 1
            except Exception as e:
                print(_fail(label, str(e)))
                failed += 1

    set_config(None)

    parts = [
        _c("32", f"{passed} passed") if passed else f"{passed} passed",
        _c("31", f"{failed} failed") if failed else f"{failed} failed",
        _c("33", f"{skipped} skipped") if skipped else f"{skipped} skipped",
    ]
    print(f"\n{', '.join(parts)}")
    return 1 if failed else 0


if __name__ == "__main__":
    rc = asyncio.run(main())
    # os._exit avoids segfault during interpreter shutdown caused by
    # conflicting native libs (cv2/av dylib duplicates, MLX, torch).
    os._exit(rc)
