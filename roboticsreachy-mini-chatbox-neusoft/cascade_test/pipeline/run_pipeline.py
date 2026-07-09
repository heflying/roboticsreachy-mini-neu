"""Audio file → ASR → LLM → TTS pipeline integration test.

Feeds audio file(s) through the full cascade pipeline and saves:
  - ASR transcript       ({name}_transcript.txt)
  - LLM response text    ({name}_response.txt)
  - TTS synthesized audio ({name}_audio.wav)

Memory (conversation history) is toggleable via --memory flag.
Default: OFF — each audio file starts fresh with only system prompt.

Usage:
    cd project_root

    # Single file, no memory (default)
    python cascade_test/pipeline/run_pipeline.py --input audio.wav

    # Directory of WAV files, with memory
    python cascade_test/pipeline/run_pipeline.py --input ./测试/ --memory

    # Override providers
    python cascade_test/pipeline/run_pipeline.py --input audio.wav \\
        --llm-provider ollama-qwen2.5-0.5b --language zh
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import os
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(override=True)

from reachy_mini_conversation_app.cascade.asr.base import ASRProvider
from reachy_mini_conversation_app.cascade.llm.base import LLMChunk, LLMProvider
from reachy_mini_conversation_app.cascade.tts.base import TTSProvider
from reachy_mini_conversation_app.cascade.config import get_config, set_config
from reachy_mini_conversation_app.cascade.provider_factory import (
    init_asr_provider,
    init_llm_provider,
    init_tts_provider,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# WAV utilities
# ---------------------------------------------------------------------------


def _write_wav(path: Path, pcm_bytes: bytes, sample_rate: int, channels: int = 1, sample_width: int = 2) -> None:
    """Write raw PCM bytes as a WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def _append_silence(audio_bytes: bytes, duration_ms: int = 500) -> bytes:
    """Append silence to WAV bytes in memory — no file I/O, just byte manipulation.

    Reads params via wave module, appends zero PCM frames, updates RIFF and
    data chunk sizes in-place.
    """
    import struct

    if audio_bytes[:4] != b"RIFF" or audio_bytes[8:12] != b"WAVE":
        return audio_bytes

    buf = io.BytesIO(audio_bytes)
    with wave.open(buf, "rb") as wf:
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()

    silence_frames = int(framerate * duration_ms / 1000)
    silence = b"\x00" * (silence_frames * nchannels * sampwidth)

    data_tag_pos = audio_bytes.find(b"data")
    if data_tag_pos < 0:
        return audio_bytes

    data_size_pos = data_tag_pos + 4
    current_data_size = struct.unpack_from("<I", audio_bytes, data_size_pos)[0]

    result = bytearray(audio_bytes)
    result.extend(silence)

    # Update RIFF chunk size (byte 4) and data chunk size
    struct.pack_into("<I", result, 4, len(result) - 8)
    struct.pack_into("<I", result, data_size_pos, current_data_size + len(silence))

    return bytes(result)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class TurnResult:
    """Result for one audio file processed through the pipeline."""

    file_name: str
    rel_path: str = ""  # relative path from input root, preserved in output
    transcript: str = ""
    llm_response: str = ""
    tts_audio_data: bytes = b""
    asr_ms: float = 0.0
    llm_ttft_ms: Optional[float] = None
    llm_total_ms: float = 0.0
    tts_duration_ms: float = 0.0
    tts_sample_rate: int = 24000
    error: str = ""


# ---------------------------------------------------------------------------
# Provider helpers
# ---------------------------------------------------------------------------


def _override_env(provider_type: str, name: str) -> None:
    """Set CASCADE_{TYPE}_PROVIDER env var and reset config singleton."""
    env_key = f"CASCADE_{provider_type.upper()}_PROVIDER"
    os.environ[env_key] = name
    set_config(None)


def get_available_providers(provider_type: str) -> List[str]:
    """Get available provider names from cascade.yaml."""
    import yaml

    config_file = Path("cascade.yaml")
    if not config_file.exists():
        return []
    with open(config_file, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    section = data.get(provider_type, {})
    return list(section.get("providers", {}).keys())


# ---------------------------------------------------------------------------
# LLM streaming helper
# ---------------------------------------------------------------------------


async def _stream_llm_response(
    llm: LLMProvider,
    messages: List[Dict[str, Any]],
    temperature: float,
) -> tuple[str, Optional[float], float]:
    """Stream LLM response and return (text, ttft_ms, total_ms)."""
    accumulated_text = ""
    first_token_time: Optional[float] = None
    t_start = time.perf_counter()

    async for chunk in llm.generate(
        messages=messages,
        tools=None,
        temperature=temperature,
        token=None,
    ):
        if chunk.type == "text_delta" and chunk.content:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            accumulated_text += chunk.content
        elif chunk.type == "done":
            break

    total_ms = (time.perf_counter() - t_start) * 1000
    ttft_ms = (first_token_time - t_start) * 1000 if first_token_time else None
    return accumulated_text.strip(), ttft_ms, total_ms


# ---------------------------------------------------------------------------
# Core pipeline: one audio file → ASR → LLM → TTS
# ---------------------------------------------------------------------------


async def process_audio_file(
    audio_path: Path,
    asr: ASRProvider,
    llm: LLMProvider,
    tts: TTSProvider,
    *,
    rel_path: Path = Path(),
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    language: Optional[str] = None,
    temperature: float = 0.0,
) -> TurnResult:
    """Run a single audio file through ASR → LLM → TTS.

    Args:
        audio_path: Path to the audio file.
        asr: Initialized ASR provider.
        llm: Initialized LLM provider.
        tts: Initialized TTS provider.
        rel_path: Relative path from input root, used to mirror
            directory structure in output.
        conversation_history: Previous conversation turns to prepend
            (only used when --memory is on). Each turn = {role, content}.
        language: ASR language hint (e.g. 'zh', 'en').
        temperature: LLM sampling temperature.

    Returns:
        TurnResult with transcript, response, timing, and TTS audio data.

    """
    result = TurnResult(file_name=audio_path.name, rel_path=str(rel_path))

    # ---- Step 1: Read audio ----
    if not audio_path.exists():
        result.error = f"Audio file not found: {audio_path}"
        return result

    audio_bytes = audio_path.read_bytes()
    audio_bytes = _append_silence(audio_bytes, duration_ms=500)

    # ---- Step 2: ASR ----
    t0 = time.perf_counter()
    try:
        transcript = await asr.transcribe(audio_bytes, language=language)
    except Exception as e:
        result.error = f"ASR failed: {e}"
        return result
    result.asr_ms = (time.perf_counter() - t0) * 1000
    result.transcript = transcript.strip()

    if not result.transcript:
        result.error = "ASR returned empty transcript"
        return result

    # ---- Step 3: LLM ----
    # Build messages: [history..., {"role": "user", "content": transcript}]
    messages: List[Dict[str, Any]] = list(conversation_history) if conversation_history else []
    messages.append({"role": "user", "content": transcript})

    try:
        response_text, ttft_ms, total_ms = await _stream_llm_response(
            llm, messages, temperature,
        )
    except Exception as e:
        result.error = f"LLM failed: {e}"
        return result

    result.llm_response = response_text
    result.llm_ttft_ms = ttft_ms
    result.llm_total_ms = total_ms

    if not result.llm_response:
        result.error = "LLM returned empty response"
        return result

    # ---- Step 4: TTS ----
    all_audio = bytearray()
    t_tts_start = time.perf_counter()
    try:
        async for audio_chunk in tts.synthesize(text=result.llm_response):
            all_audio.extend(audio_chunk)
    except Exception as e:
        result.error = f"TTS failed: {e}"
        return result
    result.tts_duration_ms = (time.perf_counter() - t_tts_start) * 1000
    result.tts_audio_data = bytes(all_audio)
    result.tts_sample_rate = tts.sample_rate

    return result


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


async def run_pipeline(
    input_path: Path,
    output_dir: Path,
    *,
    memory: bool = False,
    language: Optional[str] = None,
    temperature: float = 0.0,
    asr_provider: Optional[str] = None,
    llm_provider: Optional[str] = None,
    tts_provider: Optional[str] = None,
) -> List[TurnResult]:
    """Run pipeline on audio file(s).

    Args:
        input_path: Single WAV file or directory of WAV files.
        output_dir: Directory for output files.
        memory: If True, accumulate conversation history across turns.
            If False (default), each turn gets only system prompt.
        language: ASR language hint (e.g. 'zh', 'en').
        temperature: LLM sampling temperature.
        asr_provider: Override ASR provider from cascade.yaml.
        llm_provider: Override LLM provider from cascade.yaml.
        tts_provider: Override TTS provider from cascade.yaml.

    Returns:
        List of TurnResult, one per audio file processed.

    """
    # Override providers if requested
    if asr_provider:
        _override_env("asr", asr_provider)
    if llm_provider:
        _override_env("llm", llm_provider)
    if tts_provider:
        _override_env("tts", tts_provider)

    # Ensure config is fresh
    set_config(None)

    # Collect audio files (recursively through subdirectories)
    _AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
    audio_files: List[Path] = []
    if input_path.is_file():
        audio_files = [input_path]
    elif input_path.is_dir():
        audio_files = sorted(
            p for p in input_path.rglob("*")
            if p.is_file() and p.suffix.lower() in _AUDIO_EXTS
        )
    else:
        print(f"ERROR: Input path not found: {input_path}")
        return []

    if not audio_files:
        print(f"No audio files found in: {input_path}")
        return []

    # Initialize providers
    cfg = get_config()
    print(
        f"Initializing providers "
        f"(ASR={asr_provider or cfg.asr_provider}, "
        f"LLM={llm_provider or cfg.llm_provider}, "
        f"TTS={tts_provider or cfg.tts_provider})"
    )
    asr = init_asr_provider()
    llm = init_llm_provider()
    tts = init_tts_provider()

    model_name = getattr(llm, "model", "unknown")
    system_instructions = getattr(llm, "system_instructions", None)
    sys_len = len(system_instructions) if system_instructions else 0
    print(f"LLM model: {model_name} | System prompt: {sys_len} chars")
    print(f"Memory: {'ON (accumulating history)' if memory else 'OFF (system prompt only)'}")
    print(f"Files to process: {len(audio_files)}\n")

    # Prepare output directory — clear it first so each run starts fresh
    import shutil

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Conversation history — only accumulated when --memory is set
    conversation_history: List[Dict[str, Any]] = []

    results: List[TurnResult] = []
    total_start = time.perf_counter()

    for i, audio_path in enumerate(audio_files):
        stem = audio_path.stem
        # Relative path from input root; mirrors directory structure in output
        if input_path.is_dir():
            rel = audio_path.relative_to(input_path)
        else:
            rel = Path(audio_path.name)

        print(f"[{i+1}/{len(audio_files)}] {rel}  ", end="", flush=True)

        # Pass history (or empty list) — the provider handles system instructions itself
        result = await process_audio_file(
            audio_path,
            asr,
            llm,
            tts,
            rel_path=rel,
            conversation_history=conversation_history if memory else None,
            language=language,
            temperature=temperature,
        )

        # Write outputs — mirror subdirectory structure from input
        if not result.error:
            out_subdir = output_dir / rel.parent
            out_subdir.mkdir(parents=True, exist_ok=True)
            (out_subdir / f"{stem}_transcript.txt").write_text(
                result.transcript, encoding="utf-8"
            )
            (out_subdir / f"{stem}_response.txt").write_text(
                result.llm_response, encoding="utf-8"
            )
            if result.tts_audio_data:
                _write_wav(
                    out_subdir / f"{stem}_audio.wav",
                    result.tts_audio_data,
                    sample_rate=result.tts_sample_rate,
                )

        # Accumulate history if memory is on
        if memory and not result.error:
            conversation_history.append({"role": "user", "content": result.transcript})
            conversation_history.append({"role": "assistant", "content": result.llm_response})
        elif not memory:
            conversation_history = []

        # Print summary line
        if result.error:
            print(f"FAIL: {result.error}")
        else:
            ttft_str = f"{result.llm_ttft_ms:.0f}ms" if result.llm_ttft_ms else "N/A"
            audio_kb = len(result.tts_audio_data) / 1024
            print(
                f"ASR={result.asr_ms:.0f}ms | "
                f"LLM TTFT={ttft_str} total={result.llm_total_ms:.0f}ms | "
                f"TTS={result.tts_duration_ms:.0f}ms ({audio_kb:.1f}KB)"
            )

        results.append(result)

    total_elapsed = time.perf_counter() - total_start
    ok = sum(1 for r in results if not r.error)
    fail = sum(1 for r in results if r.error)
    print(f"\nDone: {ok} OK, {fail} failed | Total: {total_elapsed:.1f}s | Output: {output_dir}")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audio → ASR → LLM → TTS pipeline integration test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cascade_test/pipeline/run_pipeline.py --input audio.wav
  python cascade_test/pipeline/run_pipeline.py --input ./测试/ --memory
  python cascade_test/pipeline/run_pipeline.py --input audio.wav --llm-provider ollama-qwen2.5-0.5b --language zh
        """,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to audio file (.wav) or directory of audio files",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory (default: cascade_test/pipeline/output/<timestamp>)",
    )
    parser.add_argument(
        "--memory", "-m",
        action="store_true",
        default=False,
        help="Enable conversation history accumulation across turns (default: OFF)",
    )
    parser.add_argument(
        "--language", "-l",
        default=None,
        help="ASR language hint (e.g. zh, en)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--asr-provider",
        default=None,
        help="Override ASR provider name from cascade.yaml",
    )
    parser.add_argument(
        "--llm-provider",
        default=None,
        help="Override LLM provider name from cascade.yaml",
    )
    parser.add_argument(
        "--tts-provider",
        default=None,
        help="Override TTS provider name from cascade.yaml",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: WARNING)",
    )

    args = parser.parse_args()

    # Logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    # Output directory
    input_path = Path(args.input).resolve()
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).resolve().parent / "output" / ts

    # Run
    results = asyncio.run(
        run_pipeline(
            input_path=input_path,
            output_dir=output_dir,
            memory=args.memory,
            language=args.language,
            temperature=args.temperature,
            asr_provider=args.asr_provider,
            llm_provider=args.llm_provider,
            tts_provider=args.tts_provider,
        )
    )

    # Save summary JSON
    summary_path = output_dir / "summary.json"
    summary = {
        "memory_enabled": args.memory,
        "input": str(input_path),
        "language": args.language,
        "temperature": args.temperature,
        "total_files": len(results),
        "ok": sum(1 for r in results if not r.error),
        "failed": sum(1 for r in results if r.error),
        "results": [
            {
                "file": r.file_name,
                "rel_path": r.rel_path,
                "transcript": r.transcript[:200] if r.transcript else "",
                "response": r.llm_response[:200] if r.llm_response else "",
                "asr_ms": round(r.asr_ms, 1),
                "llm_ttft_ms": round(r.llm_ttft_ms, 1) if r.llm_ttft_ms else None,
                "llm_total_ms": round(r.llm_total_ms, 1),
                "tts_duration_ms": round(r.tts_duration_ms, 1),
                "tts_audio_kb": round(len(r.tts_audio_data) / 1024, 1),
                "error": r.error,
            }
            for r in results
        ],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
