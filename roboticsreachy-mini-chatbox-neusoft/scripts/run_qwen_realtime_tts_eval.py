#!/usr/bin/env python3
"""Run real Qwen realtime memory scenarios with local TTS audio fixtures."""

from __future__ import annotations
import os
import json
import time
import wave
import asyncio
import argparse
import tempfile
import subprocess
from typing import Any, Literal
from pathlib import Path
from dataclasses import field, asdict, dataclass

import numpy as np
from dotenv import load_dotenv
from fastrtc import AdditionalOutputs

from reachy_mini_conversation_app.config import refresh_runtime_config_from_env
from reachy_mini_conversation_app.memory.eval import (
    _turns,
    load_cases,
    _case_db_path,
    _seed_runtime,
    evaluate_expectations,
)
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.qwen_omni_realtime import QWEN_INPUT_SAMPLE_RATE, QwenOmniRealtimeHandler


RealtimeToolMode = Literal["router", "native"]


@dataclass(slots=True)
class RealtimeProbe:
    """Timing and transcript evidence collected from a realtime run."""

    mode: str
    case_id: str
    session_update_sent_at: float | None = None
    session_updated_at: float | None = None
    content_audio_done_at: float | None = None
    speech_stopped_at: float | None = None
    first_user_transcript_at: float | None = None
    first_audio_at: float | None = None
    assistant_transcript_done_at: float | None = None
    response_done_at: float | None = None
    errors: list[str] = field(default_factory=list)
    event_counts: dict[str, int] = field(default_factory=dict)
    user_transcripts: list[str] = field(default_factory=list)
    assistant_transcripts: list[str] = field(default_factory=list)
    system_outputs: list[str] = field(default_factory=list)
    audio_chunks: int = 0
    output_audio_ms: float = 0.0
    session_update: dict[str, Any] = field(default_factory=dict)


class ProbeQwenHandler(QwenOmniRealtimeHandler):
    """Qwen handler that records protocol timings for a headless test run."""

    def __init__(
        self,
        deps: ToolDependencies,
        *,
        probe: RealtimeProbe,
        instance_path: str | None = None,
    ):
        """Create a probing handler around the production Qwen handler."""
        super().__init__(deps, gradio_mode=False, instance_path=instance_path)
        self.probe = probe

    def _build_session_update(self) -> dict[str, Any]:
        update = super()._build_session_update()
        self.probe.session_update = update
        return update

    async def _send_json(self, payload: dict[str, Any]) -> None:
        if payload.get("type") == "session.update" and self.probe.session_update_sent_at is None:
            self.probe.session_update_sent_at = time.perf_counter()
        await super()._send_json(payload)

    async def _handle_message(self, message: dict[str, Any]) -> None:
        event_type = str(message.get("type") or "")
        self.probe.event_counts[event_type] = self.probe.event_counts.get(event_type, 0) + 1
        now = time.perf_counter()
        if event_type == "session.updated" and self.probe.session_updated_at is None:
            self.probe.session_updated_at = now
        elif event_type == "input_audio_buffer.speech_stopped" and self.probe.speech_stopped_at is None:
            self.probe.speech_stopped_at = now
        elif event_type in {
            "conversation.item.input_audio_transcription.completed",
            "input_audio_transcription.completed",
        } and self.probe.first_user_transcript_at is None:
            self.probe.first_user_transcript_at = now
            _append_once(self.probe.user_transcripts, str(message.get("transcript") or message.get("text") or ""))
        elif event_type in {"response.audio.delta", "response.output_audio.delta"}:
            if self.probe.first_audio_at is None:
                self.probe.first_audio_at = now
        elif event_type in {"response.audio_transcript.done", "response.output_audio_transcript.done"}:
            self.probe.assistant_transcript_done_at = now
            transcript = str(message.get("transcript") or message.get("text") or "")
            if not transcript:
                transcript = "".join(self._pending_assistant_transcript_chunks).strip()
            _append_once(self.probe.assistant_transcripts, transcript)
        elif event_type == "response.done":
            self.probe.response_done_at = now
        elif event_type == "error":
            self.probe.errors.append(json.dumps(message.get("error") or message, ensure_ascii=False))
        await super()._handle_message(message)


class MovementManagerStub:
    """Minimal movement manager used by local tool/router paths."""

    def __init__(self) -> None:
        """Create the stub with a neutral listening state."""
        self.listening = False

    def set_listening(self, listening: bool) -> None:
        """Record whether the handler currently considers the user to be speaking."""
        self.listening = listening

    def is_idle(self) -> bool:
        """Return false to prevent idle actions during deterministic tests."""
        return False


@dataclass(slots=True)
class RealtimeCaseResult:
    """Serializable result for one realtime case and tool mode."""

    case_id: str
    title: str
    mode: str
    status: str
    assertions: list[dict[str, Any]]
    metrics: dict[str, Any]
    db_path: str
    memory_context: str
    user_transcripts: list[str]
    assistant_transcripts: list[str]
    system_outputs: list[str]
    event_counts: dict[str, int]
    session_tools_count: int
    error: str | None = None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="tests/memory_scenarios/eldercare_expanded_25.json")
    parser.add_argument("--case-id", action="append", help="Run only selected case id. May be repeated.")
    parser.add_argument("--limit", type=int, help="Run at most N selected cases.")
    parser.add_argument("--tool-mode", action="append", choices=["router", "native"], default=None)
    parser.add_argument("--db", default="/tmp/reachy_realtime_tts_eval_dbs", help="SQLite DB directory.")
    parser.add_argument("--report-dir", default="/tmp/reachy_realtime_tts_eval_reports")
    parser.add_argument("--audio-dir", default="tests/realtime_audio_fixtures")
    parser.add_argument("--voice", default="auto", help="macOS say voice. Use auto for a Mandarin voice.")
    parser.add_argument("--tts-rate", type=int, default=165)
    parser.add_argument("--turn-timeout-s", type=float, default=45.0)
    parser.add_argument("--connect-timeout-s", type=float, default=15.0)
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--send-speed", type=float, default=4.0, help="Audio send speed multiplier.")
    parser.add_argument("--overwrite-audio", action="store_true")
    parser.add_argument("--allow-real-api", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    return parser.parse_args()


async def main_async() -> int:
    """Run selected realtime TTS scenarios."""
    load_dotenv(override=True)
    args = parse_args()
    if not args.allow_real_api:
        raise SystemExit("Real realtime tests require --allow-real-api")
    if not (os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")):
        raise SystemExit("DASHSCOPE_API_KEY or QWEN_API_KEY is required")

    cases = load_cases(args.cases)
    selected_ids = set(args.case_id or [])
    selected = [case for case in cases if not selected_ids or str(case.get("id")) in selected_ids]
    if args.limit is not None:
        selected = selected[: args.limit]

    tool_modes = args.tool_mode or ["router", "native"]
    results: list[RealtimeCaseResult] = []
    for mode in tool_modes:
        for index, case in enumerate(selected):
            db_path = _case_db_path(Path(args.db) / mode, case, index, keep_db=True)
            result = await run_realtime_case(
                case,
                tool_mode=mode,
                db_path=db_path,
                audio_dir=Path(args.audio_dir),
                voice=args.voice,
                tts_rate=args.tts_rate,
                turn_timeout_s=args.turn_timeout_s,
                connect_timeout_s=args.connect_timeout_s,
                chunk_ms=args.chunk_ms,
                send_speed=args.send_speed,
                overwrite_audio=args.overwrite_audio,
            )
            results.append(result)

    json_path, md_path = write_realtime_reports(results, args.report_dir)
    passed = sum(result.status == "passed" for result in results)
    failed = sum(result.status == "failed" for result in results)
    errors = sum(result.status == "error" for result in results)
    print(f"Realtime TTS eval complete: {passed}/{len(results)} passed, {failed} failed, {errors} errors.")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    if args.fail_on_error and (failed or errors):
        return 1
    return 0


async def run_realtime_case(
    case: dict[str, Any],
    *,
    tool_mode: str,
    db_path: Path,
    audio_dir: Path,
    voice: str,
    tts_rate: int,
    turn_timeout_s: float,
    connect_timeout_s: float,
    chunk_ms: int,
    send_speed: float,
    overwrite_audio: bool,
) -> RealtimeCaseResult:
    """Run one case through a real Qwen realtime session."""
    case_id = str(case.get("id") or "case")
    title = str(case.get("title") or case_id)
    started = time.perf_counter()
    previous_env = _apply_case_env(db_path, tool_mode)
    refresh_runtime_config_from_env()
    probe = RealtimeProbe(mode=tool_mode, case_id=case_id)
    task: asyncio.Task[None] | None = None
    handler: ProbeQwenHandler | None = None
    session_id: str | None = None
    try:
        deps = ToolDependencies(reachy_mini=object(), movement_manager=MovementManagerStub())
        handler = ProbeQwenHandler(deps, probe=probe)
        _seed_runtime(handler.memory_runtime, case.get("seed", {}))
        connect_started = time.perf_counter()
        task = asyncio.create_task(handler.start_up(), name=f"qwen-realtime-{tool_mode}-{case_id}")
        await asyncio.wait_for(handler._connected_event.wait(), timeout=connect_timeout_s)
        connect_ms = _elapsed_ms(connect_started)
        session_id = handler.memory_runtime.current_session_id
        await _wait_for_session_updated(probe, timeout_s=connect_timeout_s)

        turn_started = time.perf_counter()
        for turn in _turns(case):
            if str(turn.get("role") or "user") != "user":
                continue
            text = str(turn.get("text") or "")
            wav_path = _ensure_tts_wav(
                case_id=case_id,
                turn_index=int(turn.get("index", 0)) if str(turn.get("index", "")).isdigit() else 0,
                text=text,
                audio_dir=audio_dir,
                voice=voice,
                rate=tts_rate,
                overwrite=overwrite_audio,
            )
            await _send_wav(handler, wav_path, chunk_ms=chunk_ms, send_speed=send_speed)
            await _wait_for_turn_outputs(handler, probe, timeout_s=turn_timeout_s)
        turns_ms = _elapsed_ms(turn_started)

        shutdown_started = time.perf_counter()
        await handler.shutdown()
        if task is not None:
            await asyncio.wait_for(task, timeout=10.0)
        end_session_ms = _elapsed_ms(shutdown_started)

        context_started = time.perf_counter()
        memory_context = handler.memory_runtime.build_memory_context()
        metrics = {
            "connect_ms": connect_ms,
            "session_update_ack_ms": _delta_ms(probe.session_update_sent_at, probe.session_updated_at),
            "audio_to_user_transcript_ms": _delta_ms(turn_started, probe.first_user_transcript_at),
            "audio_to_first_audio_ms": _delta_ms(turn_started, probe.first_audio_at),
            "audio_to_assistant_transcript_ms": _delta_ms(turn_started, probe.assistant_transcript_done_at),
            "content_done_to_speech_stopped_ms": _delta_ms(probe.content_audio_done_at, probe.speech_stopped_at),
            "speech_stopped_to_first_audio_ms": _delta_ms(probe.speech_stopped_at, probe.first_audio_at),
            "content_done_to_first_audio_ms": _delta_ms(probe.content_audio_done_at, probe.first_audio_at),
            "vad_silence_duration_ms": probe.session_update.get("session", {})
            .get("turn_detection", {})
            .get("silence_duration_ms"),
            "audio_turn_ms": turns_ms,
            "end_session_ms": end_session_ms,
            "context_build_ms": _elapsed_ms(context_started),
            "context_chars": len(memory_context),
            "total_ms": _elapsed_ms(started),
            "audio_chunks": probe.audio_chunks,
            "output_audio_ms": round(probe.output_audio_ms, 1),
        }
        assertions = evaluate_expectations(case, handler.memory_runtime, memory_context, session_id, metrics)
        assertions.extend(_realtime_assertions(probe, tool_mode))
        status = "passed" if all(assertion.passed for assertion in assertions) else "failed"
        return RealtimeCaseResult(
            case_id=case_id,
            title=title,
            mode=tool_mode,
            status=status,
            assertions=[asdict(assertion) for assertion in assertions],
            metrics=metrics,
            db_path=str(db_path),
            memory_context=memory_context,
            user_transcripts=probe.user_transcripts,
            assistant_transcripts=probe.assistant_transcripts,
            system_outputs=probe.system_outputs,
            event_counts=probe.event_counts,
            session_tools_count=len(probe.session_update.get("session", {}).get("tools", [])),
            error="; ".join(probe.errors) or None,
        )
    except Exception as exc:
        if handler is not None:
            try:
                await handler.shutdown()
            except Exception:
                pass
        if task is not None:
            task.cancel()
            try:
                await task
            except Exception:
                pass
        return RealtimeCaseResult(
            case_id=case_id,
            title=title,
            mode=tool_mode,
            status="error",
            assertions=[],
            metrics={"total_ms": _elapsed_ms(started)},
            db_path=str(db_path),
            memory_context="",
            user_transcripts=probe.user_transcripts,
            assistant_transcripts=probe.assistant_transcripts,
            system_outputs=probe.system_outputs,
            event_counts=probe.event_counts,
            session_tools_count=len(probe.session_update.get("session", {}).get("tools", [])),
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        _restore_env(previous_env)
        refresh_runtime_config_from_env()


async def _wait_for_session_updated(probe: RealtimeProbe, *, timeout_s: float) -> None:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if probe.session_updated_at is not None or probe.errors:
            return
        await asyncio.sleep(0.05)


async def _wait_for_turn_outputs(
    handler: ProbeQwenHandler,
    probe: RealtimeProbe,
    *,
    timeout_s: float,
) -> None:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        remaining = max(0.05, min(0.5, deadline - time.perf_counter()))
        try:
            item = await asyncio.wait_for(handler.output_queue.get(), timeout=remaining)
        except asyncio.TimeoutError:
            if _turn_has_minimum_outputs(probe):
                return
            continue
        _record_output(item, probe)
        if _turn_has_minimum_outputs(probe):
            return
    probe.errors.append("turn output timeout")


def _turn_has_minimum_outputs(probe: RealtimeProbe) -> bool:
    return bool(probe.user_transcripts) and (
        bool(probe.assistant_transcripts) or probe.response_done_at is not None
    )


def _record_output(item: Any, probe: RealtimeProbe) -> None:
    if isinstance(item, tuple):
        sample_rate, audio = item
        samples = int(audio.size)
        probe.output_audio_ms += samples / int(sample_rate) * 1000.0
        return
    if not isinstance(item, AdditionalOutputs):
        return
    payload = item.args[0] if item.args else {}
    if not isinstance(payload, dict):
        return
    role = str(payload.get("role") or "")
    content = str(payload.get("content") or "")
    if role == "user":
        _append_once(probe.user_transcripts, content)
    elif role == "assistant":
        _append_once(probe.assistant_transcripts, content)
    else:
        _append_once(probe.system_outputs, content)


def _append_once(values: list[str], value: str) -> None:
    value = value.strip()
    if value and value not in values:
        values.append(value)


async def _send_wav(
    handler: ProbeQwenHandler,
    wav_path: Path,
    *,
    chunk_ms: int,
    send_speed: float,
) -> None:
    sample_rate, samples = _read_wav_int16(wav_path)
    chunk_size = max(1, int(sample_rate * chunk_ms / 1000))
    sleep_s = max(0.0, (chunk_ms / 1000) / max(send_speed, 0.1))
    for start in range(0, len(samples), chunk_size):
        chunk = samples[start : start + chunk_size].reshape(-1, 1)
        await handler.receive((sample_rate, chunk))
        handler.probe.audio_chunks += 1
        if sleep_s:
            await asyncio.sleep(sleep_s)
    if handler.probe.content_audio_done_at is None:
        handler.probe.content_audio_done_at = time.perf_counter()
    silence = np.zeros((int(sample_rate * 1.3), 1), dtype=np.int16)
    for start in range(0, len(silence), chunk_size):
        await handler.receive((sample_rate, silence[start : start + chunk_size]))
        handler.probe.audio_chunks += 1
        if sleep_s:
            await asyncio.sleep(sleep_s)


def _read_wav_int16(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError(f"WAV must be mono PCM16: {path}")
        sample_rate = wav.getframerate()
        data = wav.readframes(wav.getnframes())
    return sample_rate, np.frombuffer(data, dtype="<i2").astype(np.int16)


def _ensure_tts_wav(
    *,
    case_id: str,
    turn_index: int,
    text: str,
    audio_dir: Path,
    voice: str,
    rate: int,
    overwrite: bool,
) -> Path:
    audio_dir.mkdir(parents=True, exist_ok=True)
    wav_path = audio_dir / f"{_safe_name(case_id)}_turn{turn_index:02d}.wav"
    if wav_path.exists() and not overwrite:
        return wav_path
    _require_command("say")
    _require_command("afconvert")
    resolved_voice = _resolve_voice(voice)
    with tempfile.TemporaryDirectory() as tmp_dir:
        aiff_path = Path(tmp_dir) / "tts.aiff"
        say_cmd = ["say", "-r", str(rate), "-o", str(aiff_path)]
        if resolved_voice:
            say_cmd.extend(["-v", resolved_voice])
        say_cmd.append(text)
        subprocess.run(say_cmd, check=True)
        subprocess.run(
            [
                "afconvert",
                "-f",
                "WAVE",
                "-d",
                f"LEI16@{QWEN_INPUT_SAMPLE_RATE}",
                "-c",
                "1",
                str(aiff_path),
                str(wav_path),
            ],
            check=True,
        )
    return wav_path


def _resolve_voice(configured: str) -> str | None:
    configured = configured.strip()
    if configured.lower() in {"", "default", "system", "none"}:
        return None
    if configured.lower() != "auto":
        return configured
    voices = _available_voice_names()
    for preferred in ("Tingting", "Grandma", "Grandpa", "Meijia", "Sinji"):
        if preferred in voices:
            return preferred
    for name, locale in voices.items():
        if locale.startswith("zh_"):
            return name
    return None


def _available_voice_names() -> dict[str, str]:
    output = subprocess.run(["say", "-v", "?"], check=True, capture_output=True, text=True).stdout
    voices: dict[str, str] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        locale_index = next((index for index, part in enumerate(parts) if "_" in part), None)
        if locale_index is None:
            continue
        voices[" ".join(parts[:locale_index])] = parts[locale_index]
    return voices


def _require_command(command: str) -> None:
    subprocess.run(["/usr/bin/which", command], check=True, capture_output=True, text=True)


def _realtime_assertions(probe: RealtimeProbe, tool_mode: str) -> list[Any]:
    from reachy_mini_conversation_app.memory.eval import AssertionResult

    session = probe.session_update.get("session", {})
    tools = session.get("tools", [])
    return [
        AssertionResult(
            "realtime_session_updated",
            probe.session_updated_at is not None,
            "session.updated received" if probe.session_updated_at else "session.updated missing",
            severity="P1",
        ),
        AssertionResult(
            "realtime_user_transcript",
            bool(probe.user_transcripts),
            f"user transcript {probe.user_transcripts!r}" if probe.user_transcripts else "user transcript missing",
            severity="P1",
        ),
        AssertionResult(
            "realtime_audio_or_assistant",
            probe.first_audio_at is not None and (bool(probe.assistant_transcripts) or probe.response_done_at is not None),
            (
                "assistant audio and completion received"
                if probe.first_audio_at and (probe.assistant_transcripts or probe.response_done_at)
                else "assistant audio/completion missing"
            ),
            severity="P1",
        ),
        AssertionResult(
            "realtime_errors",
            not probe.errors,
            "no realtime errors" if not probe.errors else "; ".join(probe.errors),
            severity="P1",
        ),
        AssertionResult(
            "native_tools_presence",
            (tool_mode != "native") or bool(tools),
            "native mode included tools" if tools else "native tools not required for router mode",
            severity="P2",
        ),
    ]


def _apply_case_env(db_path: Path, tool_mode: str) -> dict[str, str | None]:
    keys = [
        "BACKEND_PROVIDER",
        "QWEN_TOOL_MODE",
        "REACHY_MINI_MEMORY_DB_PATH",
        "REACHY_MINI_MEMORY_EXTRACTOR",
        "REACHY_MINI_MEMORY_WRITE_MODE",
    ]
    previous = {key: os.environ.get(key) for key in keys}
    os.environ["BACKEND_PROVIDER"] = "qwen_omni"
    os.environ["QWEN_TOOL_MODE"] = tool_mode
    os.environ["REACHY_MINI_MEMORY_DB_PATH"] = str(db_path)
    os.environ["REACHY_MINI_MEMORY_EXTRACTOR"] = os.environ.get("REACHY_MINI_MEMORY_EXTRACTOR", "qwen")
    os.environ["REACHY_MINI_MEMORY_WRITE_MODE"] = "extractor_only"
    return previous


def _restore_env(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def write_realtime_reports(results: list[RealtimeCaseResult], report_dir: str | Path) -> tuple[Path, Path]:
    """Write JSON and Markdown realtime reports."""
    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"qwen_realtime_tts_eval_{stamp}.json"
    md_path = output_dir / f"qwen_realtime_tts_eval_{stamp}.md"
    payload = {
        "summary": _summary(results),
        "results": [asdict(result) for result in results],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(results), encoding="utf-8")
    return json_path, md_path


def _summary(results: list[RealtimeCaseResult]) -> dict[str, Any]:
    total = len(results)
    passed = sum(result.status == "passed" for result in results)
    failed = sum(result.status == "failed" for result in results)
    errors = sum(result.status == "error" for result in results)
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "pass_rate": round(passed / total, 4) if total else 0.0,
    }


def _render_markdown(results: list[RealtimeCaseResult]) -> str:
    summary = _summary(results)
    lines = [
        "# Qwen Realtime TTS Evaluation Report",
        "",
        f"- Total: {summary['total']}",
        f"- Passed: {summary['passed']}",
        f"- Failed: {summary['failed']}",
        f"- Errors: {summary['errors']}",
        f"- Pass rate: {summary['pass_rate']:.2%}",
        "",
        "| Case | Mode | Status | Total ms | Connect ms | Last audio->first audio ms | VAD stop->first audio ms | Context chars | Issues |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        failed = [assertion["message"] for assertion in result.assertions if not assertion.get("passed")]
        issues = "; ".join(failed) or (result.error or "")
        metrics = result.metrics
        lines.append(
            f"| {result.case_id} | {result.mode} | {result.status} | "
            f"{metrics.get('total_ms', 0):.1f} | {metrics.get('connect_ms', 0):.1f} | "
            f"{metrics.get('content_done_to_first_audio_ms') or 0:.1f} | "
            f"{metrics.get('speech_stopped_to_first_audio_ms') or 0:.1f} | "
            f"{metrics.get('context_chars', 0)} | {_escape_table(issues[:220])} |"
        )
    lines.append("")
    for result in results:
        lines.extend(
            [
                f"## {result.case_id} {result.title} ({result.mode})",
                "",
                f"- Status: `{result.status}`",
                f"- Metrics: `{json.dumps(result.metrics, ensure_ascii=False)}`",
                f"- Session tools count: `{result.session_tools_count}`",
                f"- User transcripts: `{json.dumps(result.user_transcripts, ensure_ascii=False)}`",
                f"- Assistant transcripts: `{json.dumps(result.assistant_transcripts, ensure_ascii=False)}`",
            ]
        )
        if result.error:
            lines.append(f"- Error: `{result.error}`")
        for assertion in result.assertions:
            mark = "PASS" if assertion.get("passed") else "FAIL"
            lines.append(f"- {mark} [{assertion.get('severity')}] {assertion.get('name')}: {assertion.get('message')}")
        lines.append("")
    return "\n".join(lines)


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _delta_ms(started: float | None, ended: float | None) -> float | None:
    if started is None or ended is None:
        return None
    return (ended - started) * 1000


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async()))
