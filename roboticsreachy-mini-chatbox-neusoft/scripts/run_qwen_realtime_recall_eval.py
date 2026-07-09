#!/usr/bin/env python3
"""Run two-session Qwen realtime recall checks for elder-care memory scenarios."""

from __future__ import annotations
import os
import json
import time
import asyncio
import argparse
import contextlib
from typing import Any
from pathlib import Path
from dataclasses import asdict, dataclass

from dotenv import load_dotenv
from run_qwen_realtime_tts_eval import (
    RealtimeProbe,
    ProbeQwenHandler,
    MovementManagerStub,
    _delta_ms,
    _send_wav,
    _safe_name,
    _elapsed_ms,
    _restore_env,
    _record_output,
    _apply_case_env,
    _ensure_tts_wav,
    _realtime_assertions,
    _wait_for_turn_outputs,
    _wait_for_session_updated,
)

from reachy_mini_conversation_app.config import refresh_runtime_config_from_env
from reachy_mini_conversation_app.memory.eval import (
    AssertionResult,
    _turns,
    load_cases,
    _seed_runtime,
    evaluate_expectations,
)
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


@dataclass(slots=True)
class SessionRun:
    """Evidence collected from one realtime session."""

    label: str
    probe: RealtimeProbe
    runtime: Any
    metrics: dict[str, Any]
    memory_context: str
    injected_memory_context: str
    session_id: str | None
    session_tools_count: int


@dataclass(slots=True)
class RecallCaseResult:
    """Serializable result for one two-session recall case."""

    case_id: str
    title: str
    mode: str
    status: str
    recall_prompt: str
    assertions: list[dict[str, Any]]
    metrics: dict[str, Any]
    db_path: str
    write_memory_context: str
    recall_memory_context: str
    recall_assistant_text: str
    write_user_transcripts: list[str]
    recall_user_transcripts: list[str]
    recall_assistant_transcripts: list[str]
    write_event_counts: dict[str, int]
    recall_event_counts: dict[str, int]
    write_session_tools_count: int
    recall_session_tools_count: int
    error: str | None = None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="tests/memory_scenarios/eldercare_expanded_25.json")
    parser.add_argument("--recall-cases", default="tests/memory_scenarios/eldercare_recall_25.json")
    parser.add_argument("--case-id", action="append", help="Run only selected case id. May be repeated.")
    parser.add_argument("--limit", type=int, help="Run at most N selected cases.")
    parser.add_argument("--tool-mode", action="append", choices=["router", "native"], default=None)
    parser.add_argument("--db", default="/tmp/reachy_realtime_recall_eval_dbs", help="SQLite DB directory.")
    parser.add_argument("--report-dir", default="/tmp/reachy_realtime_recall_eval_reports")
    parser.add_argument("--audio-dir", default="tests/realtime_audio_fixtures")
    parser.add_argument("--voice", default="auto", help="macOS say voice. Use auto for a Mandarin voice.")
    parser.add_argument("--tts-rate", type=int, default=165)
    parser.add_argument("--turn-timeout-s", type=float, default=45.0)
    parser.add_argument("--connect-timeout-s", type=float, default=15.0)
    parser.add_argument("--memory-timeout-s", type=float, default=90.0)
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--send-speed", type=float, default=4.0, help="Audio send speed multiplier.")
    parser.add_argument("--overwrite-audio", action="store_true")
    parser.add_argument("--allow-real-api", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    return parser.parse_args()


async def main_async() -> int:
    """Run selected two-session realtime recall scenarios."""
    load_dotenv(override=True)
    args = parse_args()
    if not args.allow_real_api:
        raise SystemExit("Real realtime recall tests require --allow-real-api")
    if not (os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")):
        raise SystemExit("DASHSCOPE_API_KEY or QWEN_API_KEY is required")

    cases = load_cases(args.cases)
    recall_specs = _load_recall_specs(args.recall_cases)
    selected_ids = set(args.case_id or [])
    selected = [
        case
        for case in cases
        if (not selected_ids or str(case.get("id")) in selected_ids) and str(case.get("id")) in recall_specs
    ]
    if args.limit is not None:
        selected = selected[: args.limit]
    missing = selected_ids.difference(str(case.get("id")) for case in selected)
    if missing:
        raise SystemExit(f"Selected case ids missing from cases or recall specs: {sorted(missing)}")

    tool_modes = args.tool_mode or [os.getenv("QWEN_TOOL_MODE", "router")]
    results: list[RecallCaseResult] = []
    for mode in tool_modes:
        for index, case in enumerate(selected):
            db_path = _recall_db_path(Path(args.db) / mode, case, index)
            result = await run_recall_case(
                case,
                recall_spec=recall_specs[str(case.get("id"))],
                tool_mode=mode,
                db_path=db_path,
                audio_dir=Path(args.audio_dir),
                voice=args.voice,
                tts_rate=args.tts_rate,
                turn_timeout_s=args.turn_timeout_s,
                connect_timeout_s=args.connect_timeout_s,
                memory_timeout_s=args.memory_timeout_s,
                chunk_ms=args.chunk_ms,
                send_speed=args.send_speed,
                overwrite_audio=args.overwrite_audio,
            )
            results.append(result)

    json_path, md_path = write_recall_reports(results, args.report_dir)
    passed = sum(result.status == "passed" for result in results)
    failed = sum(result.status == "failed" for result in results)
    errors = sum(result.status == "error" for result in results)
    print(f"Realtime recall eval complete: {passed}/{len(results)} passed, {failed} failed, {errors} errors.")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    if args.fail_on_error and (failed or errors):
        return 1
    return 0


async def run_recall_case(
    case: dict[str, Any],
    *,
    recall_spec: dict[str, Any],
    tool_mode: str,
    db_path: Path,
    audio_dir: Path,
    voice: str,
    tts_rate: int,
    turn_timeout_s: float,
    connect_timeout_s: float,
    memory_timeout_s: float,
    chunk_ms: int,
    send_speed: float,
    overwrite_audio: bool,
) -> RecallCaseResult:
    """Run Session A for memory write, then Session B for recall output."""
    case_id = str(case.get("id") or "case")
    title = str(case.get("title") or case_id)
    started = time.perf_counter()
    _reset_sqlite_files(db_path)
    previous_env = _apply_case_env(db_path, tool_mode)
    refresh_runtime_config_from_env()
    write_run: SessionRun | None = None
    recall_run: SessionRun | None = None
    recall_prompt = str(recall_spec.get("recall_turn") or "你还记得什么和我有关的信息？")
    try:
        write_run = await _run_realtime_session(
            label=f"{case_id}-write",
            case=case,
            tool_mode=tool_mode,
            db_path=db_path,
            audio_dir=audio_dir,
            voice=voice,
            tts_rate=tts_rate,
            turn_timeout_s=turn_timeout_s,
            connect_timeout_s=connect_timeout_s,
            memory_timeout_s=memory_timeout_s,
            chunk_ms=chunk_ms,
            send_speed=send_speed,
            overwrite_audio=overwrite_audio,
            seed=case.get("seed", {}),
            user_turns=[str(turn.get("text") or "") for turn in _turns(case) if str(turn.get("role") or "user") == "user"],
        )
        recall_run = await _run_realtime_session(
            label=f"{case_id}-recall",
            case=case,
            tool_mode=tool_mode,
            db_path=db_path,
            audio_dir=audio_dir,
            voice=voice,
            tts_rate=tts_rate,
            turn_timeout_s=turn_timeout_s,
            connect_timeout_s=connect_timeout_s,
            memory_timeout_s=memory_timeout_s,
            chunk_ms=chunk_ms,
            send_speed=send_speed,
            overwrite_audio=overwrite_audio,
            seed={},
            user_turns=[recall_prompt],
        )
        metrics = {
            "total_ms": _elapsed_ms(started),
            "context_build_ms": write_run.metrics.get("context_build_ms"),
            "context_chars": write_run.metrics.get("context_chars"),
            **{f"write_{key}": value for key, value in write_run.metrics.items()},
            **{f"recall_{key}": value for key, value in recall_run.metrics.items()},
        }
        assertions = _build_recall_assertions(
            case=case,
            recall_spec=recall_spec,
            tool_mode=tool_mode,
            write_run=write_run,
            recall_run=recall_run,
            metrics=metrics,
        )
        status = "passed" if all(assertion.passed for assertion in assertions) else "failed"
        return RecallCaseResult(
            case_id=case_id,
            title=title,
            mode=tool_mode,
            status=status,
            recall_prompt=recall_prompt,
            assertions=[asdict(assertion) for assertion in assertions],
            metrics=metrics,
            db_path=str(db_path),
            write_memory_context=write_run.memory_context,
            recall_memory_context=recall_run.injected_memory_context,
            recall_assistant_text=_assistant_text(recall_run.probe),
            write_user_transcripts=write_run.probe.user_transcripts,
            recall_user_transcripts=recall_run.probe.user_transcripts,
            recall_assistant_transcripts=recall_run.probe.assistant_transcripts,
            write_event_counts=write_run.probe.event_counts,
            recall_event_counts=recall_run.probe.event_counts,
            write_session_tools_count=write_run.session_tools_count,
            recall_session_tools_count=recall_run.session_tools_count,
            error="; ".join(write_run.probe.errors + recall_run.probe.errors) or None,
        )
    except Exception as exc:
        return RecallCaseResult(
            case_id=case_id,
            title=title,
            mode=tool_mode,
            status="error",
            recall_prompt=recall_prompt,
            assertions=[],
            metrics={"total_ms": _elapsed_ms(started)},
            db_path=str(db_path),
            write_memory_context=write_run.memory_context if write_run else "",
            recall_memory_context=recall_run.memory_context if recall_run else "",
            recall_assistant_text=_assistant_text(recall_run.probe) if recall_run else "",
            write_user_transcripts=write_run.probe.user_transcripts if write_run else [],
            recall_user_transcripts=recall_run.probe.user_transcripts if recall_run else [],
            recall_assistant_transcripts=recall_run.probe.assistant_transcripts if recall_run else [],
            write_event_counts=write_run.probe.event_counts if write_run else {},
            recall_event_counts=recall_run.probe.event_counts if recall_run else {},
            write_session_tools_count=write_run.session_tools_count if write_run else 0,
            recall_session_tools_count=recall_run.session_tools_count if recall_run else 0,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        _restore_env(previous_env)
        refresh_runtime_config_from_env()


async def _run_realtime_session(
    *,
    label: str,
    case: dict[str, Any],
    tool_mode: str,
    db_path: Path,
    audio_dir: Path,
    voice: str,
    tts_rate: int,
    turn_timeout_s: float,
    connect_timeout_s: float,
    memory_timeout_s: float,
    chunk_ms: int,
    send_speed: float,
    overwrite_audio: bool,
    seed: Any,
    user_turns: list[str],
) -> SessionRun:
    started = time.perf_counter()
    probe = RealtimeProbe(mode=tool_mode, case_id=label)
    handler: ProbeQwenHandler | None = None
    task: asyncio.Task[None] | None = None
    try:
        deps = ToolDependencies(reachy_mini=object(), movement_manager=MovementManagerStub())
        handler = ProbeQwenHandler(deps, probe=probe)
        _seed_runtime(handler.memory_runtime, seed)
        connect_started = time.perf_counter()
        task = asyncio.create_task(handler.start_up(), name=f"qwen-realtime-recall-{tool_mode}-{label}")
        await asyncio.wait_for(handler._connected_event.wait(), timeout=connect_timeout_s)
        connect_ms = _elapsed_ms(connect_started)
        session_id = handler.memory_runtime.current_session_id
        await _wait_for_session_updated(probe, timeout_s=connect_timeout_s)
        injected_memory_context = handler.memory_runtime.build_memory_context()

        turn_started = time.perf_counter()
        for turn_index, text in enumerate(user_turns):
            wav_path = _ensure_tts_wav(
                case_id=f"{label}_{turn_index}",
                turn_index=0,
                text=text,
                audio_dir=audio_dir,
                voice=voice,
                rate=tts_rate,
                overwrite=overwrite_audio,
            )
            await _send_wav(handler, wav_path, chunk_ms=chunk_ms, send_speed=send_speed)
            await _wait_for_turn_outputs(handler, probe, timeout_s=turn_timeout_s)
            await _drain_outputs(handler, probe)
        turns_ms = _elapsed_ms(turn_started)

        shutdown_started = time.perf_counter()
        await handler.shutdown()
        if task is not None:
            await asyncio.wait_for(task, timeout=10.0)
        end_session_ms = _elapsed_ms(shutdown_started)

        extraction_wait_started = time.perf_counter()
        await handler.memory_runtime.wait_for_pending_extractions(timeout_s=memory_timeout_s)
        extraction_wait_ms = _elapsed_ms(extraction_wait_started)

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
            "turns_ms": turns_ms,
            "end_session_ms": end_session_ms,
            "extraction_wait_ms": extraction_wait_ms,
            "context_build_ms": _elapsed_ms(context_started),
            "context_chars": len(memory_context),
            "total_ms": _elapsed_ms(started),
            "audio_chunks": probe.audio_chunks,
            "output_audio_ms": round(probe.output_audio_ms, 1),
        }
        return SessionRun(
            label=label,
            probe=probe,
            runtime=handler.memory_runtime,
            metrics=metrics,
            memory_context=memory_context,
            injected_memory_context=injected_memory_context,
            session_id=session_id,
            session_tools_count=len(probe.session_update.get("session", {}).get("tools", [])),
        )
    except Exception:
        if handler is not None:
            with contextlib.suppress(Exception):
                await handler.shutdown()
        if task is not None:
            task.cancel()
            with contextlib.suppress(BaseException):
                await task
        raise


async def _drain_outputs(handler: ProbeQwenHandler, probe: RealtimeProbe) -> None:
    while not handler.output_queue.empty():
        _record_output(handler.output_queue.get_nowait(), probe)


def _build_recall_assertions(
    *,
    case: dict[str, Any],
    recall_spec: dict[str, Any],
    tool_mode: str,
    write_run: SessionRun,
    recall_run: SessionRun,
    metrics: dict[str, Any],
) -> list[AssertionResult]:
    assertions: list[AssertionResult] = []
    assertions.extend(
        _prefix_assertions(
            "write",
            evaluate_expectations(case, write_run.runtime, write_run.memory_context, write_run.session_id, metrics),
        )
    )
    assertions.extend(_prefix_assertions("write_realtime", _realtime_assertions(write_run.probe, tool_mode)))
    assertions.extend(_prefix_assertions("recall_realtime", _realtime_assertions(recall_run.probe, tool_mode)))
    assertions.extend(_assert_recall_injection(case, recall_run))
    assertions.extend(_assert_assistant_recall(recall_spec, _assistant_text(recall_run.probe)))
    return assertions


def _assert_recall_injection(case: dict[str, Any], recall_run: SessionRun) -> list[AssertionResult]:
    expectations = case.get("expect", {})
    if not isinstance(expectations, dict):
        expectations = {}
    instructions = str(recall_run.probe.session_update.get("session", {}).get("instructions") or "")
    assertions: list[AssertionResult] = []
    for index, value in enumerate(expectations.get("memory_context_contains", []) or []):
        token = str(value)
        assertions.append(
            AssertionResult(
                f"recall_instructions_contains[{index}]",
                token in instructions,
                f"recall instructions contain {token!r}" if token in instructions else f"recall instructions miss {token!r}",
                severity="P1",
            )
        )
    for index, value in enumerate(expectations.get("memory_context_not_contains", []) or []):
        token = str(value)
        assertions.append(
            AssertionResult(
                f"recall_instructions_not_contains[{index}]",
                token not in instructions,
                f"recall instructions exclude {token!r}"
                if token not in instructions
                else f"recall instructions leaked {token!r}",
                severity="P1",
            )
        )
    return assertions


def _assert_assistant_recall(recall_spec: dict[str, Any], assistant_text: str) -> list[AssertionResult]:
    assertions = [
        AssertionResult(
            "assistant_nonempty",
            bool(assistant_text.strip()),
            "assistant transcript captured" if assistant_text.strip() else "assistant transcript missing",
            severity="P1",
        )
    ]
    for index, token in enumerate(_as_list(recall_spec.get("assistant_contains"))):
        value = str(token)
        assertions.append(
            AssertionResult(
                f"assistant_contains[{index}]",
                value in assistant_text,
                f"assistant mentions {value!r}" if value in assistant_text else f"assistant missing {value!r}",
                severity="P1",
            )
        )
    for index, group in enumerate(_as_list(recall_spec.get("assistant_contains_any"))):
        options = [str(value) for value in _as_list(group)]
        matched = [value for value in options if value in assistant_text]
        assertions.append(
            AssertionResult(
                f"assistant_contains_any[{index}]",
                bool(matched),
                f"assistant mentions one of {options!r}" if matched else f"assistant missing all of {options!r}",
                severity="P1",
                details={"matched": matched, "options": options},
            )
        )
    for index, token in enumerate(_as_list(recall_spec.get("assistant_not_contains"))):
        value = str(token)
        assertions.append(
            AssertionResult(
                f"assistant_not_contains[{index}]",
                value not in assistant_text,
                f"assistant excludes {value!r}" if value not in assistant_text else f"assistant leaked {value!r}",
                severity="P1",
            )
        )
    return assertions


def _prefix_assertions(prefix: str, assertions: list[AssertionResult]) -> list[AssertionResult]:
    return [
        AssertionResult(
            name=f"{prefix}.{assertion.name}",
            passed=assertion.passed,
            message=assertion.message,
            severity=assertion.severity,
            details=assertion.details,
        )
        for assertion in assertions
    ]


def _assistant_text(probe: RealtimeProbe) -> str:
    return "\n".join(text for text in probe.assistant_transcripts if text.strip()).strip()


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _load_recall_specs(path: str | Path) -> dict[str, dict[str, Any]]:
    cases = load_cases(path)
    specs: dict[str, dict[str, Any]] = {}
    for case in cases:
        case_id = str(case.get("id") or "")
        if case_id:
            specs[case_id] = case
    return specs


def _recall_db_path(base: Path, case: dict[str, Any], index: int) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{index:03d}_{_safe_name(str(case.get('id') or 'case'))}.sqlite3"


def _reset_sqlite_files(path: Path) -> None:
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        with contextlib.suppress(FileNotFoundError):
            candidate.unlink()


def write_recall_reports(results: list[RecallCaseResult], report_dir: str | Path) -> tuple[Path, Path]:
    """Write JSON and Markdown recall reports."""
    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"qwen_realtime_recall_eval_{stamp}.json"
    md_path = output_dir / f"qwen_realtime_recall_eval_{stamp}.md"
    payload = {
        "summary": _summary(results),
        "results": [asdict(result) for result in results],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(results), encoding="utf-8")
    return json_path, md_path


def _summary(results: list[RecallCaseResult]) -> dict[str, Any]:
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


def _render_markdown(results: list[RecallCaseResult]) -> str:
    summary = _summary(results)
    lines = [
        "# Qwen Realtime Two-Session Recall Evaluation Report",
        "",
        f"- Total: {summary['total']}",
        f"- Passed: {summary['passed']}",
        f"- Failed: {summary['failed']}",
        f"- Errors: {summary['errors']}",
        f"- Pass rate: {summary['pass_rate']:.2%}",
        "",
        "| Case | Mode | Status | Recall last audio->first audio ms | Recall context chars | Assistant | Issues |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for result in results:
        failed = [assertion["message"] for assertion in result.assertions if not assertion.get("passed")]
        issues = "; ".join(failed) or (result.error or "")
        assistant = _escape_table(result.recall_assistant_text[:120])
        lines.append(
            f"| {result.case_id} | {result.mode} | {result.status} | "
            f"{result.metrics.get('recall_content_done_to_first_audio_ms') or 0:.1f} | "
            f"{result.metrics.get('recall_context_chars', 0)} | {assistant} | {_escape_table(issues[:220])} |"
        )
    lines.append("")
    for result in results:
        lines.extend(
            [
                f"## {result.case_id} {result.title} ({result.mode})",
                "",
                f"- Status: `{result.status}`",
                f"- Recall prompt: `{result.recall_prompt}`",
                f"- Recall assistant: `{result.recall_assistant_text}`",
                f"- Metrics: `{json.dumps(result.metrics, ensure_ascii=False)}`",
                f"- Write user transcripts: `{json.dumps(result.write_user_transcripts, ensure_ascii=False)}`",
                f"- Recall user transcripts: `{json.dumps(result.recall_user_transcripts, ensure_ascii=False)}`",
                f"- Recall assistant transcripts: `{json.dumps(result.recall_assistant_transcripts, ensure_ascii=False)}`",
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


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async()))
