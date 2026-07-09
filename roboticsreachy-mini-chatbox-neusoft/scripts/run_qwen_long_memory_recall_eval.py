#!/usr/bin/env python3
"""Run a long transcript memory extraction and Qwen realtime recall evaluation."""

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
from run_qwen_realtime_recall_eval import _run_realtime_session

from reachy_mini_conversation_app.config import refresh_runtime_config_from_env
from reachy_mini_conversation_app.memory.eval import AssertionResult
from reachy_mini_conversation_app.memory.store import SQLiteMemoryStore
from reachy_mini_conversation_app.memory.runtime import MemoryRuntime


@dataclass(slots=True)
class RecallResult:
    """One realtime recall prompt result."""

    prompt_id: str
    prompt: str
    status: str
    assistant_text: str
    assertions: list[dict[str, Any]]
    metrics: dict[str, Any]
    recall_memory_context: str
    user_transcripts: list[str]
    error: str | None = None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="tests/memory_scenarios/eldercare_long_conversation.json")
    parser.add_argument("--db", default="/tmp/reachy_long_memory_eval.sqlite3")
    parser.add_argument("--report-dir", default="/tmp/reachy_long_memory_eval_reports")
    parser.add_argument("--audio-dir", default="tests/realtime_audio_fixtures")
    parser.add_argument("--tool-mode", choices=["router", "native"], default="router")
    parser.add_argument("--voice", default="auto")
    parser.add_argument("--tts-rate", type=int, default=165)
    parser.add_argument("--turn-timeout-s", type=float, default=45.0)
    parser.add_argument("--connect-timeout-s", type=float, default=15.0)
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--send-speed", type=float, default=4.0)
    parser.add_argument("--memory-timeout-s", type=float, default=90.0)
    parser.add_argument("--overwrite-audio", action="store_true")
    parser.add_argument("--allow-real-api", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    return parser.parse_args()


async def main_async() -> int:
    """Run the long-conversation storage and realtime recall test."""
    load_dotenv(override=True)
    args = parse_args()
    if not args.allow_real_api:
        raise SystemExit("Long realtime recall test requires --allow-real-api")
    if not (os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")):
        raise SystemExit("DASHSCOPE_API_KEY or QWEN_API_KEY is required")

    scenario = _load_json(args.scenario)
    db_path = Path(args.db)
    _reset_sqlite_files(db_path)
    previous_env = _apply_env(db_path, args.tool_mode, extractor="qwen", memory_timeout_s=args.memory_timeout_s)
    refresh_runtime_config_from_env()
    try:
        extraction_payload = await run_long_extraction(scenario, db_path)
    finally:
        _restore_env(previous_env)
        refresh_runtime_config_from_env()

    recall_results: list[RecallResult] = []
    previous_env = _apply_env(db_path, args.tool_mode, extractor="none", memory_timeout_s=args.memory_timeout_s)
    refresh_runtime_config_from_env()
    try:
        for recall_spec in scenario.get("recall_prompts", []):
            if isinstance(recall_spec, dict):
                result = await run_realtime_recall(
                    scenario=scenario,
                    recall_spec=recall_spec,
                    db_path=db_path,
                    tool_mode=args.tool_mode,
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
                recall_results.append(result)
    finally:
        _restore_env(previous_env)
        refresh_runtime_config_from_env()

    report_paths = write_reports(
        extraction_payload=extraction_payload,
        recall_results=recall_results,
        report_dir=Path(args.report_dir),
        db_path=db_path,
    )
    passed = sum(result.status == "passed" for result in recall_results)
    failed = sum(result.status == "failed" for result in recall_results)
    errors = sum(result.status == "error" for result in recall_results)
    print(f"Long memory extraction complete: db={db_path}")
    print(f"Realtime recall complete: {passed}/{len(recall_results)} passed, {failed} failed, {errors} errors.")
    print(f"JSON report: {report_paths[0]}")
    print(f"Markdown report: {report_paths[1]}")
    if args.fail_on_error and (failed or errors):
        return 1
    return 0


async def run_long_extraction(scenario: dict[str, Any], db_path: Path) -> dict[str, Any]:
    """Record the long transcript into SQLite and run the real session-end extractor."""
    started = time.perf_counter()
    runtime = MemoryRuntime(SQLiteMemoryStore(db_path))
    session_id = runtime.start_session({"eval_case_id": "long_conversation", "eval_mode": "qwen-extractor-long"})
    turns = [turn for turn in scenario.get("turns", []) if isinstance(turn, dict)]
    for turn in turns:
        role = str(turn.get("role") or "user")
        text = str(turn.get("text") or "")
        if role == "assistant":
            runtime.record_assistant_transcript(text, metadata={"source": "long_memory_eval"})
        else:
            runtime.record_user_transcript(text, metadata={"source": "long_memory_eval"})

    end_started = time.perf_counter()
    await runtime.end_session(reason="long_memory_eval")
    end_session_ms = _elapsed_ms(end_started)
    context_started = time.perf_counter()
    memory_context = runtime.build_memory_context()
    snapshot = inspect_runtime(runtime, memory_context)
    snapshot["metrics"] = {
        "total_ms": _elapsed_ms(started),
        "end_session_ms": end_session_ms,
        "context_build_ms": _elapsed_ms(context_started),
        "context_chars": len(memory_context),
        "turn_count": len(turns),
        "transcript_chars": sum(len(str(turn.get("text") or "")) for turn in turns),
    }
    snapshot["session_id"] = session_id
    snapshot["memory_context"] = memory_context
    return snapshot


async def run_realtime_recall(
    *,
    scenario: dict[str, Any],
    recall_spec: dict[str, Any],
    db_path: Path,
    tool_mode: str,
    audio_dir: Path,
    voice: str,
    tts_rate: int,
    turn_timeout_s: float,
    connect_timeout_s: float,
    memory_timeout_s: float,
    chunk_ms: int,
    send_speed: float,
    overwrite_audio: bool,
) -> RecallResult:
    """Ask one recall prompt through real Qwen realtime using the stored DB."""
    prompt_id = str(recall_spec.get("id") or "recall")
    prompt = str(recall_spec.get("prompt") or "")
    try:
        run = await _run_realtime_session(
            label=f"long-{prompt_id}",
            case={"id": "long_conversation", "title": scenario.get("description", "long conversation")},
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
            user_turns=[prompt],
        )
        assistant_text = "\n".join(run.probe.assistant_transcripts).strip()
        assertions = _assert_recall(recall_spec, assistant_text)
        assertions.extend(_assert_context_safety(run.injected_memory_context))
        status = "passed" if all(assertion.passed for assertion in assertions) else "failed"
        return RecallResult(
            prompt_id=prompt_id,
            prompt=prompt,
            status=status,
            assistant_text=assistant_text,
            assertions=[asdict(assertion) for assertion in assertions],
            metrics=run.metrics,
            recall_memory_context=run.injected_memory_context,
            user_transcripts=run.probe.user_transcripts,
            error="; ".join(run.probe.errors) or None,
        )
    except Exception as exc:
        return RecallResult(
            prompt_id=prompt_id,
            prompt=prompt,
            status="error",
            assistant_text="",
            assertions=[],
            metrics={},
            recall_memory_context="",
            user_transcripts=[],
            error=f"{type(exc).__name__}: {exc}",
        )


def inspect_runtime(runtime: MemoryRuntime, memory_context: str) -> dict[str, Any]:
    """Return a compact storage snapshot for reports."""
    facts = runtime.list_user_profile(include_pending=True)
    tasks = [
        runtime._task_to_dict(task)
        for task in runtime.store.list_care_tasks(
            runtime.user.id,
            statuses=("active", "pending_confirmation", "completed", "disabled", "archived"),
            limit=100,
        )
    ]
    all_notes = [
        runtime._note_to_dict(note)
        for note in runtime.store.list_memory_notes(
            runtime.user.id,
            statuses=("active", "pending_confirmation", "archived"),
            limit=100,
        )
    ]
    occurrences = [
        {
            "id": occurrence.id,
            "task_id": occurrence.task_id,
            "occurrence_key": occurrence.occurrence_key,
            "status": occurrence.status,
            "completed_at": occurrence.completed_at,
        }
        for occurrence in runtime.store.list_care_task_occurrences(
            runtime.user.id,
            statuses=("completed", "skipped", "archived"),
            limit=100,
        )
    ]
    sessions = runtime.store.get_recent_sessions(runtime.user.id, limit=20)
    return {
        "counts": {
            "profile_facts": len(facts),
            "profile_active": sum(1 for fact in facts if fact["status"] == "active"),
            "profile_pending": sum(1 for fact in facts if fact["status"] == "pending_confirmation"),
            "profile_archived": sum(1 for fact in facts if fact["status"] == "archived"),
            "care_tasks": len(tasks),
            "care_active": sum(1 for task in tasks if task["status"] == "active"),
            "care_pending": sum(1 for task in tasks if task["status"] == "pending_confirmation"),
            "care_completed": sum(1 for task in tasks if task["status"] == "completed"),
            "care_disabled": sum(1 for task in tasks if task["status"] == "disabled"),
            "task_occurrences": len(occurrences),
            "memory_notes": len(all_notes),
            "memory_notes_active": sum(1 for note in all_notes if note["status"] == "active"),
            "memory_notes_pending": sum(1 for note in all_notes if note["status"] == "pending_confirmation"),
            "memory_context_chars": len(memory_context),
            "sessions": len(sessions),
        },
        "profile_facts": facts,
        "care_tasks": tasks,
        "care_task_occurrences": occurrences,
        "memory_notes": all_notes,
        "sessions": [
            {
                "id": session.id,
                "status": session.status,
                "reason": session.reason,
                "summary": session.summary,
                "started_at": session.started_at,
                "ended_at": session.ended_at,
            }
            for session in sessions
        ],
    }


def _assert_recall(recall_spec: dict[str, Any], assistant_text: str) -> list[AssertionResult]:
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


def _assert_context_safety(memory_context: str) -> list[AssertionResult]:
    sensitive_tokens = ["幸福路", "阿司匹林", "血压偏高", "保证金"]
    return [
        AssertionResult(
            f"context_not_contains_sensitive[{index}]",
            token not in memory_context,
            f"context excludes {token!r}" if token not in memory_context else f"context leaked {token!r}",
            severity="P1",
        )
        for index, token in enumerate(sensitive_tokens)
    ]


def write_reports(
    *,
    extraction_payload: dict[str, Any],
    recall_results: list[RecallResult],
    report_dir: Path,
    db_path: Path,
) -> tuple[Path, Path]:
    """Write JSON and Markdown reports."""
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = report_dir / f"qwen_long_memory_recall_{stamp}.json"
    md_path = report_dir / f"qwen_long_memory_recall_{stamp}.md"
    payload = {
        "db_path": str(db_path),
        "extraction": extraction_payload,
        "recall_summary": _summary(recall_results),
        "recall_results": [asdict(result) for result in recall_results],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    return json_path, md_path


def _render_markdown(payload: dict[str, Any]) -> str:
    extraction = payload["extraction"]
    counts = extraction["counts"]
    summary = payload["recall_summary"]
    lines = [
        "# Qwen Long Memory Recall Evaluation",
        "",
        f"- DB: `{payload['db_path']}`",
        f"- Turns: `{extraction['metrics']['turn_count']}`",
        f"- Transcript chars: `{extraction['metrics']['transcript_chars']}`",
        f"- Extractor end_session ms: `{extraction['metrics']['end_session_ms']:.1f}`",
        f"- Memory context chars: `{extraction['metrics']['context_chars']}`",
        f"- Recall pass rate: `{summary['passed']}/{summary['total']}`",
        "",
        "## Performance Summary",
        "",
        "### Extraction And Context",
        "",
        "| 指标说明 | 字段 | 数值 |",
        "|---|---|---:|",
        f"| 本轮长对话 turn 数 | turn_count | {extraction['metrics']['turn_count']} |",
        f"| 本轮 transcript 总字符数 | transcript_chars | {extraction['metrics']['transcript_chars']} |",
        f"| session end 后 extractor 抽取、写库、结束 session 的耗时 | extractor_end_session_ms | {_format_ms(extraction['metrics'].get('end_session_ms'))} |",
        f"| 本地从 SQLite 构建下一轮 memory context 的耗时 | context_build_ms | {_format_ms(extraction['metrics'].get('context_build_ms'))} |",
        f"| 下一轮会注入 realtime instructions 的 memory context 字符数 | memory_context_chars | {extraction['metrics']['context_chars']} |",
        f"| 长对话 extraction 阶段总耗时 | total_extraction_phase_ms | {_format_ms(extraction['metrics'].get('total_ms'))} |",
        "",
        "### Realtime Recall Aggregates",
        "",
        "| 指标说明 | 字段 | 平均 ms | 最大 ms |",
        "|---|---|---:|---:|",
        _metric_row(payload["recall_results"], "connect_ms"),
        _metric_row(payload["recall_results"], "session_update_ack_ms"),
        _metric_row(payload["recall_results"], "audio_to_user_transcript_ms"),
        _metric_row(payload["recall_results"], "speech_stopped_to_first_audio_ms"),
        _metric_row(payload["recall_results"], "content_done_to_first_audio_ms"),
        _metric_row(payload["recall_results"], "audio_to_assistant_transcript_ms"),
        _metric_row(payload["recall_results"], "end_session_ms"),
        _metric_row(payload["recall_results"], "context_build_ms"),
        "",
        "### Realtime Recall By Prompt",
        "",
        _prompt_metric_header(payload["recall_results"]),
        _prompt_metric_separator(payload["recall_results"]),
    ]
    for key in _PROMPT_METRIC_KEYS:
        lines.append(_prompt_metric_row(payload["recall_results"], key))
    lines.extend(
        [
            "",
        "## Storage Counts",
        "",
        "| Item | Count |",
        "|---|---:|",
        ]
    )
    for key, value in counts.items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Profile Facts",
            "",
            "| Key | Value | Category | Status | Source |",
            "|---|---|---|---|---|",
        ]
    )
    for fact in extraction["profile_facts"]:
        lines.append(
            f"| {_escape(fact['key'])} | {_escape(fact['value'])} | {_escape(fact['category'])} | "
            f"{_escape(fact['status'])} | {_escape(fact['source'])} |"
        )
    lines.extend(["", "## Care Tasks", "", "| Title | Type | Status | Due | Repeat |", "|---|---|---|---|---|"])
    for task in extraction["care_tasks"]:
        lines.append(
            f"| {_escape(task['title'])} | {_escape(task['task_type'])} | {_escape(task['status'])} | "
            f"{_escape(task.get('due_at') or '')} | {_escape(task.get('recurrence_rule') or '')} |"
        )
    lines.extend(
        [
            "",
            "## Care Task Occurrences",
            "",
            "| Task ID | Occurrence | Status | Completed At |",
            "|---|---|---|---|",
        ]
    )
    for occurrence in extraction.get("care_task_occurrences", []):
        lines.append(
            f"| {_escape(occurrence['task_id'])} | {_escape(occurrence['occurrence_key'])} | "
            f"{_escape(occurrence['status'])} | {_escape(occurrence.get('completed_at') or '')} |"
        )
    lines.extend(["", "## Memory Notes", "", "| Status | Note |", "|---|---|"])
    for note in extraction["memory_notes"]:
        lines.append(f"| {_escape(note['status'])} | {_escape(note['note'])} |")
    lines.extend(
        [
            "",
            "## Realtime Recall",
            "",
            "| Prompt | Status | Last audio->first audio ms | Assistant | Issues |",
            "|---|---|---:|---|---|",
        ]
    )
    for result in payload["recall_results"]:
        failed = [assertion["message"] for assertion in result["assertions"] if not assertion.get("passed")]
        lines.append(
            f"| {_escape(result['prompt_id'])} | {result['status']} | "
            f"{result['metrics'].get('content_done_to_first_audio_ms') or 0:.1f} | "
            f"{_escape(result['assistant_text'][:120])} | {_escape('; '.join(failed)[:180])} |"
        )
    return "\n".join(lines)


def _summary(results: list[RecallResult]) -> dict[str, Any]:
    total = len(results)
    passed = sum(result.status == "passed" for result in results)
    failed = sum(result.status == "failed" for result in results)
    errors = sum(result.status == "error" for result in results)
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "pass_rate": round(passed / total, 4) if total else 0,
    }


def _metric_row(results: list[dict[str, Any]], key: str) -> str:
    values = [
        float(result["metrics"][key])
        for result in results
        if isinstance(result.get("metrics"), dict) and result["metrics"].get(key) is not None
    ]
    if not values:
        return f"| {_metric_description(key)} | {key} | n/a | n/a |"
    return f"| {_metric_description(key)} | {key} | {_format_ms(sum(values) / len(values))} | {_format_ms(max(values))} |"


_PROMPT_METRIC_KEYS = (
    "connect_ms",
    "session_update_ack_ms",
    "audio_to_user_transcript_ms",
    "speech_stopped_to_first_audio_ms",
    "content_done_to_first_audio_ms",
    "audio_to_assistant_transcript_ms",
    "end_session_ms",
    "context_chars",
)


_METRIC_DESCRIPTIONS = {
    "connect_ms": "WebSocket 建连耗时",
    "session_update_ack_ms": "session.update 发送后收到 ack 的耗时",
    "audio_to_user_transcript_ms": "发送用户音频到用户转写完成的耗时",
    "speech_stopped_to_first_audio_ms": "服务端确认用户停顿到 assistant 首包音频的耗时",
    "content_done_to_first_audio_ms": "用户文本完成到 assistant 首包音频的耗时",
    "audio_to_assistant_transcript_ms": "发送用户音频到 assistant 完整 transcript 完成的耗时",
    "end_session_ms": "recall session 结束和后台清理/抽取调度耗时",
    "context_build_ms": "本地构建 memory context 的耗时",
    "context_chars": "本轮注入 realtime instructions 的 memory context 字符数",
}


def _metric_description(key: str) -> str:
    return _METRIC_DESCRIPTIONS.get(key, key)


def _prompt_metric_header(results: list[dict[str, Any]]) -> str:
    prompts = [_escape(str(result.get("prompt_id") or "unknown")) for result in results]
    return "| 指标说明 | 字段 | " + " | ".join(prompts) + " |"


def _prompt_metric_separator(results: list[dict[str, Any]]) -> str:
    return "|---|---|" + "|".join("---:" for _ in results) + "|"


def _prompt_metric_row(results: list[dict[str, Any]], key: str) -> str:
    values = []
    for result in results:
        metrics = result.get("metrics") or {}
        value = metrics.get(key)
        values.append(str(value or 0) if key == "context_chars" else _format_ms(value))
    return f"| {_metric_description(key)} | {key} | " + " | ".join(values) + " |"


def _format_ms(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "n/a"


def _apply_env(db_path: Path, tool_mode: str, *, extractor: str, memory_timeout_s: float) -> dict[str, str | None]:
    keys = [
        "BACKEND_PROVIDER",
        "QWEN_TOOL_MODE",
        "REACHY_MINI_MEMORY_DB_PATH",
        "REACHY_MINI_MEMORY_EXTRACTOR",
        "REACHY_MINI_MEMORY_WRITE_MODE",
        "QWEN_MEMORY_TIMEOUT_S",
    ]
    previous = {key: os.environ.get(key) for key in keys}
    os.environ["BACKEND_PROVIDER"] = "qwen_omni"
    os.environ["QWEN_TOOL_MODE"] = tool_mode
    os.environ["REACHY_MINI_MEMORY_DB_PATH"] = str(db_path)
    os.environ["REACHY_MINI_MEMORY_EXTRACTOR"] = extractor
    os.environ["REACHY_MINI_MEMORY_WRITE_MODE"] = "extractor_only"
    os.environ["QWEN_MEMORY_TIMEOUT_S"] = str(memory_timeout_s)
    return previous


def _restore_env(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("scenario file must be a JSON object")
    return data


def _reset_sqlite_files(path: Path) -> None:
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        with contextlib.suppress(FileNotFoundError):
            candidate.unlink()


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _escape(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async()))
