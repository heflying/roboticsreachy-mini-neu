"""Scenario evaluation runner for elder-care memory behavior."""

from __future__ import annotations
import os
import json
import time
import uuid
import tempfile
from typing import Any, Literal
from pathlib import Path
from dataclasses import field, dataclass

from reachy_mini_conversation_app.memory.store import SQLiteMemoryStore
from reachy_mini_conversation_app.memory.models import (
    Turn,
    ExtractionResult,
)
from reachy_mini_conversation_app.memory.policy import memory_command_writes_enabled
from reachy_mini_conversation_app.memory.runtime import MemoryRuntime
from reachy_mini_conversation_app.memory.extractors import NoopMemoryExtractor, _extraction_from_json
from reachy_mini_conversation_app.memory.command_router import MemoryCommandRouter


EvalMode = Literal["case", "offline", "qwen-extractor", "realtime-headless"]


@dataclass(slots=True)
class AssertionResult:
    """One functional or performance assertion result."""

    name: str
    passed: bool
    message: str
    severity: str = "P2"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ScenarioResult:
    """Result for a single memory evaluation scenario."""

    case_id: str
    title: str
    mode: str
    status: str
    assertions: list[AssertionResult]
    metrics: dict[str, Any]
    db_path: str
    memory_context: str
    error: str | None = None

    @property
    def passed(self) -> bool:
        """Return whether all assertions passed and the case was not skipped."""
        return self.status == "passed"


class ScenarioExtractor:
    """Extractor that returns a scenario-provided JSON payload."""

    def __init__(self, payload: dict[str, Any] | None = None):
        """Create a deterministic extractor for offline tests."""
        self.payload = payload or {}

    async def extract(self, *, session_id: str, turns: list[Turn], memory_context: str) -> ExtractionResult:
        """Return the configured extraction result."""
        transcript = "\n".join(f"{turn.role}: {turn.content}" for turn in turns if turn.content.strip())
        return _extraction_from_json(session_id, self.payload, transcript=transcript)


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    """Load evaluation cases from a JSON file."""
    with Path(path).open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    cases = data.get("cases") if isinstance(data, dict) else data
    if not isinstance(cases, list):
        raise ValueError("case file must contain a list or an object with a 'cases' list")
    return [case for case in cases if isinstance(case, dict)]


async def run_cases(
    cases: list[dict[str, Any]],
    *,
    mode: EvalMode = "case",
    db_path: str | Path | None = None,
    keep_db: bool = False,
    allow_real_api: bool = False,
    case_ids: set[str] | None = None,
    limit: int | None = None,
) -> list[ScenarioResult]:
    """Run multiple memory evaluation scenarios."""
    selected = [case for case in cases if not case_ids or str(case.get("id")) in case_ids]
    if limit is not None:
        selected = selected[:limit]

    results: list[ScenarioResult] = []
    for index, case in enumerate(selected):
        case_db = _case_db_path(db_path, case, index, keep_db)
        result = await run_case(
            case,
            mode=mode,
            db_path=case_db,
            allow_real_api=allow_real_api,
        )
        results.append(result)
        if not keep_db and db_path is None and result.db_path:
            _remove_sqlite_files(result.db_path)
    return results


async def run_case(
    case: dict[str, Any],
    *,
    mode: EvalMode = "case",
    db_path: str | Path | None = None,
    allow_real_api: bool = False,
) -> ScenarioResult:
    """Run one scenario and evaluate expectations."""
    case_id = str(case.get("id") or f"case-{uuid.uuid4().hex[:8]}")
    title = str(case.get("title") or case_id)
    effective_mode = _resolve_mode(case, mode)
    if effective_mode == "qwen-extractor" and not allow_real_api:
        return ScenarioResult(
            case_id=case_id,
            title=title,
            mode=effective_mode,
            status="skipped",
            assertions=[],
            metrics={},
            db_path=str(db_path or ""),
            memory_context="",
            error="real Qwen extractor requires --allow-real-api",
        )

    db_file = Path(db_path) if db_path else _temporary_db_path(case_id)
    metrics: dict[str, Any] = {}
    started = time.perf_counter()
    try:
        if effective_mode == "realtime-headless":
            runtime, session_id, setup_metrics = await _run_realtime_headless_case(
                case,
                db_file,
                allow_real_api=allow_real_api,
            )
        else:
            runtime, session_id, setup_metrics = await _run_runtime_case(
                case,
                db_file,
                mode=effective_mode,
            )
        metrics.update(setup_metrics)
        context_started = time.perf_counter()
        memory_context = runtime.build_memory_context()
        metrics["context_build_ms"] = _elapsed_ms(context_started)
        metrics["context_chars"] = len(memory_context)
        metrics["total_ms"] = _elapsed_ms(started)

        assertions = evaluate_expectations(case, runtime, memory_context, session_id, metrics)
        status = "passed" if all(assertion.passed for assertion in assertions) else "failed"
        return ScenarioResult(
            case_id=case_id,
            title=title,
            mode=effective_mode,
            status=status,
            assertions=assertions,
            metrics=metrics,
            db_path=str(db_file),
            memory_context=memory_context,
        )
    except Exception as exc:
        return ScenarioResult(
            case_id=case_id,
            title=title,
            mode=effective_mode,
            status="error",
            assertions=[],
            metrics={**metrics, "total_ms": _elapsed_ms(started)},
            db_path=str(db_file),
            memory_context="",
            error=f"{type(exc).__name__}: {exc}",
        )


async def _run_runtime_case(
    case: dict[str, Any],
    db_file: Path,
    *,
    mode: str,
) -> tuple[MemoryRuntime, str | None, dict[str, Any]]:
    extractor = _extractor_for_mode(case, mode)
    metrics: dict[str, Any] = {}
    setup_started = time.perf_counter()
    runtime = MemoryRuntime(SQLiteMemoryStore(db_file), extractor=extractor)
    _seed_runtime(runtime, case.get("seed", {}))
    session_id = runtime.start_session({"eval_case_id": case.get("id"), "eval_mode": mode})
    metrics["setup_ms"] = _elapsed_ms(setup_started)

    router = MemoryCommandRouter(runtime)
    turn_started = time.perf_counter()
    for turn in _turns(case):
        role = turn.get("role", "user")
        text = str(turn.get("text") or "")
        if role == "assistant":
            runtime.record_assistant_transcript(text, metadata={"source": "memory_eval"})
            continue
        runtime.record_user_transcript(text, metadata={"source": "memory_eval"})
        if memory_command_writes_enabled() and (mode in {"router", "offline"} or str(case.get("runner")) == "router"):
            await router.handle(text)
    if memory_command_writes_enabled():
        _execute_tool_calls(runtime, case.get("tool_calls", []))
    metrics["turns_ms"] = _elapsed_ms(turn_started)

    if case.get("end_session", True):
        end_started = time.perf_counter()
        await runtime.end_session(reason="memory_eval")
        metrics["end_session_ms"] = _elapsed_ms(end_started)
    return runtime, session_id, metrics


async def _run_realtime_headless_case(
    case: dict[str, Any],
    db_file: Path,
    *,
    allow_real_api: bool,
) -> tuple[MemoryRuntime, str | None, dict[str, Any]]:
    previous_db = os.environ.get("REACHY_MINI_MEMORY_DB_PATH")
    previous_extractor = os.environ.get("REACHY_MINI_MEMORY_EXTRACTOR")
    os.environ["REACHY_MINI_MEMORY_DB_PATH"] = str(db_file)
    if not allow_real_api:
        os.environ["REACHY_MINI_MEMORY_EXTRACTOR"] = "none"

    try:
        from reachy_mini_conversation_app.config import config
        from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
        from reachy_mini_conversation_app.qwen_omni_realtime import QwenOmniRealtimeHandler
        class MovementManagerStub:
            def set_listening(self, listening: bool) -> None:
                self.listening = listening

            def is_idle(self) -> bool:
                return False

        deps = ToolDependencies(reachy_mini=object(), movement_manager=MovementManagerStub())
        old_tool_mode = config.QWEN_TOOL_MODE
        config.QWEN_TOOL_MODE = "router"
        try:
            setup_started = time.perf_counter()
            handler = QwenOmniRealtimeHandler(deps)
            if not allow_real_api:
                handler.memory_runtime.extractor = ScenarioExtractor(case.get("mock_extraction", {}))
            _seed_runtime(handler.memory_runtime, case.get("seed", {}))
            session_id = handler.memory_runtime.start_session({"eval_case_id": case.get("id"), "eval_mode": "headless"})
            metrics = {"setup_ms": _elapsed_ms(setup_started)}
            turn_started = time.perf_counter()
            for turn in _turns(case):
                role = turn.get("role", "user")
                text = str(turn.get("text") or "")
                event_type = (
                    "response.audio_transcript.done"
                    if role == "assistant"
                    else "conversation.item.input_audio_transcription.completed"
                )
                await handler._handle_message({"type": event_type, "transcript": text})
            if memory_command_writes_enabled():
                _execute_tool_calls(handler.memory_runtime, case.get("tool_calls", []))
            metrics["turns_ms"] = _elapsed_ms(turn_started)
            if case.get("end_session", True):
                end_started = time.perf_counter()
                await handler.memory_runtime.end_session(reason="memory_eval_headless")
                metrics["end_session_ms"] = _elapsed_ms(end_started)
            return handler.memory_runtime, session_id, metrics
        finally:
            config.QWEN_TOOL_MODE = old_tool_mode
    finally:
        if previous_db is not None:
            os.environ["REACHY_MINI_MEMORY_DB_PATH"] = previous_db
        else:
            os.environ.pop("REACHY_MINI_MEMORY_DB_PATH", None)
        if previous_extractor is not None:
            os.environ["REACHY_MINI_MEMORY_EXTRACTOR"] = previous_extractor
        else:
            os.environ.pop("REACHY_MINI_MEMORY_EXTRACTOR", None)


def evaluate_expectations(
    case: dict[str, Any],
    runtime: MemoryRuntime,
    memory_context: str,
    session_id: str | None,
    metrics: dict[str, Any],
) -> list[AssertionResult]:
    """Evaluate functional and performance expectations for a scenario."""
    expectations = case.get("expect", {})
    if not isinstance(expectations, dict):
        expectations = {}
    assertions: list[AssertionResult] = []
    facts = runtime.list_user_profile(include_pending=True)
    notes = [
        runtime._note_to_dict(note)
        for note in runtime.store.list_memory_notes(
            runtime.user.id,
            statuses=("active", "pending_confirmation", "archived"),
            limit=100,
        )
    ]
    tasks = [
        runtime._task_to_dict(task)
        for task in runtime.store.list_care_tasks(
            runtime.user.id,
            statuses=("active", "pending_confirmation", "completed", "disabled", "archived"),
            limit=100,
        )
    ]
    occurrences = []
    for occurrence in runtime.store.list_care_task_occurrences(
        runtime.user.id,
        statuses=("completed",),
        limit=100,
    ):
        task = runtime.store.get_care_task(runtime.user.id, occurrence.task_id)
        occurrences.append(
            {
                "id": occurrence.id,
                "task_id": occurrence.task_id,
                "title": task.title if task is not None else "",
                "occurrence_key": occurrence.occurrence_key,
                "status": occurrence.status,
                "completed_at": occurrence.completed_at,
            }
        )
    turns = runtime.store.get_turns(session_id) if session_id else []
    sessions = runtime.store.get_recent_sessions(runtime.user.id, limit=10)

    assertions.extend(_assert_matches("profile_facts", expectations.get("profile_facts", []), facts))
    assertions.extend(
        _assert_absent_matches("absent_profile_facts", expectations.get("absent_profile_facts", []), facts)
    )
    assertions.extend(_assert_matches("memory_notes", expectations.get("memory_notes", []), notes))
    assertions.extend(_assert_matches("care_tasks", expectations.get("care_tasks", []), tasks))
    assertions.extend(_assert_matches("care_task_occurrences", expectations.get("care_task_occurrences", []), occurrences))
    assertions.extend(_assert_contains("memory_context_contains", expectations, memory_context))
    assertions.extend(_assert_not_contains("memory_context_not_contains", expectations, memory_context))
    assertions.extend(_assert_turns(expectations, turns))
    assertions.extend(_assert_session_summary(expectations, sessions))
    assertions.extend(_assert_budgets(case.get("performance_budget", {}), metrics))
    if not assertions:
        assertions.append(AssertionResult("has_expectations", True, "no explicit expectations were configured"))
    return assertions


def write_reports(results: list[ScenarioResult], report_dir: str | Path) -> tuple[Path, Path]:
    """Write JSON and Markdown reports."""
    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"memory_eval_{stamp}.json"
    md_path = output_dir / f"memory_eval_{stamp}.md"
    payload = {
        "summary": summarize_results(results),
        "results": [_result_to_dict(result) for result in results],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown_report(results), encoding="utf-8")
    return json_path, md_path


def summarize_results(results: list[ScenarioResult]) -> dict[str, Any]:
    """Return aggregate evaluation metrics."""
    total = len(results)
    passed = sum(1 for result in results if result.status == "passed")
    failed = sum(1 for result in results if result.status == "failed")
    skipped = sum(1 for result in results if result.status == "skipped")
    errors = sum(1 for result in results if result.status == "error")
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "errors": errors,
        "pass_rate": round(passed / total, 4) if total else 0.0,
    }


def render_markdown_report(results: list[ScenarioResult]) -> str:
    """Render a compact human-readable report."""
    summary = summarize_results(results)
    lines = [
        "# Memory Evaluation Report",
        "",
        f"- Total: {summary['total']}",
        f"- Passed: {summary['passed']}",
        f"- Failed: {summary['failed']}",
        f"- Skipped: {summary['skipped']}",
        f"- Errors: {summary['errors']}",
        f"- Pass rate: {summary['pass_rate']:.2%}",
        "",
        "| Case | Mode | Status | Total ms | Context chars | Key issues |",
        "|---|---|---|---:|---:|---|",
    ]
    for result in results:
        issues = "; ".join(
            assertion.message for assertion in result.assertions if not assertion.passed
        ) or (result.error or "")
        lines.append(
            f"| {result.case_id} | {result.mode} | {result.status} | "
            f"{result.metrics.get('total_ms', 0):.1f} | {result.metrics.get('context_chars', 0)} | "
            f"{_escape_table(issues[:220])} |"
        )
    lines.append("")
    for result in results:
        lines.append(f"## {result.case_id} {result.title}")
        lines.append("")
        lines.append(f"- Mode: `{result.mode}`")
        lines.append(f"- Status: `{result.status}`")
        if result.error:
            lines.append(f"- Error: `{result.error}`")
        lines.append(f"- Metrics: `{json.dumps(result.metrics, ensure_ascii=False)}`")
        for assertion in result.assertions:
            mark = "PASS" if assertion.passed else "FAIL"
            lines.append(f"- {mark} [{assertion.severity}] {assertion.name}: {assertion.message}")
        lines.append("")
    return "\n".join(lines)


def _extractor_for_mode(case: dict[str, Any], mode: str) -> Any:
    if mode == "qwen-extractor":
        return None
    if case.get("mock_extraction") is not None:
        return ScenarioExtractor(case.get("mock_extraction", {}))
    return NoopMemoryExtractor()


def _seed_runtime(runtime: MemoryRuntime, seed: Any) -> None:
    if not isinstance(seed, dict):
        return
    for fact in seed.get("profile_facts", []):
        if isinstance(fact, dict):
            runtime.remember_user_fact(
                key=str(fact.get("key") or ""),
                value=str(fact.get("value") or ""),
                category=str(fact.get("category") or "preference"),
                confidence=float(fact.get("confidence", 0.95)),
                source=str(fact.get("source") or "seed"),
                confirmed=bool(fact.get("confirmed", True)),
            )
    for task in seed.get("care_tasks", []):
        if isinstance(task, dict):
            runtime.create_care_task(
                title=str(task.get("title") or ""),
                task_type=str(task.get("task_type") or "reminder"),
                due_at=task.get("due_at"),
                recurrence_rule=task.get("recurrence_rule"),
                confirmed=bool(task.get("confirmed", True)),
            )
    for note in seed.get("memory_notes", []):
        runtime.store.add_memory_note(runtime.user.id, note=str(note), source="seed", salience=0.8)


def _execute_tool_calls(runtime: MemoryRuntime, tool_calls: Any) -> None:
    if not isinstance(tool_calls, list):
        return
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        name = str(call.get("name") or "")
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        if name == "remember_user_fact":
            runtime.remember_user_fact(**args)
        elif name == "update_user_fact":
            runtime.update_user_fact(**args)
        elif name == "forget_user_fact":
            runtime.forget_user_fact(str(args.get("query") or ""))
        elif name == "create_care_task":
            runtime.create_care_task(**args)
        elif name == "update_care_task":
            task_id = str(args.pop("task_id"))
            runtime.update_care_task(task_id, **args)
        elif name == "complete_care_task":
            runtime.complete_care_task(task_id=args.get("task_id"), query=args.get("query"))


def _assert_matches(name: str, expected_items: Any, actual_items: list[dict[str, Any]]) -> list[AssertionResult]:
    assertions: list[AssertionResult] = []
    if not isinstance(expected_items, list):
        return assertions
    for index, expected in enumerate(expected_items):
        if not isinstance(expected, dict):
            continue
        match = _find_match(expected, actual_items)
        assertions.append(
            AssertionResult(
                f"{name}[{index}]",
                match is not None,
                "matched expected item" if match is not None else f"no item matched {expected}",
                severity=str(expected.get("severity") or "P1"),
                details={"expected": expected, "match": match},
            )
        )
    return assertions


def _assert_absent_matches(name: str, expected_items: Any, actual_items: list[dict[str, Any]]) -> list[AssertionResult]:
    assertions: list[AssertionResult] = []
    if not isinstance(expected_items, list):
        return assertions
    for index, expected in enumerate(expected_items):
        if not isinstance(expected, dict):
            continue
        match = _find_match(expected, actual_items)
        assertions.append(
            AssertionResult(
                f"{name}[{index}]",
                match is None,
                "no forbidden item found" if match is None else f"forbidden item matched {match}",
                severity=str(expected.get("severity") or "P1"),
                details={"expected_absent": expected, "match": match},
            )
        )
    return assertions


def _assert_contains(name: str, expectations: dict[str, Any], text: str) -> list[AssertionResult]:
    values = expectations.get(name, [])
    if not isinstance(values, list):
        return []
    return [
        AssertionResult(
            f"{name}[{index}]",
            str(value) in text,
            f"context contains {value!r}" if str(value) in text else f"context missing {value!r}",
            severity="P1",
        )
        for index, value in enumerate(values)
    ]


def _assert_not_contains(name: str, expectations: dict[str, Any], text: str) -> list[AssertionResult]:
    values = expectations.get(name, [])
    if not isinstance(values, list):
        return []
    return [
        AssertionResult(
            f"{name}[{index}]",
            str(value) not in text,
            f"context excludes {value!r}" if str(value) not in text else f"context leaked {value!r}",
            severity="P1",
        )
        for index, value in enumerate(values)
    ]


def _assert_turns(expectations: dict[str, Any], turns: list[Turn]) -> list[AssertionResult]:
    assertions: list[AssertionResult] = []
    if "turn_count" in expectations:
        expected_count = int(expectations["turn_count"])
        assertions.append(
            AssertionResult(
                "turn_count",
                len(turns) == expected_count,
                f"turn count {len(turns)} == {expected_count}",
                severity="P2",
            )
        )
    if "turn_roles" in expectations and isinstance(expectations["turn_roles"], list):
        roles = [turn.role for turn in turns]
        expected_roles = [str(role) for role in expectations["turn_roles"]]
        assertions.append(
            AssertionResult(
                "turn_roles",
                roles == expected_roles,
                f"turn roles {roles} == {expected_roles}",
                severity="P2",
            )
        )
    return assertions


def _assert_session_summary(expectations: dict[str, Any], sessions: list[Any]) -> list[AssertionResult]:
    values = expectations.get("session_summary_contains", [])
    if not isinstance(values, list):
        return []
    summary_text = "\n".join(session.summary or "" for session in sessions)
    return [
        AssertionResult(
            f"session_summary_contains[{index}]",
            str(value) in summary_text,
            f"summary contains {value!r}" if str(value) in summary_text else f"summary missing {value!r}",
            severity="P2",
        )
        for index, value in enumerate(values)
    ]


def _assert_budgets(budgets: Any, metrics: dict[str, Any]) -> list[AssertionResult]:
    if not isinstance(budgets, dict):
        return []
    assertions: list[AssertionResult] = []
    budget_map = {
        "total_ms_max": "total_ms",
        "setup_ms_max": "setup_ms",
        "turns_ms_max": "turns_ms",
        "end_session_ms_max": "end_session_ms",
        "context_build_ms_max": "context_build_ms",
        "context_chars_max": "context_chars",
    }
    for budget_key, metric_key in budget_map.items():
        if budget_key not in budgets or metric_key not in metrics:
            continue
        actual = float(metrics[metric_key])
        maximum = float(budgets[budget_key])
        assertions.append(
            AssertionResult(
                budget_key,
                actual <= maximum,
                f"{metric_key} {actual:.1f} <= {maximum:.1f}",
                severity="P2",
                details={"metric": metric_key, "actual": actual, "maximum": maximum},
            )
        )
    return assertions


def _find_match(expected: dict[str, Any], actual_items: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in actual_items:
        if _item_matches(expected, item):
            return item
    return None


def _item_matches(expected: dict[str, Any], item: dict[str, Any]) -> bool:
    for key, expected_value in expected.items():
        if key in {"severity"}:
            continue
        if key.endswith("_contains"):
            field = key[: -len("_contains")]
            if str(expected_value) not in str(item.get(field, "")):
                return False
        elif key.endswith("_not_contains"):
            field = key[: -len("_not_contains")]
            if str(expected_value) in str(item.get(field, "")):
                return False
        elif item.get(key) != expected_value:
            return False
    return True


def _turns(case: dict[str, Any]) -> list[dict[str, Any]]:
    turns = case.get("turns", [])
    return [turn for turn in turns if isinstance(turn, dict)] if isinstance(turns, list) else []


def _resolve_mode(case: dict[str, Any], mode: EvalMode) -> str:
    if mode == "offline":
        runner = str(case.get("runner") or "router")
        return "router" if runner in {"qwen-extractor", "realtime-headless"} else runner
    if mode == "qwen-extractor":
        return "qwen-extractor"
    if mode == "realtime-headless":
        return "realtime-headless"
    return str(case.get("runner") or "router")


def _case_db_path(db_path: str | Path | None, case: dict[str, Any], index: int, keep_db: bool) -> Path:
    if db_path is not None:
        base = Path(db_path)
        if len(base.suffix) > 0:
            if index == 0:
                return base
            safe_case_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(case.get("id", "case")))
            return base.with_name(f"{base.stem}_{index:03d}_{safe_case_id}{base.suffix}")
        base.mkdir(parents=True, exist_ok=True)
        return base / f"{index:03d}_{case.get('id', 'case')}.sqlite3"
    if keep_db:
        output_dir = Path(tempfile.gettempdir()) / "reachy_memory_eval"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"{int(time.time())}_{index:03d}_{case.get('id', 'case')}.sqlite3"
    return _temporary_db_path(str(case.get("id") or "case"))


def _temporary_db_path(case_id: str) -> Path:
    safe_case_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in case_id)
    return Path(tempfile.gettempdir()) / f"reachy_memory_eval_{safe_case_id}_{uuid.uuid4().hex}.sqlite3"


def _remove_sqlite_files(path: str | Path) -> None:
    base = Path(path)
    for candidate in [base, Path(f"{base}-wal"), Path(f"{base}-shm")]:
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _result_to_dict(result: ScenarioResult) -> dict[str, Any]:
    return {
        "case_id": result.case_id,
        "title": result.title,
        "mode": result.mode,
        "status": result.status,
        "assertions": [
            {
                "name": assertion.name,
                "passed": assertion.passed,
                "message": assertion.message,
                "severity": assertion.severity,
                "details": assertion.details,
            }
            for assertion in result.assertions
        ],
        "metrics": result.metrics,
        "db_path": result.db_path,
        "memory_context": result.memory_context,
        "error": result.error,
    }


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
