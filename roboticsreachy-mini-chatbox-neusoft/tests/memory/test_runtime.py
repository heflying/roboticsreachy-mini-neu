import pytest

from reachy_mini_conversation_app.memory.store import SQLiteMemoryStore
from reachy_mini_conversation_app.memory.models import (
    Turn,
    MemoryAction,
    SessionSummary,
    MemoryCandidate,
    ExtractionResult,
    CareTaskCandidate,
)
from reachy_mini_conversation_app.memory.runtime import MemoryRuntime


class FakeExtractor:
    """Mock extractor that returns deterministic memory candidates."""

    async def extract(self, *, session_id: str, turns: list[Turn], memory_context: str) -> ExtractionResult:
        """Return a fixed extraction payload without model calls."""
        return ExtractionResult(
            summary=SessionSummary(
                session_id=session_id,
                summary="用户上次聊到晚饭后想听越剧，并希望下次继续这个话题。",
                topics=["越剧"],
                follow_ups=["下次问问是否还想听越剧"],
            ),
            profile_candidates=[
                MemoryCandidate(
                    key="preference.likes.yueju",
                    value="晚饭后听越剧",
                    category="preference",
                    confidence=0.9,
                    source="extractor",
                    evidence="用户说晚饭后喜欢听越剧",
                ),
                MemoryCandidate(
                    key="medication.current",
                    value="阿司匹林",
                    category="medication",
                    confidence=0.9,
                    source="extractor",
                    evidence="用户提到阿司匹林",
                ),
            ],
            memory_notes=["下次可以自然回访越剧话题"],
            care_task_candidates=[
                CareTaskCandidate(
                    title="晚饭后喝水",
                    task_type="hydration",
                    confidence=0.9,
                    source="extractor",
                )
            ],
        )


class ActionExtractor:
    """Mock extractor that returns explicit session-end memory actions."""

    def __init__(self, actions: list[MemoryAction]):
        """Create an extractor that returns ``actions`` unchanged."""
        self.actions = actions

    async def extract(self, *, session_id: str, turns: list[Turn], memory_context: str) -> ExtractionResult:
        """Return explicit memory actions without calling a model."""
        return ExtractionResult(memory_actions=self.actions)


class SlowActionExtractor(ActionExtractor):
    """Extractor that yields once so background scheduling is observable."""

    async def extract(self, *, session_id: str, turns: list[Turn], memory_context: str) -> ExtractionResult:
        """Yield to the event loop before returning actions."""
        import asyncio

        await asyncio.sleep(0.01)
        return await super().extract(session_id=session_id, turns=turns, memory_context=memory_context)


@pytest.mark.asyncio
async def test_runtime_ends_session_with_mock_extraction(tmp_path):
    """End-of-session extraction stores active, pending, note, and task records."""
    runtime = MemoryRuntime(SQLiteMemoryStore(tmp_path / "memory.db"), extractor=FakeExtractor())
    runtime.start_session()
    runtime.record_user_transcript("我晚饭后喜欢听越剧，也提到了阿司匹林。")
    runtime.record_assistant_transcript("我记下了，之后可以继续聊越剧。")

    await runtime.end_session(reason="test")

    facts = runtime.list_user_profile(include_pending=True)
    assert any(fact["key"] == "preference.likes.yueju" and fact["status"] == "active" for fact in facts)
    assert any(fact["key"] == "medication.current" and fact["status"] == "pending_confirmation" for fact in facts)

    context = runtime.build_memory_context()
    assert "晚饭后听越剧" in context
    assert "阿司匹林" not in context
    assert "下次可以自然回访越剧话题" in context
    assert "晚饭后喝水" in context


@pytest.mark.asyncio
async def test_runtime_stores_sensitive_notes_as_pending(tmp_path):
    """Sensitive middle-term notes are persisted but excluded from realtime context."""
    runtime = MemoryRuntime(
        SQLiteMemoryStore(tmp_path / "memory.db"),
        extractor=ActionExtractor([]),
    )
    runtime.start_session()
    runtime.store.add_memory_note(
        runtime.user.id,
        note="用户今早服阿司匹林一片，但未确认是否长期用药。",
        source="extractor",
        status=runtime.safety_filter.evaluate_memory_note("用户今早服阿司匹林一片").status,
    )
    runtime.store.add_memory_note(
        runtime.user.id,
        note="用户提到胃不舒服，但强调先不要当成健康结论。",
        source="extractor",
        status=runtime.safety_filter.evaluate_memory_note("用户提到胃不舒服").status,
    )

    notes = runtime.store.list_memory_notes(runtime.user.id, statuses=("pending_confirmation",))

    assert len(notes) == 2
    assert "阿司匹林" not in runtime.build_memory_context()
    assert "胃不舒服" not in runtime.build_memory_context()


def test_runtime_explicit_sensitive_fact_is_pending_without_confirmation(tmp_path):
    """Explicit but unconfirmed sensitive facts remain pending."""
    runtime = MemoryRuntime(SQLiteMemoryStore(tmp_path / "memory.db"))

    fact = runtime.remember_user_fact(
        key="health.blood_pressure",
        value="午饭后容易头晕",
        category="health",
        source="tool",
        confirmed=False,
    )

    assert fact is not None
    assert fact.status == "pending_confirmation"


@pytest.mark.asyncio
async def test_runtime_applies_create_care_task_action_at_session_end(tmp_path):
    """Session-end memory_actions can create active care tasks without router/native tools."""
    runtime = MemoryRuntime(
        SQLiteMemoryStore(tmp_path / "memory.db"),
        extractor=ActionExtractor(
            [
                MemoryAction(
                    action="create_care_task",
                    title="晚饭后喝水",
                    task_type="hydration",
                    recurrence_rule="daily",
                    confidence=0.95,
                    requires_confirmation=False,
                )
            ]
        ),
    )
    runtime.start_session()
    runtime.record_user_transcript("每天晚饭后提醒我喝水。")

    await runtime.end_session(reason="test")

    tasks = runtime.list_today_care_tasks()
    assert len(tasks) == 1
    assert tasks[0]["title"] == "晚饭后喝水"
    assert tasks[0]["status"] == "active"
    assert "晚饭后喝水" in runtime.build_memory_context()


@pytest.mark.asyncio
async def test_runtime_applies_complete_care_task_action_at_session_end(tmp_path):
    """Session-end memory_actions can complete existing care tasks."""
    runtime = MemoryRuntime(
        SQLiteMemoryStore(tmp_path / "memory.db"),
        extractor=ActionExtractor([MemoryAction(action="complete_care_task", query="喝水", confidence=0.95)]),
    )
    task = runtime.create_care_task(title="晚饭后喝水", task_type="hydration", confirmed=True)
    assert task is not None
    runtime.start_session()
    runtime.record_user_transcript("我已经喝水了。")

    await runtime.end_session(reason="test")

    tasks = runtime.list_today_care_tasks(include_completed=True)
    assert any(item["title"] == "晚饭后喝水" and item["status"] == "completed" for item in tasks)
    assert "晚饭后喝水" not in runtime.build_memory_context()


@pytest.mark.asyncio
async def test_runtime_completes_recurring_task_as_occurrence(tmp_path):
    """Session-end completion of a recurring task should not disable future reminders."""
    runtime = MemoryRuntime(
        SQLiteMemoryStore(tmp_path / "memory.db"),
        extractor=ActionExtractor([MemoryAction(action="complete_care_task", query="喝水", confidence=0.95)]),
    )
    task = runtime.create_care_task(
        title="晚饭后喝水",
        task_type="hydration",
        recurrence_rule="FREQ=DAILY;BYHOUR=18;BYMINUTE=30",
        confirmed=True,
    )
    assert task is not None
    runtime.start_session()
    runtime.record_user_transcript("我已经喝水了。")

    await runtime.end_session(reason="test")

    tasks = runtime.list_today_care_tasks(include_completed=True)
    occurrences = runtime.store.list_care_task_occurrences(runtime.user.id, task_id=task.id)
    assert any(item["title"] == "晚饭后喝水" and item["status"] == "active" for item in tasks)
    assert len(occurrences) == 1
    assert occurrences[0].status == "completed"


@pytest.mark.asyncio
async def test_runtime_applies_disable_care_task_action_at_session_end(tmp_path):
    """Session-end memory_actions can disable existing active or pending reminders."""
    runtime = MemoryRuntime(
        SQLiteMemoryStore(tmp_path / "memory.db"),
        extractor=ActionExtractor([MemoryAction(action="disable_care_task", query="睡前拉伸", confidence=0.95)]),
    )
    task = runtime.create_care_task(title="睡前拉伸", task_type="exercise", confirmed=True)
    assert task is not None
    runtime.start_session()
    runtime.record_user_transcript("以后不用提醒我睡前拉伸。")

    await runtime.end_session(reason="test")

    disabled = runtime.store.list_care_tasks(runtime.user.id, statuses=("disabled",), query="睡前拉伸")
    assert len(disabled) == 1
    assert "睡前拉伸" not in runtime.build_memory_context()


@pytest.mark.asyncio
async def test_runtime_disable_action_creates_tombstone_when_task_missing(tmp_path):
    """A cancellation should persist even when the original task is absent from the DB."""
    runtime = MemoryRuntime(
        SQLiteMemoryStore(tmp_path / "memory.db"),
        extractor=ActionExtractor([MemoryAction(action="disable_care_task", query="睡前拉伸", confidence=0.95)]),
    )
    runtime.start_session()
    runtime.record_user_transcript("睡前拉伸那个提醒不用了。")

    await runtime.end_session(reason="test")

    disabled = runtime.store.list_care_tasks(runtime.user.id, statuses=("disabled",), query="睡前拉伸")
    assert len(disabled) == 1
    assert disabled[0].metadata["tombstone"] is True


@pytest.mark.asyncio
async def test_runtime_applies_forget_action_to_profile_and_notes(tmp_path):
    """Session-end memory_actions can archive profile facts and related middle-term notes."""
    runtime = MemoryRuntime(
        SQLiteMemoryStore(tmp_path / "memory.db"),
        extractor=ActionExtractor([MemoryAction(action="forget_user_fact", query="越剧", confidence=0.95)]),
    )
    runtime.remember_user_fact(
        key="preference.likes",
        value="晚饭后听越剧",
        category="preference",
        confirmed=True,
    )
    runtime.store.add_memory_note(runtime.user.id, note="下次可以自然回访越剧话题", source="test")
    runtime.start_session()
    runtime.record_user_transcript("忘掉越剧。")

    await runtime.end_session(reason="test")

    assert not runtime.recall_user_memory("越剧")["facts"]
    assert not runtime.recall_user_memory("越剧")["notes"]


@pytest.mark.asyncio
async def test_runtime_background_end_session_does_not_block_current_turn(tmp_path):
    """Realtime close can schedule extractor work and return before writes are applied."""
    runtime = MemoryRuntime(
        SQLiteMemoryStore(tmp_path / "memory.db"),
        extractor=SlowActionExtractor(
            [
                MemoryAction(
                    action="remember_user_fact",
                    key="preferred_name",
                    value="王阿姨",
                    category="identity",
                    confidence=0.95,
                    requires_confirmation=False,
                )
            ]
        ),
    )
    runtime.start_session()
    runtime.record_user_transcript("以后叫我王阿姨。")

    task = runtime.end_session_background(reason="test")

    assert task is not None
    assert runtime.current_session_id is None
    assert runtime.list_user_profile() == []
    await runtime.wait_for_pending_extractions(timeout_s=1)
    assert runtime.list_user_profile()[0]["value"] == "王阿姨"
