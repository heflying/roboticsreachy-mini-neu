import pytest

from reachy_mini_conversation_app.memory.store import SQLiteMemoryStore
from reachy_mini_conversation_app.memory.runtime import MemoryRuntime
from reachy_mini_conversation_app.memory.command_router import MemoryCommandRouter


pytestmark = pytest.mark.asyncio


async def test_command_router_handles_preferred_name(tmp_path):
    """Explicit naming commands save preferred_name."""
    runtime = MemoryRuntime(SQLiteMemoryStore(tmp_path / "memory.db"))
    router = MemoryCommandRouter(runtime)

    result = await router.handle("以后叫我张老师")

    assert result is not None
    assert result["type"] == "memory_saved"
    profile = runtime.list_user_profile()
    assert profile[0]["key"] == "preferred_name"
    assert profile[0]["value"] == "张老师"


async def test_command_router_marks_sensitive_memory_pending(tmp_path):
    """Explicit health notes are routed but still require confirmation."""
    runtime = MemoryRuntime(SQLiteMemoryStore(tmp_path / "memory.db"))
    router = MemoryCommandRouter(runtime)

    await router.handle("记住我每天饭后量血压")

    profile = runtime.list_user_profile(include_pending=True)
    assert profile[0]["status"] == "pending_confirmation"
    assert profile[0]["category"] == "health"


async def test_command_router_forgets_matching_memory(tmp_path):
    """Explicit forget commands archive matching facts."""
    runtime = MemoryRuntime(SQLiteMemoryStore(tmp_path / "memory.db"))
    router = MemoryCommandRouter(runtime)
    await router.handle("以后叫我张老师")

    result = await router.handle("忘掉张老师")

    assert result is not None
    assert result["archived_facts"] == 1
    assert runtime.list_user_profile() == []


async def test_command_router_creates_daily_hydration_reminder(tmp_path):
    """Explicit reminder commands create active non-sensitive care tasks."""
    runtime = MemoryRuntime(SQLiteMemoryStore(tmp_path / "memory.db"))
    router = MemoryCommandRouter(runtime)

    result = await router.handle("每天晚饭后提醒我喝水。")

    assert result is not None
    assert result["type"] == "care_task_created"
    tasks = runtime.list_today_care_tasks()
    assert tasks[0]["title"] == "晚饭后喝水"
    assert tasks[0]["task_type"] == "hydration"
    assert tasks[0]["status"] == "active"


async def test_command_router_keeps_unconfirmed_medication_reminder_pending(tmp_path):
    """Medication reminders stay pending unless the user confirms them."""
    runtime = MemoryRuntime(SQLiteMemoryStore(tmp_path / "memory.db"))
    router = MemoryCommandRouter(runtime)

    await router.handle("每天早上提醒我吃降压药。")

    tasks = runtime.store.list_care_tasks(
        runtime.user.id,
        statuses=("active", "pending_confirmation"),
    )
    assert tasks[0].title == "早上吃降压药"
    assert tasks[0].task_type == "medication"
    assert tasks[0].status == "pending_confirmation"


async def test_command_router_activates_confirmed_medication_reminder(tmp_path):
    """Confirmed medication reminder commands can become active."""
    runtime = MemoryRuntime(SQLiteMemoryStore(tmp_path / "memory.db"))
    router = MemoryCommandRouter(runtime)

    await router.handle("我确认每天早饭后提醒我服药。")

    tasks = runtime.list_today_care_tasks()
    assert tasks[0]["title"] == "早饭后服药"
    assert tasks[0]["task_type"] == "medication"
    assert tasks[0]["status"] == "active"
