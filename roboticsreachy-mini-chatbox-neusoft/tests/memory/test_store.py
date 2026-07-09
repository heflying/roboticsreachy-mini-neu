from reachy_mini_conversation_app.memory.store import SQLiteMemoryStore


def test_store_persists_turns_and_profile_facts(tmp_path):
    """Store transcripts and archive matching profile facts."""
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    user = store.get_or_create_user("elder-1", display_name="张老师")
    session = store.start_session(user.id)

    turn = store.append_turn(session.id, user.id, "user", "记住我喜欢听越剧")
    assert turn.content == "记住我喜欢听越剧"
    assert store.get_turns(session.id)[0].id == turn.id

    fact = store.upsert_profile_fact(
        user.id,
        key="preference.likes.yueju",
        value="听越剧",
        category="preference",
        confidence=0.95,
        status="active",
        source="tool",
    )
    assert fact.status == "active"
    assert store.search_profile_facts(user.id, "越剧")[0].value == "听越剧"

    archived = store.archive_profile_fact(user.id, "越剧")
    assert [item.id for item in archived] == [fact.id]
    assert store.search_profile_facts(user.id, "越剧") == []


def test_store_care_task_lifecycle(tmp_path):
    """Create and complete a care task."""
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    user = store.get_or_create_user("elder-1")

    task = store.create_care_task(
        user.id,
        title="饭后喝水",
        task_type="hydration",
        due_at="2026-05-06T12:30:00+08:00",
    )
    assert store.list_care_tasks(user.id)[0].title == "饭后喝水"

    completed = store.complete_care_task(user.id, task.id)
    assert completed is not None
    assert completed.status == "completed"
    assert completed.completed_at is not None


def test_store_recurring_care_task_completion_creates_occurrence(tmp_path):
    """Completing a recurring task records today's occurrence and keeps the task active."""
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    user = store.get_or_create_user("elder-1")

    task = store.create_care_task(
        user.id,
        title="晚饭后喝水",
        task_type="hydration",
        recurrence_rule="FREQ=DAILY;BYHOUR=18;BYMINUTE=30",
    )

    completed = store.complete_care_task(user.id, task.id)
    occurrences = store.list_care_task_occurrences(user.id, task_id=task.id)

    assert completed is not None
    assert completed.status == "active"
    assert len(occurrences) == 1
    assert occurrences[0].status == "completed"
    assert occurrences[0].completed_at is not None
