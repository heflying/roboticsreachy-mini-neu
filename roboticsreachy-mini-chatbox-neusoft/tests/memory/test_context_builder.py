from reachy_mini_conversation_app.memory.store import SQLiteMemoryStore
from reachy_mini_conversation_app.memory.runtime import MemoryRuntime


def test_context_builder_redacts_archived_fact_values_from_summaries_and_notes(tmp_path):
    """Archived corrections should not leak old values through middle-term context."""
    runtime = MemoryRuntime(SQLiteMemoryStore(tmp_path / "memory.db"))
    runtime.remember_user_fact(
        key="preferred_name",
        value="张老师",
        category="identity",
        confirmed=True,
    )
    runtime.remember_user_fact(
        key="preferred_name",
        value="王阿姨",
        category="identity",
        confirmed=True,
    )
    session = runtime.store.start_session(runtime.user.id)
    runtime.store.end_session(
        session.id,
        reason="test",
        summary="用户将称呼偏好从张老师改为王阿姨。",
    )
    runtime.store.add_memory_note(
        runtime.user.id,
        note="用户明确表示更喜欢王阿姨，不要再叫张老师。",
        session_id=session.id,
    )

    context = runtime.build_memory_context()

    assert "王阿姨" in context
    assert "张老师" not in context


def test_context_builder_redacts_pending_sensitive_fact_values(tmp_path):
    """Pending health details should not leak through summaries or notes."""
    runtime = MemoryRuntime(SQLiteMemoryStore(tmp_path / "memory.db"))
    runtime.remember_user_fact(
        key="health.dizziness_after_lunch",
        value="最近午饭后有点头晕，原因未确认",
        category="health",
        source="extractor",
    )
    session = runtime.store.start_session(runtime.user.id)
    runtime.store.end_session(
        session.id,
        reason="test",
        summary="用户最近提到午饭后有点头晕，但原因未确认。",
    )
    runtime.store.add_memory_note(
        runtime.user.id,
        note="下次可温和回访头晕是否好转。",
        session_id=session.id,
    )

    context = runtime.build_memory_context()

    assert "头晕" not in context
    assert "午饭后有点头晕" not in context


def test_context_builder_drops_pending_address_notes(tmp_path):
    """Pending address summaries should not teach realtime to treat the address as confirmed."""
    runtime = MemoryRuntime(SQLiteMemoryStore(tmp_path / "memory.db"))
    runtime.remember_user_fact(
        key="address.home",
        value="幸福路18号",
        category="address",
        source="extractor",
        evidence="我住在幸福路18号，门牌别弄错",
    )
    session = runtime.store.start_session(runtime.user.id)
    runtime.store.end_session(
        session.id,
        reason="test",
        summary="用户提供了家庭住址信息，强调门牌号准确性。",
    )
    runtime.store.add_memory_note(
        runtime.user.id,
        note="用户刚提供住址，待确认是否存入长期记忆。",
        session_id=session.id,
    )

    context = runtime.build_memory_context()

    assert "幸福路18号" not in context
    assert "家庭住址" not in context
    assert "门牌" not in context
    assert "待确认" not in context


def test_context_builder_groups_family_facts_for_recall(tmp_path):
    """Family facts are injected as one compact overview so realtime can answer fully."""
    runtime = MemoryRuntime(SQLiteMemoryStore(tmp_path / "memory.db"))
    for key, value in {
        "family.daughter.name": "李敏",
        "family.son.name": "陈强",
        "family.grandchild.name": "小宝",
        "family.visit_pattern": "女儿每周六下午来访",
    }.items():
        runtime.remember_user_fact(key=key, value=value, category="family", confirmed=True)

    context = runtime.build_memory_context()

    assert "family.overview" in context
    assert "女儿李敏" in context
    assert "儿子陈强" in context
    assert "外孙小宝" in context


def test_context_builder_includes_completed_recurring_task_occurrences(tmp_path):
    """Completed occurrences should be visible without disabling recurring reminders."""
    runtime = MemoryRuntime(SQLiteMemoryStore(tmp_path / "memory.db"))
    task = runtime.create_care_task(
        title="晚饭后喝水",
        task_type="hydration",
        recurrence_rule="daily after dinner",
        confirmed=True,
    )
    assert task is not None
    runtime.complete_care_task(task_id=task.id)

    context = runtime.build_memory_context()

    assert "今日或仍有效的照护提醒" in context
    assert "今日已完成的重复提醒实例" in context
    assert "晚饭后喝水" in context
    assert "后续提醒仍有效" in context
