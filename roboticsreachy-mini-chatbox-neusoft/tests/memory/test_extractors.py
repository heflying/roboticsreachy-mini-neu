from reachy_mini_conversation_app.memory.extractors import (
    QwenMemoryExtractor,
    _extraction_from_json,
    create_default_extractor,
)


def test_create_default_extractor_uses_configured_timeout(monkeypatch):
    """Long-session extraction can raise the model timeout without code changes."""
    monkeypatch.setenv("REACHY_MINI_MEMORY_EXTRACTOR", "qwen")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "fake-key")
    monkeypatch.setenv("QWEN_MEMORY_TIMEOUT_S", "90")

    extractor = create_default_extractor()

    assert isinstance(extractor, QwenMemoryExtractor)
    assert extractor.timeout_s == 90


def test_extractor_normalizes_common_key_aliases_and_chinese_values():
    """Extractor output aliases are normalized before safety filtering."""
    result = _extraction_from_json(
        "sess_1",
        {
            "profile_candidates": [
                {
                    "key": "communication.speech_rate",
                    "value": "slower speech rate",
                    "category": "communication",
                    "confidence": 0.9,
                    "requires_confirmation": True,
                },
                {
                    "key": "medication.aspirin",
                    "value": "Takes one aspirin tablet in the morning",
                    "category": "medication",
                    "confidence": 0.9,
                    "requires_confirmation": False,
                },
                {
                    "key": "family.daughter.name",
                    "value": "李敏",
                    "category": "family",
                    "confidence": 0.9,
                    "requires_confirmation": True,
                },
            ],
            "memory_notes": [
                "User flagged a potential scam call.",
            ],
        },
    )

    candidates = {candidate.key: candidate for candidate in result.profile_candidates}
    assert candidates["communication.speaking_pace"].value == "说慢一点"
    assert candidates["communication.speaking_pace"].requires_confirmation is False
    assert candidates["medication.current"].value == "阿司匹林"
    assert candidates["medication.current"].requires_confirmation is True
    assert candidates["family.daughter.name"].requires_confirmation is False
    assert "诈骗" in result.memory_notes[0]


def test_extractor_filters_transient_preferences_and_delete_commands():
    """Transient likes and deletion commands should not become long-term memory."""
    result = _extraction_from_json(
        "sess_2",
        {
            "summary": {"summary": "用户主动要求忘记越剧相关内容"},
            "profile_candidates": [
                {
                    "key": "preference.likes",
                    "value": "热闹的歌",
                    "category": "preference",
                    "confidence": 0.7,
                    "evidence": "今天就想听点热闹的歌",
                },
                {
                    "key": "preference.dislikes",
                    "value": "越剧",
                    "category": "preference",
                    "confidence": 0.95,
                    "evidence": "用户说忘掉越剧",
                },
                {
                    "key": "preference.dislikes",
                    "value": "过于热闹的音乐（长期）",
                    "category": "preference",
                    "confidence": 0.9,
                    "evidence": "平时我还是喜欢安静",
                },
            ],
            "memory_notes": [
                "用户主动提出清除越剧相关记忆",
                "需后续确认是否已采取防诈措施",
            ],
        },
    )

    candidates = {candidate.key: candidate for candidate in result.profile_candidates}
    assert set(candidates) == {"preference.audio_style"}
    assert candidates["preference.audio_style"].value == "平时喜欢安静"
    assert result.summary is None
    assert result.memory_notes == ["需后续确认是否已采取防诈骗措施"]

    delete_only_result = _extraction_from_json(
        "sess_3",
        {
            "memory_notes": [
                "用户近期表示不希望与越剧产生关联，已记录为偏好排除项",
            ],
        },
        transcript="user: 忘掉越剧。",
    )
    assert delete_only_result.memory_notes == []
    assert len(delete_only_result.memory_actions) == 1
    assert delete_only_result.memory_actions[0].action == "forget_user_fact"
    assert delete_only_result.memory_actions[0].query == "越剧"

    disable_reminder_result = _extraction_from_json(
        "sess_4",
        {
            "summary": {"summary": "用户明确表示不希望再被提醒睡前拉伸"},
            "profile_candidates": [
                {
                    "key": "care_preference.reminder_style",
                    "value": "不希望接收睡前拉伸提醒",
                    "category": "care_preference",
                    "confidence": 0.9,
                    "evidence": "以后不用提醒我睡前拉伸",
                }
            ],
            "memory_notes": [
                "用户本次主动取消睡前拉伸提醒",
            ],
            "care_task_candidates": [
                {
                    "title": "停止睡前拉伸提醒",
                    "task_type": "reminder",
                    "confidence": 0.9,
                }
            ],
        },
        transcript="user: 以后不用提醒我睡前拉伸。",
    )
    assert disable_reminder_result.summary is None
    assert disable_reminder_result.profile_candidates == []
    assert disable_reminder_result.memory_notes == []
    assert disable_reminder_result.care_task_candidates == []
    assert len(disable_reminder_result.memory_actions) == 1
    assert disable_reminder_result.memory_actions[0].action == "disable_care_task"
    assert disable_reminder_result.memory_actions[0].query == "睡前拉伸"


def test_extractor_does_not_treat_mixed_long_transcript_as_action_only():
    """Cleanup commands inside a long session should not suppress unrelated memory candidates."""
    result = _extraction_from_json(
        "sess_long",
        {
            "profile_candidates": [
                {
                    "key": "preferred_name",
                    "value": "林阿姨",
                    "category": "identity",
                    "confidence": 0.95,
                    "requires_confirmation": False,
                }
            ],
            "memory_notes": ["用户午饭后会午休，避免主动打扰。"],
        },
        transcript=(
            "user: 以后叫我林阿姨。\n"
            "assistant: 好的。\n"
            "user: 睡前拉伸那个提醒不用了。\n"
            "assistant: 好的。"
        ),
    )

    assert result.profile_candidates[0].key == "preferred_name"
    assert result.memory_notes == ["用户午饭后会午休，避免主动打扰。"]


def test_extractor_scopes_sensitive_task_detection_to_task_evidence():
    """Sensitive health text elsewhere in a long transcript should not make hydration pending."""
    result = _extraction_from_json(
        "sess_long_task",
        {
            "care_task_candidates": [
                {
                    "title": "晚饭后喝水",
                    "task_type": "hydration",
                    "confidence": 0.9,
                    "evidence": "用户说每天晚饭后提醒我喝水",
                    "requires_confirmation": True,
                }
            ]
        },
        transcript=(
            "user: 每天晚饭后提醒我喝水。\n"
            "user: 医生说我最近血压有点高，但还没复查。"
        ),
    )

    assert result.care_task_candidates[0].title == "晚饭后喝水"
    assert result.care_task_candidates[0].requires_confirmation is False


def test_extractor_merges_model_actions_with_inferred_transcript_actions():
    """A partial model action list should not suppress obvious transcript CRUD commands."""
    result = _extraction_from_json(
        "sess_merge_actions",
        {
            "memory_actions": [
                {
                    "action": "remember_user_fact",
                    "key": "preferred_name",
                    "value": "林阿姨",
                    "category": "identity",
                    "confidence": 0.95,
                    "requires_confirmation": False,
                }
            ]
        },
        transcript=(
            "user: 以后叫我林阿姨。\n"
            "user: 每天晚饭后提醒我喝水，周五上午提醒我去社区医院复诊。\n"
            "user: 睡前拉伸那个提醒不用了。"
        ),
    )

    actions = {(action.action, action.title or action.query or action.key) for action in result.memory_actions}
    assert ("remember_user_fact", "preferred_name") in actions
    assert ("create_care_task", "晚饭后喝水") in actions
    assert ("create_care_task", "周五上午社区医院复诊") in actions
    assert ("disable_care_task", "睡前拉伸") in actions


def test_extractor_infers_preference_update_and_old_value_forget():
    """Stable preference corrections become an update plus an archive query for the old value."""
    result = _extraction_from_json(
        "sess_tea_update",
        {},
        transcript="user: 饮茶口味改成红茶，绿茶先忘掉。",
    )

    actions = {(action.action, action.key or action.query, action.value) for action in result.memory_actions}
    assert ("update_user_fact", "preference.likes", "红茶") in actions
    assert ("forget_user_fact", "绿茶", None) in actions


def test_extractor_infers_family_visit_pattern_from_long_user_line():
    """Family visit cadence should not be lost when the model only extracts names."""
    result = _extraction_from_json(
        "sess_family_visit",
        {},
        transcript="user: 我女儿叫李敏，周六下午常来看我；儿子叫陈强，外孙叫小宝。",
    )

    actions = {(action.action, action.key, action.value) for action in result.memory_actions}
    assert ("update_user_fact", "family.visit_pattern", "女儿每周六下午来访") in actions


def test_extractor_rejects_unanchored_abstract_safety_reminders():
    """Abstract safety requests without a schedule should not become active care tasks."""
    result = _extraction_from_json(
        "sess_safety",
        {
            "care_task_candidates": [
                {
                    "title": "你以后听到这种事要提醒我小心",
                    "task_type": "reminder",
                    "due_at": "你以后听到这种事要",
                    "confidence": 0.95,
                    "requires_confirmation": False,
                }
            ]
        },
        transcript="user: 他说中奖要交保证金，你以后听到这种事要提醒我小心。",
    )

    assert result.care_task_candidates == []
    assert result.memory_actions == []


def test_extractor_drops_cancelled_reminder_notes():
    """Cancellation notes should be represented as disable actions, not active middle-term notes."""
    result = _extraction_from_json(
        "sess_cancel_note",
        {
            "memory_notes": [
                "用户膝盖不适，已取消睡前拉伸提醒",
            ],
        },
        transcript="user: 睡前拉伸那个提醒不用了。",
    )

    assert result.memory_notes == []
    assert result.memory_actions[0].action == "disable_care_task"
    assert result.memory_actions[0].query == "睡前拉伸"


def test_extractor_infers_explicit_care_task_actions_at_session_end():
    """Explicit task CRUD commands become memory_actions even if the model omits them."""
    create_result = _extraction_from_json(
        "sess_5",
        {},
        transcript="user: 每天晚饭后提醒我喝水。",
    )
    assert len(create_result.memory_actions) == 1
    create_action = create_result.memory_actions[0]
    assert create_action.action == "create_care_task"
    assert create_action.title == "晚饭后喝水"
    assert create_action.task_type == "hydration"
    assert create_action.recurrence_rule == "daily"
    assert create_action.requires_confirmation is False

    medication_result = _extraction_from_json(
        "sess_6",
        {},
        transcript="user: 每天早上提醒我吃降压药。",
    )
    medication_action = medication_result.memory_actions[0]
    assert medication_action.action == "create_care_task"
    assert medication_action.title == "早上吃降压药"
    assert medication_action.task_type == "medication"
    assert medication_action.requires_confirmation is True

    complete_result = _extraction_from_json(
        "sess_7",
        {"summary": {"summary": "用户说已经喝水了"}},
        transcript="user: 我已经喝水了。",
    )
    assert complete_result.summary is None
    assert len(complete_result.memory_actions) == 1
    assert complete_result.memory_actions[0].action == "complete_care_task"
    assert complete_result.memory_actions[0].query == "喝水"


def test_extractor_normalizes_model_memory_actions():
    """Model-emitted memory_actions are normalized into runtime-ready actions."""
    result = _extraction_from_json(
        "sess_8",
        {
            "memory_actions": [
                {
                    "action": "create_reminder",
                    "title": "每天晚饭后提醒喝水",
                    "task_type": "reminder",
                    "confidence": 0.9,
                    "requires_confirmation": False,
                },
                {
                    "action": "forget_memory",
                    "query": "越剧",
                    "confidence": 0.9,
                },
            ]
        },
        transcript="user: 每天晚饭后提醒我喝水。另外忘掉越剧。",
    )

    assert [action.action for action in result.memory_actions] == ["create_care_task", "forget_user_fact"]
    assert result.memory_actions[0].title == "晚饭后喝水"
    assert result.memory_actions[0].task_type == "hydration"
    assert result.memory_actions[1].query == "越剧"


def test_extractor_forces_unconfirmed_medication_actions_to_pending():
    """Medication reminders require confirmation even if the model marks them confirmed."""
    result = _extraction_from_json(
        "sess_9",
        {
            "memory_actions": [
                {
                    "action": "create_care_task",
                    "title": "服用降压药",
                    "task_type": "medication",
                    "recurrence_rule": "FREQ=DAILY;BYHOUR=8;BYMINUTE=0",
                    "confidence": 0.9,
                    "requires_confirmation": False,
                }
            ]
        },
        transcript="user: 每天早上提醒我吃降压药。",
    )

    action = result.memory_actions[0]
    assert action.action == "create_care_task"
    assert action.title == "早上吃降压药"
    assert action.task_type == "medication"
    assert action.requires_confirmation is True


def test_extractor_normalizes_care_task_candidate_titles_with_transcript():
    """Care task candidates use the same title normalization as memory_actions."""
    result = _extraction_from_json(
        "sess_10",
        {
            "care_task_candidates": [
                {
                    "title": "服用降压药",
                    "task_type": "medication",
                    "confidence": 0.9,
                    "requires_confirmation": False,
                }
            ]
        },
        transcript="user: 每天早上提醒我吃降压药。",
    )

    assert len(result.care_task_candidates) == 1
    assert result.care_task_candidates[0].title == "早上吃降压药"
    assert result.care_task_candidates[0].task_type == "medication"


def test_extractor_normalizes_wake_time_value_to_chinese():
    """Wake-time profile values stay readable in injected memory context."""
    result = _extraction_from_json(
        "sess_11",
        {
            "profile_candidates": [
                {
                    "key": "routine.wake_time",
                    "value": "06:30",
                    "category": "routine",
                    "confidence": 0.9,
                    "requires_confirmation": False,
                }
            ]
        },
        transcript="user: 我通常早上六点半起床。",
    )

    assert len(result.profile_candidates) == 1
    assert result.profile_candidates[0].key == "routine.wake_time"
    assert result.profile_candidates[0].value == "六点半"


def test_extractor_forces_non_sensitive_task_actions_active():
    """Assistant uncertainty should not make explicit hydration reminders pending."""
    result = _extraction_from_json(
        "sess_12",
        {
            "memory_actions": [
                {
                    "action": "create_care_task",
                    "title": "晚饭后喝水",
                    "task_type": "hydration",
                    "confidence": 0.9,
                    "requires_confirmation": True,
                }
            ]
        },
        transcript=(
            "user: 每天晚饭后提醒我喝水。\n"
            "assistant: 好的，不过我没法主动在每天晚饭后提醒你喝水。你可以设个闹钟。"
        ),
    )

    action = result.memory_actions[0]
    assert action.action == "create_care_task"
    assert action.title == "晚饭后喝水"
    assert action.task_type == "hydration"
    assert action.requires_confirmation is False


def test_extractor_keeps_sensitive_health_task_actions_pending():
    """Blood-pressure follow-up tasks should stay pending unless explicitly confirmed."""
    result = _extraction_from_json(
        "sess_13",
        {
            "memory_actions": [
                {
                    "action": "create_care_task",
                    "title": "复查血压",
                    "task_type": "appointment",
                    "confidence": 0.9,
                    "requires_confirmation": False,
                }
            ]
        },
        transcript=(
            "user: 医生说我最近血压有点高，但我还没复查。\n"
            "assistant: 明白，先别担心，按时复查最重要。"
        ),
    )

    action = result.memory_actions[0]
    assert action.action == "create_care_task"
    assert action.title == "复查血压"
    assert action.task_type == "appointment"
    assert action.requires_confirmation is True
