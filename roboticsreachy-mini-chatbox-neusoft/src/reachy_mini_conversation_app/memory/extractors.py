"""Background model extractors for middle-term and long-term memories."""

from __future__ import annotations
import os
import re
import json
import logging
from typing import Any, Protocol

from reachy_mini_conversation_app.memory.models import (
    Turn,
    MemoryAction,
    SessionSummary,
    MemoryCandidate,
    ExtractionResult,
    CareTaskCandidate,
)
from reachy_mini_conversation_app.memory.prompts import (
    MEMORY_EXTRACTION_SYSTEM_PROMPT,
    MEMORY_EXTRACTION_USER_TEMPLATE,
)


logger = logging.getLogger(__name__)

CANONICAL_KEY_ALIASES: dict[str, str] = {
    "identity.preferred_name": "preferred_name",
    "communication.speech_rate": "communication.speaking_pace",
    "communication.speech_rate_preference": "communication.speaking_pace",
    "communication.pace": "communication.speaking_pace",
    "communication.audio_style": "communication.voice_style",
    "communication.voice": "communication.voice_style",
    "communication.tone": "communication.voice_style",
    "music.preference.typical": "preference.audio_style",
    "music_preference_typical": "preference.audio_style",
    "music.preference": "preference.audio_style",
    "audio.preference": "preference.audio_style",
    "routine.weekend_visits": "family.visit_pattern",
    "symptom.dizziness_post_lunch": "health.dizziness_after_lunch",
    "health.dizziness_post_lunch": "health.dizziness_after_lunch",
    "sleep.quality_concern": "health.dizziness_after_lunch",
    "health.hearing": "health.hearing_note",
    "health.hearing_status": "health.hearing_note",
    "medication.aspirin": "medication.current",
    "medication.aspirin.use": "medication.current",
    "medication.usage": "medication.current",
    "scam_vulnerability_status": "safety.scam_risk",
}

NON_SENSITIVE_ACTIVE_KEYS = {
    "preferred_name",
    "communication.speaking_pace",
    "communication.voice_style",
    "communication.language_preference",
    "preference.likes",
    "preference.dislikes",
    "preference.audio_style",
    "routine.wake_time",
    "routine.nap",
    "family.daughter.name",
    "family.son.name",
    "family.grandchild.name",
    "family.visit_pattern",
    "care_preference.reminder_style",
}

SENSITIVE_KEYS = {
    "health.dizziness_after_lunch",
    "health.hearing_note",
    "health.blood_pressure",
    "medication.current",
    "contact.emergency_person",
    "contact.phone",
    "address.home",
    "safety.scam_risk",
}

SENSITIVE_TASK_TEXT_TOKENS = (
    "血压",
    "血糖",
    "降压药",
    "阿司匹林",
    "服药",
    "吃药",
    "药",
    "头晕",
    "听力",
)

TASK_TYPE_ALIASES = {
    "medical": "check_in",
    "medicine": "medication",
    "anti_fraud": "safety",
    "fraud": "safety",
}

ACTION_ALIASES = {
    "add_care_task": "create_care_task",
    "add_reminder": "create_care_task",
    "create_reminder": "create_care_task",
    "create_task": "create_care_task",
    "mark_care_task_done": "complete_care_task",
    "mark_task_completed": "complete_care_task",
    "complete_task": "complete_care_task",
    "disable_reminder": "disable_care_task",
    "cancel_reminder": "disable_care_task",
    "stop_reminder": "disable_care_task",
    "delete_care_task": "disable_care_task",
    "remove_care_task": "disable_care_task",
    "archive_profile_fact": "forget_user_fact",
    "delete_profile_fact": "forget_user_fact",
    "forget_memory": "forget_user_fact",
    "delete_memory_note": "delete_memory_note",
    "archive_memory_note": "delete_memory_note",
}


class MemoryExtractor(Protocol):
    """Async interface for post-session memory extraction."""

    async def extract(self, *, session_id: str, turns: list[Turn], memory_context: str) -> ExtractionResult:
        """Extract memory material from final transcript turns."""
        ...


class NoopMemoryExtractor:
    """Extractor used when no background model is configured."""

    async def extract(self, *, session_id: str, turns: list[Turn], memory_context: str) -> ExtractionResult:
        """Return an empty result without calling a model."""
        return ExtractionResult()


class QwenMemoryExtractor:
    """Qwen-compatible extractor using DashScope's OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "qwen-plus",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout_s: float = 20.0,
    ):
        """Create a Qwen extractor. Calls happen only when ``extract`` is awaited."""
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout_s = timeout_s

    async def extract(self, *, session_id: str, turns: list[Turn], memory_context: str) -> ExtractionResult:
        """Call Qwen text model once after a realtime session ends."""
        from openai import AsyncOpenAI

        transcript = "\n".join(f"{turn.role}: {turn.content}" for turn in turns if turn.content.strip())
        if not transcript.strip():
            return ExtractionResult()

        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout_s)
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": MEMORY_EXTRACTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": MEMORY_EXTRACTION_USER_TEMPLATE.format(
                        memory_context=memory_context or "(none)",
                        transcript=transcript,
                    ),
                },
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        data = _parse_json_object(content)
        return _extraction_from_json(session_id, data, transcript=transcript)


def create_default_extractor() -> MemoryExtractor:
    """Create a background extractor from env configuration."""
    mode = os.getenv("REACHY_MINI_MEMORY_EXTRACTOR", "auto").strip().lower()
    if mode in {"none", "noop", "off", "disabled"}:
        return NoopMemoryExtractor()

    api_key = os.getenv("QWEN_MEMORY_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
    if not api_key:
        logger.info("Memory extractor disabled: no Qwen/DashScope key configured.")
        return NoopMemoryExtractor()

    model = os.getenv("QWEN_MEMORY_MODEL", "qwen-plus")
    base_url = os.getenv("QWEN_MEMORY_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    timeout_s = float(os.getenv("QWEN_MEMORY_TIMEOUT_S", "20"))
    return QwenMemoryExtractor(api_key=api_key, model=model, base_url=base_url, timeout_s=timeout_s)


def _parse_json_object(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            logger.warning("Memory extractor returned non-JSON content: %r", content[:200])
            return {}
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            logger.warning("Memory extractor JSON parse failed: %r", content[:200])
            return {}
    return parsed if isinstance(parsed, dict) else {}


def _extraction_from_json(session_id: str, data: dict[str, Any], *, transcript: str = "") -> ExtractionResult:
    action_only = _is_action_only_command(transcript)
    summary = None
    raw_summary = data.get("summary")
    if isinstance(raw_summary, dict):
        summary_text = str(raw_summary.get("summary") or "").strip()
        if summary_text:
            summary = SessionSummary(
                session_id=session_id,
                summary=summary_text,
                topics=_string_list(raw_summary.get("topics")),
                emotions=_string_list(raw_summary.get("emotions")),
                follow_ups=_string_list(raw_summary.get("follow_ups")),
                risks=_string_list(raw_summary.get("risks")),
            )
    elif isinstance(raw_summary, str) and raw_summary.strip():
        summary = SessionSummary(session_id=session_id, summary=raw_summary.strip())
    if summary and (action_only or _is_action_only_command(summary.summary)):
        summary = None

    memory_actions = _parse_memory_actions(data.get("memory_actions"), transcript=transcript)

    profile_candidates: list[MemoryCandidate] = []
    if not action_only:
        for item in _dict_list(data.get("profile_candidates")):
            key = str(item.get("key") or "").strip()
            value = str(item.get("value") or "").strip()
            if not key or not value:
                continue
            evidence = _normalize_optional_text(item.get("evidence"))
            normalized_candidate = _normalize_profile_candidate(key, value, evidence=evidence, transcript=transcript)
            if normalized_candidate is None:
                continue
            key, value = normalized_candidate
            category = _normalize_category(key, str(item.get("category") or "preference").strip())
            profile_candidates.append(
                MemoryCandidate(
                    key=key,
                    value=value,
                    category=category,
                    confidence=_float(item.get("confidence"), 0.7),
                    source="extractor",
                    evidence=evidence,
                    requires_confirmation=_normalize_confirmation(key, item.get("requires_confirmation")),
                )
            )

    care_task_candidates: list[CareTaskCandidate] = []
    if not action_only:
        for item in _dict_list(data.get("care_task_candidates")):
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            normalized_title = _normalize_task_title(title, transcript=transcript)
            task_type = _normalize_task_type(str(item.get("task_type") or "reminder").strip())
            if task_type == "reminder":
                task_type = _infer_task_type(normalized_title)
            due_at = str(item.get("due_at") or "").strip() or None
            recurrence_rule = str(item.get("recurrence_rule") or "").strip() or None
            if _is_unanchored_generic_reminder(task_type, normalized_title, due_at, recurrence_rule):
                continue
            care_task_candidates.append(
                CareTaskCandidate(
                    title=normalized_title,
                    task_type=task_type,
                    due_at=due_at,
                    recurrence_rule=recurrence_rule,
                    confidence=_float(item.get("confidence"), 0.7),
                    source="extractor",
                    evidence=_normalize_optional_text(item.get("evidence")),
                    requires_confirmation=_normalize_care_task_confirmation(
                        task_type,
                        normalized_title,
                        item.get("requires_confirmation"),
                        transcript=_normalize_optional_text(item.get("evidence")) or "",
                    ),
                )
            )

    return ExtractionResult(
        summary=summary,
        profile_candidates=profile_candidates,
        memory_notes=_normalize_memory_notes(_string_list(data.get("memory_notes")), transcript=transcript),
        care_task_candidates=care_task_candidates,
        memory_actions=memory_actions,
    )


def _normalize_profile_key(key: str) -> str:
    cleaned = key.strip()
    if cleaned in {"茶", "绿茶", "红茶", "饮茶", "饮茶口味", "饮茶偏好", "茶口味", "喝茶偏好"}:
        return "preference.likes"
    return CANONICAL_KEY_ALIASES.get(cleaned, cleaned)


def _normalize_category(key: str, category: str) -> str:
    if key.startswith("communication."):
        return "communication"
    if key.startswith("preference."):
        return "preference"
    if key.startswith("routine."):
        return "routine"
    if key.startswith("family."):
        return "family"
    if key.startswith("health."):
        return "health"
    if key.startswith("medication."):
        return "medication"
    if key.startswith("contact."):
        return "contact"
    if key.startswith("address."):
        return "address"
    if key.startswith("safety."):
        return "safety"
    return category or "preference"


def _normalize_confirmation(key: str, raw_value: Any) -> bool | None:
    if key in SENSITIVE_KEYS:
        return True
    if key in NON_SENSITIVE_ACTIVE_KEYS:
        return False
    return _optional_bool(raw_value)


def _parse_memory_actions(value: Any, *, transcript: str) -> list[MemoryAction]:
    actions: list[MemoryAction] = []
    for item in _dict_list(value):
        action = _normalize_action_name(str(item.get("action") or item.get("type") or "").strip())
        if not action:
            continue
        normalized = _normalize_memory_action(action, item, transcript=transcript)
        if normalized is not None:
            actions.append(normalized)
    return _dedupe_memory_actions([*actions, *_infer_memory_actions_from_transcript(transcript)])


def _dedupe_memory_actions(actions: list[MemoryAction]) -> list[MemoryAction]:
    deduped: list[MemoryAction] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for action in actions:
        identity = (
            action.action,
            action.query or "",
            action.key or "",
            action.value or "",
            action.title or "",
        )
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(action)
    return deduped


def _normalize_memory_action(action: str, item: dict[str, Any], *, transcript: str) -> MemoryAction | None:
    query = _normalize_optional_text(item.get("query") or item.get("target") or item.get("target_query"))
    title = _normalize_optional_text(item.get("title"))
    value = _normalize_optional_text(item.get("value"))
    evidence = _normalize_optional_text(item.get("evidence"))
    task_type = _normalize_task_type(str(item.get("task_type") or "reminder").strip())

    if action in {"complete_care_task", "disable_care_task", "delete_memory_note", "forget_user_fact"}:
        query = query or title or value
        if not query:
            return None

    if action == "create_care_task":
        title = _normalize_task_title(title or query or value or "", transcript=transcript)
        if not title:
            return None
        raw_task_type = str(item.get("task_type") or "").strip()
        task_type = _normalize_task_type(raw_task_type) if raw_task_type else _infer_task_type(title)
        if task_type == "reminder":
            task_type = _infer_task_type(title)
        due_at = str(item.get("due_at") or "").strip() or None
        recurrence_rule = str(item.get("recurrence_rule") or "").strip() or None
        if _is_unanchored_generic_reminder(task_type, title, due_at, recurrence_rule):
            return None
        return MemoryAction(
            action=action,
            query=query,
            title=title,
            task_type=task_type,
            due_at=due_at,
            recurrence_rule=recurrence_rule,
            confidence=_float(item.get("confidence"), 0.7),
            evidence=evidence,
            requires_confirmation=_normalize_care_task_confirmation(
                task_type,
                title,
                item.get("requires_confirmation"),
                transcript=evidence or "",
            ),
        )

    if action in {"remember_user_fact", "update_user_fact"}:
        key = _normalize_profile_key(str(item.get("key") or "").strip())
        raw_value = str(item.get("value") or "").strip()
        if not key or not raw_value:
            return None
        normalized_value = _normalize_profile_value(key, raw_value)
        category = _normalize_category(key, str(item.get("category") or "preference").strip())
        return MemoryAction(
            action=action,
            query=query,
            key=key,
            value=normalized_value,
            category=category,
            confidence=_float(item.get("confidence"), 0.7),
            evidence=evidence,
            requires_confirmation=_normalize_confirmation(key, item.get("requires_confirmation")),
        )

    return MemoryAction(
        action=action,
        query=query,
        title=title,
        task_type=task_type,
        due_at=str(item.get("due_at") or "").strip() or None,
        recurrence_rule=str(item.get("recurrence_rule") or "").strip() or None,
        confidence=_float(item.get("confidence"), 0.7),
        evidence=evidence,
        requires_confirmation=_optional_bool(item.get("requires_confirmation")),
    )


def _normalize_action_name(action: str) -> str:
    cleaned = action.strip()
    return ACTION_ALIASES.get(cleaned, cleaned)


def _normalize_profile_candidate(
    key: str,
    value: str,
    *,
    evidence: str | None,
    transcript: str,
) -> tuple[str, str] | None:
    normalized_key = _normalize_profile_key(key)
    normalized_value = _normalize_profile_value(normalized_key, value)
    scoped_text = "\n".join(part for part in (normalized_value, evidence or "") if part)

    if _contains_deletion_command(scoped_text):
        return None

    if normalized_key.startswith("preference."):
        if _is_transient_preference(scoped_text):
            return None
        if _points_to_quiet_audio_style(scoped_text):
            return "preference.audio_style", "平时喜欢安静"

    if normalized_key == "communication.speaking_pace":
        normalized_value = _normalize_profile_value(normalized_key, scoped_text or transcript or value)

    return normalized_key, normalized_value


def _normalize_profile_value(key: str, value: str) -> str:
    text = _normalize_memory_text(value)
    lowered = text.lower()
    if key == "communication.speaking_pace" and any(
        token in lowered for token in ("slow", "slower", "speech rate", "说慢", "慢速", "慢一点", "放慢")
    ):
        return "说慢一点"
    if key == "communication.voice_style" and any(token in text for token in ("轻声", "小声", "温柔", "柔和")):
        return "轻声"
    if key == "preference.audio_style" and any(
        token in lowered for token in ("quiet", "calm", "安静")
    ):
        return "平时喜欢安静"
    if key == "routine.wake_time" and any(
        token in text for token in ("06:30", "6:30", "六点半", "6点半", "6点30", "六点三十")
    ):
        return "六点半"
    if key == "medication.current" and ("aspirin" in lowered or "阿司匹林" in text):
        return "阿司匹林"
    if key == "health.hearing_note" and any(
        token in lowered for token in ("hearing", "poor", "impaired", "耳朵", "听")
    ):
        return "听力不佳，需确认"
    if key == "safety.scam_risk" and any(token in lowered for token in ("scam", "fraud", "中奖", "保证金")):
        return "提到疑似诈骗风险，需确认"
    return text


def _normalize_care_task_confirmation(
    task_type: str,
    title: str,
    raw_value: Any,
    *,
    transcript: str,
) -> bool | None:
    if _is_sensitive_care_task(task_type, title, transcript=transcript):
        return False if _has_explicit_confirmation(transcript) else True
    if task_type == "safety":
        return False if _has_explicit_confirmation(transcript) else True
    if task_type in {"hydration", "exercise", "reminder", "appointment", "check_in"}:
        return False
    return _optional_bool(raw_value)


def _is_sensitive_care_task(task_type: str, title: str, *, transcript: str) -> bool:
    scoped_text = f"{title}\n{transcript}"
    return task_type == "medication" or any(token in scoped_text for token in SENSITIVE_TASK_TEXT_TOKENS)


def _is_unanchored_generic_reminder(
    task_type: str,
    title: str,
    due_at: str | None,
    recurrence_rule: str | None,
) -> bool:
    """Reject abstract reminders that are not anchored to a concrete care action or time."""
    if task_type != "reminder":
        return False
    if recurrence_rule:
        return False
    if due_at and _is_schedule_phrase(due_at):
        return False
    return _infer_task_type(title) == "reminder"


def _is_schedule_phrase(text: str) -> bool:
    return any(
        token in text
        for token in (
            "每天",
            "每日",
            "早上",
            "上午",
            "中午",
            "午饭",
            "下午",
            "晚饭",
            "晚上",
            "睡前",
            "起床",
            "周一",
            "周二",
            "周三",
            "周四",
            "周五",
            "周六",
            "周日",
            "星期",
            "明天",
            "后天",
            "点",
            ":",
        )
    )


def _normalize_memory_text(text: str) -> str:
    replacements = {
        "scam": "诈骗",
        "Scam": "诈骗",
        "fraud": "诈骗",
        "Fraud": "诈骗",
        "anti-fraud": "反诈骗",
        "Anti-Fraud": "反诈骗",
        "防诈": "防诈骗",
        "aspirin": "阿司匹林",
        "Aspirin": "阿司匹林",
        "slower speech": "说慢一点",
        "slower": "更慢",
        "quiet music": "安静的音乐",
        "calm music": "安静的音乐",
    }
    normalized = text.strip()
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return normalized


def _normalize_memory_notes(notes: list[str], *, transcript: str) -> list[str]:
    if _is_action_only_command(transcript):
        return []

    normalized_notes: list[str] = []
    for note in notes:
        normalized = _normalize_memory_text(note)
        if not normalized or _is_cleanup_command(normalized):
            continue
        normalized_notes.append(normalized)
    return normalized_notes


def _is_cleanup_command(text: str) -> bool:
    return _contains_deletion_command(text) or _contains_reminder_disable_command(text)


def _is_action_only_command(text: str) -> bool:
    if _count_user_turns(text) > 1:
        return False
    return _is_cleanup_command(text) or _contains_task_completion_command(text)


def _count_user_turns(text: str) -> int:
    return len(re.findall(r"(?:^|\n)\s*user\s*:", text, flags=re.IGNORECASE))


def _contains_deletion_command(text: str) -> bool:
    return any(
        token in text
        for token in (
            "忘掉",
            "忘记",
            "别记",
            "不要记",
            "删除",
            "清除",
            "取消记忆",
            "forget",
            "delete",
        )
    )


def _contains_reminder_disable_command(text: str) -> bool:
    if "提醒" in text and any(token in text for token in ("取消", "停止", "不用", "不要", "别再", "已取消")):
        return True
    return any(
        token in text
        for token in (
            "不用提醒",
            "不要提醒",
            "别提醒",
            "不用再提醒",
            "不要再提醒",
            "别再提醒",
            "取消提醒",
            "停止提醒",
            "不希望再被提醒",
            "不希望接收",
            "disable reminder",
            "cancel reminder",
            "stop reminder",
        )
    )


def _contains_task_completion_command(text: str) -> bool:
    return any(
        token in text
        for token in (
            "我已经",
            "我刚刚",
            "已经喝",
            "已经吃",
            "已经做",
            "刚喝",
            "刚吃",
            "完成了",
            "做完了",
            "done",
            "completed",
        )
    )


def _infer_memory_actions_from_transcript(transcript: str) -> list[MemoryAction]:
    text = _strip_role_prefixes(transcript)
    actions: list[MemoryAction] = []
    for line in [line.strip() for line in text.splitlines() if line.strip()]:
        visit_pattern = _match_family_visit_pattern_action(line)
        if visit_pattern:
            actions.append(MemoryAction(action="update_user_fact", confidence=0.95, evidence=line, **visit_pattern))

    for line in _command_lines(text):
        forget_query = _match_forget_query(line)
        if forget_query:
            actions.append(MemoryAction(action="forget_user_fact", query=forget_query, confidence=0.95, evidence=line))
            continue

        disable_query = _match_disable_reminder_query(line)
        if disable_query:
            actions.append(MemoryAction(action="disable_care_task", query=disable_query, confidence=0.95, evidence=line))
            continue

        complete_query = _match_complete_task_query(line)
        if complete_query:
            actions.append(MemoryAction(action="complete_care_task", query=complete_query, confidence=0.95, evidence=line))
            continue

        profile_update = _match_profile_update_action(line)
        if profile_update:
            actions.append(MemoryAction(action="update_user_fact", confidence=0.95, evidence=line, **profile_update))
            continue

        create_args = _match_create_reminder_action(line)
        if create_args:
            actions.append(MemoryAction(action="create_care_task", confidence=0.95, evidence=line, **create_args))
    return actions


def _command_lines(text: str) -> list[str]:
    lines: list[str] = []
    for line in text.splitlines() or [text]:
        for segment in re.split(r"[，。；;]+", line):
            cleaned = _cleanup_value(segment.strip())
            if cleaned:
                lines.append(cleaned)
    return lines


def _match_forget_query(text: str) -> str | None:
    match = _search_line(r"^(?:忘掉|忘记|删除|不要记得|别记得|帮我忘掉)(.+)$", text)
    if match:
        return _cleanup_value(match.group(1))
    match = _search_line(r"^(.+?)(?:先)?(?:忘掉|忘记|删除|不要记得|别记得)$", text)
    return _cleanup_value(match.group(1)) if match else None


def _match_profile_update_action(text: str) -> dict[str, Any] | None:
    match = _search_line(r"^(?P<topic>.+?)(?:改成|改为|换成|换为|改叫|改用)(?P<value>.+)$", text)
    if not match:
        return None
    topic = _cleanup_value(match.group("topic"))
    value = _cleanup_value(match.group("value"))
    if not topic or not value or _contains_deletion_command(value):
        return None
    key = _infer_profile_key_from_topic(topic, value)
    if key is None:
        return None
    return {
        "key": key,
        "value": _normalize_profile_value(key, value),
        "category": _normalize_category(key, "preference"),
        "requires_confirmation": _normalize_confirmation(key, None),
    }


def _infer_profile_key_from_topic(topic: str, value: str) -> str | None:
    scoped_text = f"{topic}\n{value}"
    if any(token in scoped_text for token in ("称呼", "叫我", "名字")):
        return "preferred_name"
    if any(token in scoped_text for token in ("语速", "说话速度", "说慢", "说快")):
        return "communication.speaking_pace"
    if any(token in scoped_text for token in ("声音", "声调", "轻声", "小声", "温柔")):
        return "communication.voice_style"
    if any(token in scoped_text for token in ("语言", "普通话", "粤语", "方言")):
        return "communication.language_preference"
    if any(token in scoped_text for token in ("起床", "醒")):
        return "routine.wake_time"
    if any(token in scoped_text for token in ("午休", "午睡", "小睡")):
        return "routine.nap"
    if any(token in scoped_text for token in ("喜欢", "偏好", "口味", "饮茶", "喝茶", "听", "吃")):
        return "preference.likes"
    return None


def _match_family_visit_pattern_action(text: str) -> dict[str, Any] | None:
    if not any(token in text for token in ("来看", "看我", "来访", "探望", "陪我")):
        return None
    if not any(token in text for token in ("女儿", "儿子", "外孙", "孙女", "孙子", "家人", "老伴", "亲人")):
        return None
    schedule = _extract_schedule_phrase(text)
    if schedule is None:
        return None
    subject = "家人"
    for candidate in ("女儿", "儿子", "外孙", "孙女", "孙子", "老伴", "家人", "亲人"):
        if candidate in text:
            subject = candidate
            break
    prefix = "每" if any(token in text for token in ("每周", "常", "固定", "通常", "一般")) else ""
    return {
        "key": "family.visit_pattern",
        "value": f"{subject}{prefix}{schedule}来访",
        "category": "family",
        "requires_confirmation": False,
    }


def _extract_schedule_phrase(text: str) -> str | None:
    match = re.search(r"(周[一二三四五六日天]|星期[一二三四五六日天])(?:上午|中午|下午|晚上)?", text)
    if match:
        return match.group(0)
    match = re.search(r"(每天|每日)?(?:早上|上午|中午|午饭后|下午|晚饭后|晚上|睡前)", text)
    return match.group(0) if match else None


def _match_disable_reminder_query(text: str) -> str | None:
    match = _search_line(r"^(?:以后|之后|今后)?(?:不用|不要|别)(?:再)?提醒我(.+)$", text)
    if match:
        return _cleanup_task_query(match.group(1))
    match = _search_line(r"^(?P<query>.+?)(?:那个|这个)?提醒(?:不用|取消|停止)(?:了)?$", text)
    return _cleanup_task_query(match.group("query")) if match else None


def _cleanup_task_query(value: str) -> str:
    cleaned = _cleanup_value(value)
    return cleaned.removesuffix("那个").removesuffix("这个").strip()


def _match_complete_task_query(text: str) -> str | None:
    match = _search_line(r"^我(?:已经|刚刚)?(.+?)(?:了|过了)$", text)
    if not match:
        return None
    value = _cleanup_value(match.group(1))
    if any(token in value for token in ("喝水", "吃药", "服药", "运动", "散步", "拉伸", "量血压", "量血糖")):
        return value
    return None


def _match_create_reminder_action(text: str) -> dict[str, Any] | None:
    if "提醒我" not in text:
        return None
    if _contains_reminder_disable_command(text):
        return None
    cleaned = _cleanup_value(text)
    match = _search_line(r"^(?P<when>.+?)提醒我(?P<action>.+)$", cleaned)
    if not match:
        return None
    when = _cleanup_value(match.group("when").replace("我确认", "").replace("确认", ""))
    action = _cleanup_value(match.group("action"))
    if not when or not action:
        return None
    if not _is_schedule_phrase(when):
        return None
    confirmed = _has_explicit_confirmation(text)
    title_when = when.replace("每天", "").replace("每日", "")
    title = _normalize_task_title(f"{title_when}{action}", transcript=text)
    if not title:
        return None
    recurrence_rule = "daily" if any(token in when for token in ("每天", "每日")) else None
    return {
        "title": title,
        "task_type": _infer_task_type(title),
        "due_at": None if recurrence_rule else when,
        "recurrence_rule": recurrence_rule,
        "requires_confirmation": (
            True if _is_sensitive_care_task(_infer_task_type(title), title, transcript=text) and not confirmed else False
        ),
    }


def _normalize_task_title(title: str, *, transcript: str) -> str:
    text = _normalize_memory_text(" ".join(part for part in (title, transcript) if part))
    if "晚饭后" in text and "喝水" in text:
        return "晚饭后喝水"
    if "早上" in text and "降压药" in text:
        return "早上吃降压药"
    if "早饭后" in text and any(token in text for token in ("服药", "吃药", "药")):
        return "早饭后服药"
    if "社区医院" in text and "复诊" in text:
        return "周五上午社区医院复诊" if "周五" in text else "社区医院复诊"
    if "复查" in text and "血压" in text:
        return "复查血压"
    if "睡前" in text and "拉伸" in text:
        return "睡前拉伸"
    return _cleanup_value(text)


def _has_explicit_confirmation(text: str) -> bool:
    return any(token in text for token in ("我确认", "确认要", "确定要", "确认记住", "确认提醒"))


def _infer_task_type(text: str) -> str:
    if "喝水" in text:
        return "hydration"
    if any(token in text for token in ("吃药", "服药", "药")):
        return "medication"
    if any(token in text for token in ("复诊", "医院", "门诊", "医生", "检查")):
        return "appointment"
    if any(token in text for token in ("散步", "拉伸", "运动", "锻炼")):
        return "exercise"
    if any(token in text for token in ("诈骗", "反诈", "安全")):
        return "safety"
    return "reminder"


def _strip_role_prefixes(transcript: str) -> str:
    lines = []
    for line in transcript.splitlines():
        cleaned = line.strip()
        if cleaned.startswith("user:"):
            cleaned = cleaned[5:].strip()
        elif cleaned.startswith("assistant:"):
            continue
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines) or transcript.strip()


def _search_line(pattern: str, text: str) -> re.Match[str] | None:
    for line in text.splitlines() or [text]:
        match = re.match(pattern, _cleanup_value(line.strip()))
        if match:
            return match
    return None


def _cleanup_value(value: str) -> str:
    return value.strip(" ，,。.!！?？：:")


def _is_transient_preference(text: str) -> bool:
    temporary_cues = ("今天", "这次", "刚才", "这会儿", "现在就", "就想", "临时", "偶尔", "此刻", "一会儿", "今晚")
    stable_cues = ("平时", "通常", "一直", "长期", "以后", "每天", "一般", "习惯", "还是喜欢")
    return any(cue in text for cue in temporary_cues) and not any(cue in text for cue in stable_cues)


def _points_to_quiet_audio_style(text: str) -> bool:
    return any(token in text for token in ("平时喜欢安静", "还是喜欢安静", "喜欢安静", "安静的音乐", "安静风格"))


def _normalize_optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return _normalize_memory_text(text) if text else None


def _normalize_task_type(task_type: str) -> str:
    cleaned = task_type.strip()
    return TASK_TYPE_ALIASES.get(cleaned, cleaned)


def _dict_list(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None
