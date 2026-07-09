"""Low-latency router for explicit memory commands in Qwen router mode."""

from __future__ import annotations
import re
import hashlib
from typing import Any

from reachy_mini_conversation_app.memory.runtime import MemoryRuntime


SENSITIVE_HINTS = ("血压", "血糖", "药", "过敏", "医院", "医生", "急救", "紧急", "电话", "地址", "病")


class MemoryCommandRouter:
    """Handle only explicit memory CRUD commands, not implicit extraction."""

    def __init__(self, runtime: MemoryRuntime):
        """Create a router bound to a runtime."""
        self.runtime = runtime

    async def handle(self, transcript: str) -> dict[str, Any] | None:
        """Route an explicit transcript to a memory operation, if any."""
        text = _cleanup_value(transcript.strip())
        if not text:
            return None

        preferred_name = self._match_preferred_name(text)
        if preferred_name:
            fact = self.runtime.remember_user_fact(
                key="preferred_name",
                value=preferred_name,
                category="identity",
                source="user_explicit",
                evidence=text,
            )
            return {"type": "memory_saved", "fact": self.runtime._fact_to_dict(fact) if fact else None}

        forget_query = self._match_forget(text)
        if forget_query:
            archived = self.runtime.forget_user_fact(forget_query)
            notes = self.runtime.store.delete_memory_note(self.runtime.user.id, forget_query)
            return {
                "type": "memory_forgotten",
                "archived_facts": len(archived),
                "archived_notes": len(notes),
                "query": forget_query,
            }

        disable_query = self._match_disable_reminder(text)
        if disable_query:
            tasks = self.runtime.disable_care_tasks(disable_query)
            return {"type": "care_tasks_disabled", "count": len(tasks), "query": disable_query}

        complete_query = self._match_completed_task(text)
        if complete_query:
            task = self.runtime.complete_care_task(query=complete_query)
            return {
                "type": "care_task_completed",
                "task": self.runtime._task_to_dict(task) if task else None,
                "query": complete_query,
            }

        reminder_args = self._match_create_reminder(text)
        if reminder_args:
            task = self.runtime.create_care_task(**reminder_args)
            return {
                "type": "care_task_created",
                "task": self.runtime._task_to_dict(task) if task else None,
            }

        explicit_memory = self._match_remember(text)
        if explicit_memory:
            key, value, category = self._classify_explicit_memory(explicit_memory)
            fact = self.runtime.remember_user_fact(
                key=key,
                value=value,
                category=category,
                source="user_explicit",
                evidence=text,
            )
            return {"type": "memory_saved", "fact": self.runtime._fact_to_dict(fact) if fact else None}

        preference = self._match_preference(text)
        if preference:
            key, value, category = self._classify_explicit_memory(preference)
            fact = self.runtime.remember_user_fact(
                key=key,
                value=value,
                category=category,
                source="user_explicit",
                evidence=text,
            )
            return {"type": "memory_saved", "fact": self.runtime._fact_to_dict(fact) if fact else None}

        return None

    @staticmethod
    def _match_preferred_name(text: str) -> str | None:
        patterns = [
            r"^(?:以后|之后|今后)?(?:请)?叫我(.+)$",
            r"^记住(?:一下)?[，, ]*(?:以后|之后|今后)?(?:请)?叫我(.+)$",
        ]
        for pattern in patterns:
            match = re.match(pattern, text)
            if match:
                return _cleanup_value(match.group(1))
        return None

    @staticmethod
    def _match_forget(text: str) -> str | None:
        match = re.match(r"^(?:忘掉|删除|不要记得|别记得|帮我忘掉)(.+)$", text)
        return _cleanup_value(match.group(1)) if match else None

    @staticmethod
    def _match_disable_reminder(text: str) -> str | None:
        match = re.match(r"^(?:以后|之后|今后)?(?:不用|不要|别)(?:再)?提醒我(.+)$", text)
        return _cleanup_value(match.group(1)) if match else None

    @staticmethod
    def _match_completed_task(text: str) -> str | None:
        match = re.match(r"^我(?:已经|刚刚)?(.+?)(?:了|过了)$", text)
        if not match:
            return None
        value = _cleanup_value(match.group(1))
        if any(token in value for token in ("喝水", "吃药", "运动", "散步", "量血压", "量血糖")):
            return value
        return None

    @staticmethod
    def _match_create_reminder(text: str) -> dict[str, Any] | None:
        if "提醒我" not in text:
            return None
        if re.search(r"(?:不用|不要|别)(?:再)?提醒我", text):
            return None
        confirmed = any(token in text for token in ("我确认", "确认每天", "确认要", "确定"))
        cleaned = re.sub(r"^(?:我确认|确认|确定)(?:要)?", "", text).strip()
        match = re.match(r"^(?P<when>.+?)提醒我(?P<action>.+)$", cleaned)
        if not match:
            return None
        when = _cleanup_value(match.group("when"))
        action = _cleanup_value(re.sub(r"^(?:去|做|进行)", "", match.group("action")))
        if not when or not action:
            return None

        task_type = _classify_task_type(action)
        title_when = re.sub(r"^(?:每天|每日)", "", when)
        title = _cleanup_value(f"{title_when}{action}")
        recurrence_rule = "daily" if any(token in when for token in ("每天", "每日")) else None
        due_at = None if recurrence_rule else when

        return {
            "title": title,
            "task_type": task_type,
            "due_at": due_at,
            "recurrence_rule": recurrence_rule,
            "source": "user_explicit",
            "confirmed": confirmed,
        }

    @staticmethod
    def _match_remember(text: str) -> str | None:
        match = re.match(r"^(?:记住|帮我记住|请记住|记一下)(?:一下)?[，, ]*(.+)$", text)
        return _cleanup_value(match.group(1)) if match else None

    @staticmethod
    def _match_preference(text: str) -> str | None:
        match = re.match(r"^(我(?:喜欢|爱|不喜欢|讨厌).+)$", text)
        return _cleanup_value(match.group(1)) if match else None

    @staticmethod
    def _classify_explicit_memory(value: str) -> tuple[str, str, str]:
        category = "health" if any(token in value for token in SENSITIVE_HINTS) else "preference"
        if value.startswith("我喜欢") or value.startswith("我爱"):
            cleaned = _cleanup_value(re.sub(r"^我(?:喜欢|爱)", "", value))
            return f"preference.likes.{_stable_suffix(cleaned)}", cleaned, "preference"
        if value.startswith("我不喜欢") or value.startswith("我讨厌"):
            cleaned = _cleanup_value(re.sub(r"^我(?:不喜欢|讨厌)", "", value))
            return f"preference.dislikes.{_stable_suffix(cleaned)}", cleaned, "preference"
        return f"{category}.note.{_stable_suffix(value)}", value, category


def _cleanup_value(value: str) -> str:
    return value.strip(" ，,。.!！?？：:")


def _classify_task_type(text: str) -> str:
    if "喝水" in text:
        return "hydration"
    if any(token in text for token in ("吃药", "服药", "药")):
        return "medication"
    if any(token in text for token in ("复诊", "医院", "门诊", "医生", "检查")):
        return "appointment"
    if any(token in text for token in ("散步", "拉伸", "运动", "锻炼")):
        return "exercise"
    return "reminder"


def _stable_suffix(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
