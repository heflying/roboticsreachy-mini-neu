"""Tool for setting an alarm."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.scheduler.engine import Scheduler
from reachy_mini_conversation_app.scheduler.models import ScheduledEvent

# Global scheduler reference — set by handler during initialization
_scheduler: Scheduler | None = None


def set_scheduler(scheduler: Scheduler) -> None:
    """Set the global scheduler instance for tools to use."""
    global _scheduler
    _scheduler = scheduler


def get_scheduler() -> Scheduler | None:
    """Get the global scheduler instance."""
    return _scheduler


def _parse_time(time_str: str) -> datetime:
    """Parse a time string into a datetime.

    Supports:
        - ISO format: "2026-06-23T20:00"
        - Relative: "1小时后", "30分钟后", "明天20:00"
        - Time only: "20:00" (today)
    """
    now = datetime.now()

    # Try ISO format
    try:
        return datetime.fromisoformat(time_str)
    except ValueError:
        pass

    # Try relative Chinese expressions
    import re

    # "X小时后"
    match = re.match(r"(\d+)\s*小时后", time_str)
    if match:
        hours = int(match.group(1))
        return now + timedelta(hours=hours)

    # "X分钟后"
    match = re.match(r"(\d+)\s*分钟后", time_str)
    if match:
        minutes = int(match.group(1))
        return now + timedelta(minutes=minutes)

    # "明天 HH:MM"
    match = re.match(r"明天\s*(\d{1,2}):(\d{2})", time_str)
    if match:
        hour, minute = int(match.group(1)), int(match.group(2))
        tomorrow = now + timedelta(days=1)
        return tomorrow.replace(hour=hour, minute=minute, second=0, microsecond=0)

    # "HH:MM" (today, or tomorrow if already past)
    match = re.match(r"(\d{1,2}):(\d{2})$", time_str)
    if match:
        hour, minute = int(match.group(1)), int(match.group(2))
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return target

    # "YYYY-MM-DD HH:MM"
    try:
        return datetime.strptime(time_str, "%Y-%m-%d %H:%M")
    except ValueError:
        pass

    # "YYYY-MM-DD"
    try:
        return datetime.strptime(time_str, "%Y-%m-%d")
    except ValueError:
        pass

    # Default: treat as ISO with seconds
    raise ValueError(f"Cannot parse time: {time_str}")


def _normalize_repeat(repeat: str) -> str:
    """Normalize repeat rule from LLM output to internal format."""
    repeat_lower = repeat.strip().lower()

    mapping = {
        "once": "once",
        "一次": "once",
        "不重复": "once",
        "daily": "daily",
        "每天": "daily",
        "每日": "daily",
        "weekly": "weekly:mon",  # default to Monday, LLM should specify day
        "每周": "weekly:mon",
        "hourly": "hourly:1",
        "每小时": "hourly:1",
    }

    # Check exact matches
    if repeat_lower in mapping:
        return mapping[repeat_lower]

    # Check prefixes
    for prefix, value in mapping.items():
        if repeat_lower.startswith(prefix):
            return value

    return repeat_lower


class SetAlarmTool(Tool):
    """Set an alarm/reminder."""

    name = "set_alarm"
    description = (
        "设置闹钟提醒。用于设置一次性或重复的定时提醒，如吃药提醒、喝水提醒等。"
        "time 参数支持：ISO格式(2026-06-23T20:00)、相对时间(1小时后、30分钟后)、"
        "具体时间(20:00、明天08:30)。repeat 参数：once(一次)、daily(每天)、hourly:N(每N小时)。"
    )
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "time": {
                "type": "string",
                "description": "提醒时间，支持 ISO 格式、相对时间（如'1小时后'）、具体时间（如'20:00'）",
            },
            "message": {
                "type": "string",
                "description": "提醒内容，如'该吃降压药了'",
            },
            "repeat": {
                "type": "string",
                "description": "重复规则：once（默认，一次）、daily（每天）、hourly:N（每N小时）",
                "default": "once",
            },
            "priority": {
                "type": "string",
                "description": "优先级：urgent（紧急，如用药）、important（重要）、normal（普通，默认）",
                "default": "normal",
            },
        },
        "required": ["time", "message"],
    }

    async def __call__(
        self,
        deps: ToolDependencies,
        time: str,
        message: str,
        repeat: str = "once",
        priority: str = "normal",
        **_: Any,
    ) -> Dict[str, Any]:
        scheduler = get_scheduler()
        if scheduler is None:
            return {"error": "Scheduler not initialized"}

        try:
            trigger_at = _parse_time(time)
        except ValueError as e:
            return {"error": f"无法解析时间 '{time}': {e}"}

        repeat_rule = _normalize_repeat(repeat)
        priority = priority if priority in ("urgent", "important", "normal") else "normal"

        event = ScheduledEvent(
            source="alarm",
            title=message,
            description="",
            trigger_at=trigger_at,
            recurrence_rule=repeat_rule,
            priority=priority,
            status="active",
        )

        created = scheduler.create_event(event)
        return {
            "status": "success",
            "alarm_id": created.id,
            "message": message,
            "trigger_at": created.trigger_at.isoformat() if created.trigger_at else None,
            "repeat": repeat_rule,
            "priority": priority,
        }
