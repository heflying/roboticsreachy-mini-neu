"""Tool for setting a calendar event."""

from __future__ import annotations
from datetime import datetime
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.tools.set_alarm import get_scheduler, _parse_time, _normalize_repeat
from reachy_mini_conversation_app.scheduler.models import ScheduledEvent


class SetCalendarEventTool(Tool):
    """Set a calendar event."""

    name = "set_calendar_event"
    description = (
        "设置日历事件。用于记录生日、体检、纪念日等日期事件。"
        "如果设置了 trigger_at，系统会在该时间自动提醒。"
        "repeat 参数：once（默认）、yearly（每年）、monthly（每月）。"
    )
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "date": {
                "type": "string",
                "description": "事件日期，格式如'2026-07-01'或'每年6月23日'",
            },
            "title": {
                "type": "string",
                "description": "事件标题，如'用户生日'、'年度体检'",
            },
            "description": {
                "type": "string",
                "description": "事件描述，可选",
                "default": "",
            },
            "trigger_at": {
                "type": "string",
                "description": "提醒时间（可选）。如果不设置，事件仅在被查询时显示。格式同 set_alarm 的 time 参数",
            },
            "repeat": {
                "type": "string",
                "description": "重复规则：once（默认，一次）、yearly（每年）、monthly（每月）",
                "default": "once",
            },
        },
        "required": ["date", "title"],
    }

    async def __call__(
        self,
        deps: ToolDependencies,
        date: str,
        title: str,
        description: str = "",
        trigger_at: str | None = None,
        repeat: str = "once",
        **_: Any,
    ) -> Dict[str, Any]:
        scheduler = get_scheduler()
        if scheduler is None:
            return {"error": "Scheduler not initialized"}

        # Parse date
        try:
            event_date = _parse_time(date)
        except ValueError:
            return {"error": f"无法解析日期 '{date}'"}

        # Parse optional trigger_at
        parsed_trigger_at = None
        if trigger_at:
            try:
                parsed_trigger_at = _parse_time(trigger_at)
            except ValueError:
                return {"error": f"无法解析提醒时间 '{trigger_at}'"}

        # Normalize repeat for calendar
        repeat_lower = repeat.strip().lower()
        if repeat_lower in ("yearly", "每年"):
            repeat_rule = f"yearly:{event_date.month:02d}-{event_date.day:02d}"
        elif repeat_lower in ("monthly", "每月"):
            repeat_rule = f"monthly:{event_date.day}"
        else:
            repeat_rule = _normalize_repeat(repeat)

        event = ScheduledEvent(
            source="calendar",
            title=title,
            description=description,
            trigger_at=parsed_trigger_at,
            recurrence_rule=repeat_rule,
            priority="normal",
            status="active",
        )

        created = scheduler.create_event(event)
        return {
            "status": "success",
            "event_id": created.id,
            "title": title,
            "date": event_date.strftime("%Y-%m-%d"),
            "trigger_at": created.trigger_at.isoformat() if created.trigger_at else None,
            "repeat": repeat_rule,
        }
