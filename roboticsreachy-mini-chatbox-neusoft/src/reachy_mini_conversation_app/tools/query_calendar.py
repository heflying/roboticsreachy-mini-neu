"""Tool for querying calendar events."""

from __future__ import annotations
from datetime import datetime, timedelta
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.tools.set_alarm import get_scheduler


def _parse_date_range(date_range: str) -> tuple[str, str]:
    """Parse a date_range string into (start, end) ISO date strings."""
    now = datetime.now()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    mapping = {
        "today": (today_start, today_start + timedelta(days=1)),
        "今天": (today_start, today_start + timedelta(days=1)),
        "tomorrow": (today_start + timedelta(days=1), today_start + timedelta(days=2)),
        "明天": (today_start + timedelta(days=1), today_start + timedelta(days=2)),
        "this_week": (today_start, today_start + timedelta(days=7)),
        "本周": (today_start, today_start + timedelta(days=7)),
        "this_month": (today_start.replace(day=1), (today_start.replace(day=1) + timedelta(days=32)).replace(day=1)),
        "本月": (today_start.replace(day=1), (today_start.replace(day=1) + timedelta(days=32)).replace(day=1)),
    }

    if date_range in mapping:
        start, end = mapping[date_range]
        return start.isoformat(), end.isoformat()

    # Try to parse as a specific date
    try:
        dt = datetime.fromisoformat(date_range)
        start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        return start.isoformat(), end.isoformat()
    except ValueError:
        pass

    # Default: today
    return today_start.isoformat(), (today_start + timedelta(days=1)).isoformat()


class QueryCalendarEventsTool(Tool):
    """Query calendar/schedule events by date range."""

    name = "query_calendar_events"
    description = (
        "查询已保存的日历/日程事件。仅当用户明确要查看日程、会议、约定时使用。"
        "不适用于询问当前时间或日期。"
        "date_range 参数：today/今天、tomorrow/明天、this_week/本周、this_month/本月，或具体日期如'2026-07-01'。"
        "返回指定范围内的日程事件列表，包含标题、描述、触发时间和重复规则。"
    )
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "date_range": {
                "type": "string",
                "description": "查询范围：today/今天（默认）、tomorrow/明天、this_week/本周、this_month/本月，或具体日期",
                "default": "today",
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, date_range: str = "today", **_: Any) -> Dict[str, Any]:
        scheduler = get_scheduler()
        if scheduler is None:
            return {"error": "Scheduler not initialized"}

        start, end = _parse_date_range(date_range)
        events = scheduler.store.query_by_date_range(start, end)

        # Also get calendar events without trigger_at (passive-only events)
        all_calendar = scheduler.store.get_active_by_source("calendar")
        passive_events = [e for e in all_calendar if e.trigger_at is None]

        results = []
        for event in events + passive_events:
            results.append({
                "id": event.id,
                "title": event.title,
                "description": event.description,
                "trigger_at": event.trigger_at.isoformat() if event.trigger_at else None,
                "repeat": event.recurrence_rule,
            })

        return {
            "status": "success",
            "date_range": date_range,
            "count": len(results),
            "events": results,
        }
