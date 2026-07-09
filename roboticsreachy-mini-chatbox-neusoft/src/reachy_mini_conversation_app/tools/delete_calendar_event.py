"""Tool for deleting a calendar event."""

from __future__ import annotations
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.tools.set_alarm import get_scheduler


class DeleteCalendarEventTool(Tool):
    """Delete a calendar event by ID."""

    name = "delete_calendar_event"
    description = "删除指定的日历事件。需要提供事件ID（从 query_calendar_events 获取）"
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "event_id": {
                "type": "string",
                "description": "要删除的日历事件ID",
            },
        },
        "required": ["event_id"],
    }

    async def __call__(self, deps: ToolDependencies, event_id: str, **_: Any) -> Dict[str, Any]:
        scheduler = get_scheduler()
        if scheduler is None:
            return {"error": "Scheduler not initialized"}

        success = scheduler.cancel_event(event_id)
        if success:
            return {"status": "success", "message": f"日历事件 {event_id} 已删除"}
        else:
            return {"status": "error", "message": f"未找到日历事件 {event_id}"}
