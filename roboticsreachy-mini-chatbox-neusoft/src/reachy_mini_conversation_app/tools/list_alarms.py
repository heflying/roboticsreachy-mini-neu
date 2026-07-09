"""Tool for listing all active alarms."""

from __future__ import annotations
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.tools.set_alarm import get_scheduler


class ListAlarmsTool(Tool):
    """List all active alarms."""

    name = "list_alarms"
    description = "查询当前所有活跃的闹钟列表"
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **_: Any) -> Dict[str, Any]:
        scheduler = get_scheduler()
        if scheduler is None:
            return {"error": "Scheduler not initialized"}

        events = scheduler.store.get_active_by_source("alarm")
        alarms = []
        for event in events:
            alarms.append({
                "id": event.id,
                "message": event.title,
                "trigger_at": event.trigger_at.isoformat() if event.trigger_at else None,
                "repeat": event.recurrence_rule,
                "priority": event.priority,
            })

        return {
            "status": "success",
            "count": len(alarms),
            "alarms": alarms,
        }
