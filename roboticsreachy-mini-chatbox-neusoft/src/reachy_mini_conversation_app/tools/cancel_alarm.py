"""Tool for cancelling an alarm."""

from __future__ import annotations
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.tools.set_alarm import get_scheduler


class CancelAlarmTool(Tool):
    """Cancel an alarm by ID."""

    name = "cancel_alarm"
    description = "取消指定的闹钟。需要提供闹钟ID（从 list_alarms 获取）"
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "alarm_id": {
                "type": "string",
                "description": "要取消的闹钟ID",
            },
        },
        "required": ["alarm_id"],
    }

    async def __call__(self, deps: ToolDependencies, alarm_id: str, **_: Any) -> Dict[str, Any]:
        scheduler = get_scheduler()
        if scheduler is None:
            return {"error": "Scheduler not initialized"}

        success = scheduler.cancel_event(alarm_id)
        if success:
            return {"status": "success", "message": f"闹钟 {alarm_id} 已取消"}
        else:
            return {"status": "error", "message": f"未找到闹钟 {alarm_id}"}
