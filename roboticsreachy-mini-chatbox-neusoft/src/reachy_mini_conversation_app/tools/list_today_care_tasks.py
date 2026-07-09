"""Tool for listing active care tasks."""

from __future__ import annotations
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.tools.memory_helpers import get_memory_runtime, missing_memory_runtime


class ListTodayCareTasksTool(Tool):
    """List active care tasks for today's conversation."""

    name = "list_today_care_tasks"
    description = "List active care tasks and reminders for the current user."
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "include_completed": {"type": "boolean", "default": False},
        },
    }

    async def __call__(self, deps: ToolDependencies, include_completed: bool = False, **_: Any) -> Dict[str, Any]:
        """Return care tasks."""
        runtime = get_memory_runtime(deps)
        if runtime is None:
            return missing_memory_runtime()
        return {"status": "completed", "care_tasks": runtime.list_today_care_tasks(include_completed=include_completed)}
