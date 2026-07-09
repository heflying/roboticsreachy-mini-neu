"""Tool for updating an elder-care task."""

from __future__ import annotations
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.tools.memory_helpers import get_memory_runtime, missing_memory_runtime


class UpdateCareTaskTool(Tool):
    """Update a care task by id."""

    name = "update_care_task"
    description = "Update a care task by id."
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "task_id": {"type": "string"},
            "title": {"type": "string"},
            "task_type": {"type": "string"},
            "due_at": {"type": "string"},
            "recurrence_rule": {"type": "string"},
            "status": {"type": "string", "description": "active, pending_confirmation, completed, disabled, archived."},
        },
        "required": ["task_id"],
    }

    async def __call__(self, deps: ToolDependencies, task_id: str, **kwargs: Any) -> Dict[str, Any]:
        """Update a task."""
        runtime = get_memory_runtime(deps)
        if runtime is None:
            return missing_memory_runtime()
        updates = {key: value for key, value in kwargs.items() if value is not None}
        task = runtime.update_care_task(task_id, **updates)
        if task is None:
            return {"status": "not_found"}
        return {"status": "completed", "care_task": runtime._task_to_dict(task)}
