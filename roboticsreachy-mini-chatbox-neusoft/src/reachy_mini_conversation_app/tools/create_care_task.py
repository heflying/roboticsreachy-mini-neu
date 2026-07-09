"""Tool for creating an elder-care task or reminder."""

from __future__ import annotations
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.tools.memory_helpers import get_memory_runtime, missing_memory_runtime


class CreateCareTaskTool(Tool):
    """Create a care reminder under MemoryRuntime control."""

    name = "create_care_task"
    description = "Create a care task/reminder. Medication and emergency tasks may remain pending until confirmed."
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "task_type": {
                "type": "string",
                "description": "reminder, hydration, medication, appointment, exercise, check_in, medical.",
                "default": "reminder",
            },
            "due_at": {"type": "string", "description": "Optional ISO datetime or human-readable due time."},
            "recurrence_rule": {"type": "string", "description": "Optional RRULE or simple recurrence label."},
            "confirmed": {"type": "boolean", "default": False},
        },
        "required": ["title"],
    }

    async def __call__(
        self,
        deps: ToolDependencies,
        title: str,
        task_type: str = "reminder",
        due_at: str | None = None,
        recurrence_rule: str | None = None,
        confirmed: bool = False,
        **_: Any,
    ) -> Dict[str, Any]:
        """Create a task."""
        runtime = get_memory_runtime(deps)
        if runtime is None:
            return missing_memory_runtime()
        task = runtime.create_care_task(
            title=title,
            task_type=task_type,
            due_at=due_at,
            recurrence_rule=recurrence_rule,
            confirmed=confirmed,
        )
        if task is None:
            return {"status": "rejected"}
        return {"status": task.status, "care_task": runtime._task_to_dict(task)}
