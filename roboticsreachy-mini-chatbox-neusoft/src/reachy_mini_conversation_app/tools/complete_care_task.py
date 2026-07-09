"""Tool for completing an elder-care task."""

from __future__ import annotations
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.tools.memory_helpers import get_memory_runtime, missing_memory_runtime


class CompleteCareTaskTool(Tool):
    """Mark a care task as completed."""

    name = "complete_care_task"
    description = "Mark a care task completed by id or title query."
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "task_id": {"type": "string"},
            "query": {"type": "string"},
        },
    }

    async def __call__(
        self,
        deps: ToolDependencies,
        task_id: str | None = None,
        query: str | None = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """Complete a task."""
        runtime = get_memory_runtime(deps)
        if runtime is None:
            return missing_memory_runtime()
        task = runtime.complete_care_task(task_id=task_id, query=query)
        if task is None:
            return {"status": "not_found"}
        return {"status": "completed", "care_task": runtime._task_to_dict(task)}
