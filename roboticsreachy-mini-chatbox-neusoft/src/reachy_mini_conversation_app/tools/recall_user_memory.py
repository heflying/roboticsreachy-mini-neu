"""Tool for recalling active user memory."""

from __future__ import annotations
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.tools.memory_helpers import get_memory_runtime, missing_memory_runtime


class RecallUserMemoryTool(Tool):
    """Retrieve memory for answering a user's direct recall question."""

    name = "recall_user_memory"
    description = "Search active user profile facts, recent notes, and care tasks."
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Optional search text."},
            "include_pending": {
                "type": "boolean",
                "description": "Whether to include pending_confirmation memories for user review.",
                "default": False,
            },
        },
    }

    async def __call__(
        self,
        deps: ToolDependencies,
        query: str | None = None,
        include_pending: bool = False,
        **_: Any,
    ) -> Dict[str, Any]:
        """Return matching memory."""
        runtime = get_memory_runtime(deps)
        if runtime is None:
            return missing_memory_runtime()
        return {"status": "completed", "memory": runtime.recall_user_memory(query, include_pending=include_pending)}
