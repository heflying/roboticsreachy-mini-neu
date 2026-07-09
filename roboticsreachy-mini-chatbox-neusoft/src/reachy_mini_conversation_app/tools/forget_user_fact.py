"""Tool for deleting or archiving user memory."""

from __future__ import annotations
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.tools.memory_helpers import get_memory_runtime, missing_memory_runtime


class ForgetUserFactTool(Tool):
    """Archive profile facts or notes at the user's request."""

    name = "forget_user_fact"
    description = "Forget/delete stored user profile facts or middle-term notes matching a query."
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Fact id, key, value, or note text to forget."},
        },
        "required": ["query"],
    }

    async def __call__(self, deps: ToolDependencies, query: str, **_: Any) -> Dict[str, Any]:
        """Archive matching facts and notes."""
        runtime = get_memory_runtime(deps)
        if runtime is None:
            return missing_memory_runtime()
        facts = runtime.forget_user_fact(query)
        notes = runtime.store.delete_memory_note(runtime.user.id, query)
        return {
            "status": "completed",
            "archived_facts": len(facts),
            "archived_notes": len(notes),
        }
