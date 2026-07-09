"""Tool for updating an explicit user profile fact."""

from __future__ import annotations
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.tools.memory_helpers import get_memory_runtime, missing_memory_runtime


class UpdateUserFactTool(Tool):
    """Update a long-term profile fact by key or id."""

    name = "update_user_fact"
    description = "Update a stored user profile fact by id, key, or matching text."
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "key_or_id": {"type": "string", "description": "Existing fact id, key, or matching text."},
            "value": {"type": "string", "description": "Replacement value."},
            "category": {"type": "string", "description": "Optional replacement category."},
            "confirmed": {"type": "boolean", "default": False},
        },
        "required": ["key_or_id", "value"],
    }

    async def __call__(
        self,
        deps: ToolDependencies,
        key_or_id: str,
        value: str,
        category: str | None = None,
        confirmed: bool = False,
        **_: Any,
    ) -> Dict[str, Any]:
        """Update the requested profile fact."""
        runtime = get_memory_runtime(deps)
        if runtime is None:
            return missing_memory_runtime()
        fact = runtime.update_user_fact(
            key_or_id=key_or_id,
            value=value,
            category=category,
            confirmed=confirmed,
        )
        if fact is None:
            return {"status": "not_found"}
        return {"status": fact.status, "fact": runtime._fact_to_dict(fact)}
