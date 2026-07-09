"""Tool for saving an explicit user profile fact."""

from __future__ import annotations
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.tools.memory_helpers import get_memory_runtime, missing_memory_runtime


class RememberUserFactTool(Tool):
    """Persist a user-approved long-term memory candidate."""

    name = "remember_user_fact"
    description = (
        "Save an explicit long-term user fact or preference. Sensitive health, medication, contact, "
        "address, safety, legal, or financial facts may be stored as pending until confirmed."
    )
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "Stable snake_case memory key, e.g. preferred_name."},
            "value": {"type": "string", "description": "Fact value to remember."},
            "category": {
                "type": "string",
                "description": "identity, preference, family, routine, communication, health, medication, contact, care_preference.",
                "default": "preference",
            },
            "confirmed": {
                "type": "boolean",
                "description": "True only when the user or caregiver has explicitly confirmed saving sensitive information.",
                "default": False,
            },
        },
        "required": ["key", "value"],
    }

    async def __call__(
        self,
        deps: ToolDependencies,
        key: str,
        value: str,
        category: str = "preference",
        confirmed: bool = False,
        **_: Any,
    ) -> Dict[str, Any]:
        """Save the requested profile fact."""
        runtime = get_memory_runtime(deps)
        if runtime is None:
            return missing_memory_runtime()
        fact = runtime.remember_user_fact(
            key=key,
            value=value,
            category=category,
            source="tool",
            confirmed=confirmed,
        )
        if fact is None:
            return {"status": "rejected"}
        return {"status": fact.status, "fact": runtime._fact_to_dict(fact)}
