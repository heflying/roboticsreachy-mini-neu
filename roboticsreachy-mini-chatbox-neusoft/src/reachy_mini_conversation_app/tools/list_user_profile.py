"""Tool for listing the user's stored profile."""

from __future__ import annotations
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.tools.memory_helpers import get_memory_runtime, missing_memory_runtime


class ListUserProfileTool(Tool):
    """List active or pending user profile memory."""

    name = "list_user_profile"
    description = "List stored user profile facts, optionally including facts pending confirmation."
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "include_pending": {"type": "boolean", "default": False},
        },
    }

    async def __call__(self, deps: ToolDependencies, include_pending: bool = False, **_: Any) -> Dict[str, Any]:
        """Return profile facts."""
        runtime = get_memory_runtime(deps)
        if runtime is None:
            return missing_memory_runtime()
        return {"status": "completed", "profile": runtime.list_user_profile(include_pending=include_pending)}
