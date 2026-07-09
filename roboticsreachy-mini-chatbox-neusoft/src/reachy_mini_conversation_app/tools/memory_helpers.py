"""Helpers shared by memory tools."""

from __future__ import annotations
from typing import Any

from reachy_mini_conversation_app.memory.runtime import MemoryRuntime, get_global_memory_runtime
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


def get_memory_runtime(deps: ToolDependencies) -> MemoryRuntime | None:
    """Resolve memory runtime from tool deps or process global fallback."""
    return getattr(deps, "memory_runtime", None) or get_global_memory_runtime()


def missing_memory_runtime() -> dict[str, Any]:
    """Return a consistent tool error payload."""
    return {"error": "memory runtime is not configured"}
