"""Runtime switches for memory write orchestration."""

from __future__ import annotations
import os


EXTRACTOR_ONLY_MODES = {"extractor_only", "extractor-only", "session_end", "session-end"}
DEFAULT_MEMORY_WRITE_MODE = "extractor_only"


def memory_write_mode() -> str:
    """Return the configured memory write mode."""
    return os.getenv("REACHY_MINI_MEMORY_WRITE_MODE", DEFAULT_MEMORY_WRITE_MODE).strip().lower()


def memory_command_writes_enabled() -> bool:
    """Return whether router/native tool commands may write memory directly."""
    return memory_write_mode() not in EXTRACTOR_ONLY_MODES
