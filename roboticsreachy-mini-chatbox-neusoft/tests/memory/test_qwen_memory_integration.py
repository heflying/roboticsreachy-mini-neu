from unittest.mock import MagicMock

import pytest

import reachy_mini_conversation_app.qwen_omni_realtime as qwen_mod
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.core_tools import CORE_MEMORY_TOOL_NAMES, ToolDependencies, get_tool_specs
from reachy_mini_conversation_app.qwen_omni_realtime import QwenOmniRealtimeHandler


def _build_handler(tmp_path, monkeypatch):
    monkeypatch.setenv("REACHY_MINI_MEMORY_DB_PATH", str(tmp_path / "memory.db"))
    monkeypatch.setenv("REACHY_MINI_MEMORY_EXTRACTOR", "none")
    monkeypatch.setenv("REACHY_MINI_MEMORY_WRITE_MODE", "extractor_only")
    monkeypatch.setattr(config, "QWEN_TOOL_MODE", "router")
    monkeypatch.setattr(config, "MODEL_NAME", "qwen3.5-omni-flash-realtime")
    monkeypatch.setattr(qwen_mod, "get_session_instructions", lambda: "test instructions")
    monkeypatch.setattr(qwen_mod, "get_session_voice", lambda: "Cherry")
    movement_manager = MagicMock()
    movement_manager.is_idle.return_value = False
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=movement_manager)
    return QwenOmniRealtimeHandler(deps)


def test_qwen_session_instructions_include_active_memory(tmp_path, monkeypatch):
    """Qwen session instructions include active memory context."""
    handler = _build_handler(tmp_path, monkeypatch)
    handler.memory_runtime.remember_user_fact(
        key="preferred_name",
        value="张老师",
        category="identity",
        confirmed=True,
    )

    update = handler._build_session_update()

    assert "张老师" in update["session"]["instructions"]
    assert "Memory context" in update["session"]["instructions"]


def test_extractor_only_mode_hides_memory_tools(monkeypatch):
    """Native tool specs should not expose memory-write tools in extractor-only mode."""
    monkeypatch.setenv("REACHY_MINI_MEMORY_WRITE_MODE", "extractor_only")

    tool_names = {spec.get("name") for spec in get_tool_specs()}

    assert tool_names.isdisjoint(CORE_MEMORY_TOOL_NAMES)


@pytest.mark.asyncio
async def test_qwen_records_transcripts_without_router_memory_writes(tmp_path, monkeypatch):
    """Extractor-only mode records transcripts but does not write memory via router."""
    handler = _build_handler(tmp_path, monkeypatch)

    await handler._handle_message(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "以后叫我王阿姨",
        }
    )
    await handler._handle_message(
        {
            "type": "response.audio_transcript.done",
            "transcript": "好的，王阿姨。",
        }
    )

    assert handler.memory_runtime.list_user_profile() == []
    turns = handler.memory_runtime.store.get_turns(handler.memory_runtime.current_session_id)
    assert [turn.role for turn in turns] == ["user", "assistant"]
