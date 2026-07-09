import json
import base64
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastrtc import AdditionalOutputs

from tests.fakes.qwen_realtime_server import FakeQwenWebSocket, FakeQwenConnectContext
import reachy_mini_conversation_app.qwen_omni_realtime as qwen_mod
from reachy_mini_conversation_app.config import config, get_backend_choice, get_model_name_for_backend
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.qwen_omni_realtime import QwenOmniRealtimeHandler, _resolve_qwen_voice


def _build_handler(*, gradio_mode: bool = False) -> QwenOmniRealtimeHandler:
    movement_manager = MagicMock()
    movement_manager.is_idle.return_value = False
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=movement_manager,
        head_wobbler=MagicMock(),
    )
    return QwenOmniRealtimeHandler(deps, gradio_mode=gradio_mode)


def test_qwen_backend_choice_and_default_model() -> None:
    """Qwen aliases and model-name fallback should select the Qwen backend."""
    assert get_model_name_for_backend("qwen_omni") == "qwen3.5-omni-flash-realtime"
    assert get_backend_choice("qwen3.5-omni-plus-realtime") == "qwen_omni"
    assert get_backend_choice("qwen3.5-omni-flash-realtime") == "qwen_omni"


def test_qwen_session_instructions_always_include_memory_policy() -> None:
    """Realtime instructions should include memory-use guardrails even without context."""
    instructions = qwen_mod._qwen_session_instructions("")

    assert "[记忆使用规则]" in instructions
    assert "没有已确认或仍有效的记录" in instructions


@pytest.fixture(autouse=True)
def _qwen_config(monkeypatch):
    monkeypatch.setattr(config, "BACKEND_PROVIDER", "qwen_omni")
    monkeypatch.setattr(config, "MODEL_NAME", "qwen3.5-omni-flash-realtime")
    monkeypatch.setattr(config, "DASHSCOPE_API_KEY", "fake-key")
    monkeypatch.setattr(config, "QWEN_REALTIME_URL", "ws://127.0.0.1:8765")
    monkeypatch.setattr(config, "QWEN_REALTIME_VOICE", None)
    monkeypatch.setattr(config, "QWEN_TOOL_MODE", "router")
    monkeypatch.setenv("REACHY_MINI_MEMORY_WRITE_MODE", "extractor_only")
    monkeypatch.setattr(qwen_mod, "get_session_instructions", lambda: "test instructions")
    monkeypatch.setattr(qwen_mod, "get_session_voice", lambda: "Cherry")
    monkeypatch.setattr(qwen_mod, "get_tool_specs", lambda: [])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    ["qwen3.5-omni-flash-realtime", "qwen3.5-omni-plus-realtime"],
)
async def test_qwen_sends_session_update_for_flash_and_plus(monkeypatch, model_name) -> None:
    """Flash and Plus use the same handler/session shape."""
    monkeypatch.setattr(config, "MODEL_NAME", model_name)
    websocket = FakeQwenWebSocket()
    monkeypatch.setattr(qwen_mod.websockets, "connect", lambda *_a, **_kw: FakeQwenConnectContext(websocket))

    handler = _build_handler()
    object.__setattr__(handler.tool_manager, "start_up", MagicMock())
    object.__setattr__(handler.tool_manager, "shutdown", AsyncMock())

    task = asyncio.create_task(handler._run_realtime_session("fake-key"))
    await asyncio.wait_for(handler._connected_event.wait(), timeout=1.0)

    assert websocket.sent[0]["type"] == "session.update"
    session = websocket.sent[0]["session"]
    assert session["modalities"] == ["text", "audio"]
    assert "始终使用中文" in session["instructions"]
    assert session["voice"] == "Tina"
    assert session["input_audio_format"] == "pcm"
    assert session["output_audio_format"] == "pcm"
    assert session["turn_detection"]["type"] == "server_vad"
    assert session["turn_detection"]["interrupt_response"] is True
    assert session["input_audio_transcription"] == {"model": "gummy-realtime-v1"}
    assert "tools" not in session

    handler._stop_event.set()
    await websocket.incoming.put({"type": "noop"})
    await asyncio.wait_for(task, timeout=1.0)


def test_qwen35_rejects_profile_cherry_and_uses_tina(monkeypatch) -> None:
    """Qwen3.5 models should not inherit the older Qwen3 Cherry voice."""
    monkeypatch.setattr(config, "MODEL_NAME", "qwen3.5-omni-flash-realtime")
    monkeypatch.setattr(config, "QWEN_REALTIME_VOICE", None)

    assert _resolve_qwen_voice("Cherry") == "Tina"


def test_qwen35_explicit_invalid_voice_falls_back_to_tina(monkeypatch) -> None:
    """Unsupported explicit Qwen3.5 voice overrides should be made safe."""
    monkeypatch.setattr(config, "MODEL_NAME", "qwen3.5-omni-plus-realtime")
    monkeypatch.setattr(config, "QWEN_REALTIME_VOICE", "Cherry")

    assert _resolve_qwen_voice(None) == "Tina"


@pytest.mark.asyncio
async def test_qwen_receive_resamples_and_sends_pcm_base64() -> None:
    """Microphone frames are converted to 16 kHz PCM16 base64 messages."""
    handler = _build_handler()
    websocket = FakeQwenWebSocket()
    handler.websocket = websocket

    audio = np.zeros((48000, 1), dtype=np.float32)
    await handler.receive((48000, audio))

    sent = websocket.sent[0]
    assert sent["type"] == "input_audio_buffer.append"
    pcm = base64.b64decode(sent["audio"])
    assert len(pcm) == 16000 * 2


@pytest.mark.asyncio
async def test_qwen_interruption_clears_playback_and_sets_listening() -> None:
    """Server VAD speech-start events should barge in locally."""
    handler = _build_handler()
    clear_queue = MagicMock()
    object.__setattr__(handler, "_clear_queue", clear_queue)

    await handler._handle_message({"type": "input_audio_buffer.speech_started"})
    await handler._handle_message({"type": "input_audio_buffer.speech_stopped"})

    clear_queue.assert_called_once()
    handler.deps.head_wobbler.reset.assert_called_once()
    handler.deps.movement_manager.set_listening.assert_any_call(True)
    handler.deps.movement_manager.set_listening.assert_any_call(False)


@pytest.mark.asyncio
async def test_qwen_audio_delta_emits_audio_and_feeds_head_wobbler() -> None:
    """Qwen audio deltas should drive playback and speech wobble."""
    handler = _build_handler(gradio_mode=True)
    audio_bytes = b"\x00\x00\x10\x00" * 128

    await handler._handle_message(
        {"type": "response.audio.delta", "delta": base64.b64encode(audio_bytes).decode("ascii")}
    )
    await handler._handle_message({"type": "response.audio.done"})

    output = handler.output_queue.get_nowait()
    assert isinstance(output, tuple)
    assert output[0] == 24000
    assert output[1].dtype == np.int16
    handler.deps.head_wobbler.feed.assert_called_once()
    handler.deps.head_wobbler.request_reset_after_current_audio.assert_called_once()


@pytest.mark.asyncio
async def test_qwen_router_fallback_starts_robot_tool_for_transcript() -> None:
    """Fallback routing should preserve robot tool calls without native tools."""
    handler = _build_handler()
    start_tool = AsyncMock(return_value=SimpleNamespace(tool_id="dance-router-1"))
    object.__setattr__(handler.tool_manager, "start_tool", start_tool)

    await handler._handle_message(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "请跳舞",
        }
    )

    start_tool.assert_awaited_once()
    routine = start_tool.call_args.kwargs["tool_call_routine"]
    assert routine.tool_name == "dance"
    user_output = handler.output_queue.get_nowait()
    tool_output = handler.output_queue.get_nowait()
    assert isinstance(user_output, AdditionalOutputs)
    assert isinstance(tool_output, AdditionalOutputs)
    assert "fallback tool dance" in tool_output.args[0]["content"]


@pytest.mark.asyncio
async def test_qwen_router_fallback_starts_move_head_for_left_turn() -> None:
    """Chinese head-turn commands should map to the local move_head tool."""
    handler = _build_handler()
    start_tool = AsyncMock(return_value=SimpleNamespace(tool_id="move-head-router-1"))
    object.__setattr__(handler.tool_manager, "start_tool", start_tool)

    await handler._handle_message(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "\u8bf7\u628a\u5934\u8f6c\u5411\u5de6\u8fb9",
        }
    )

    start_tool.assert_awaited_once()
    routine = start_tool.call_args.kwargs["tool_call_routine"]
    assert routine.tool_name == "move_head"
    assert json.loads(routine.args_json_str) == {"direction": "left"}


@pytest.mark.asyncio
async def test_qwen_router_fallback_treats_left_bend_as_head_turn() -> None:
    """Common Chinese turn wording should map to the local move_head tool."""
    handler = _build_handler()
    start_tool = AsyncMock(return_value=SimpleNamespace(tool_id="move-head-router-2"))
    object.__setattr__(handler.tool_manager, "start_tool", start_tool)

    await handler._handle_message(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "\u8bf7\u5411\u5de6\u62d0",
        }
    )

    start_tool.assert_awaited_once()
    routine = start_tool.call_args.kwargs["tool_call_routine"]
    assert routine.tool_name == "move_head"
    assert json.loads(routine.args_json_str) == {"direction": "left"}


@pytest.mark.asyncio
async def test_qwen_native_tool_mode_includes_tool_specs(monkeypatch) -> None:
    """Native mode can be enabled once provider-side custom tools are confirmed."""
    monkeypatch.setattr(config, "QWEN_TOOL_MODE", "native")
    monkeypatch.setattr(
        qwen_mod,
        "get_tool_specs",
        lambda: [{"type": "function", "name": "dance", "description": "Dance", "parameters": {"type": "object"}}],
    )

    handler = _build_handler()
    update = handler._build_session_update()

    session = update["session"]
    assert session["tool_choice"] == "auto"
    assert session["tools"][0]["name"] == "dance"
