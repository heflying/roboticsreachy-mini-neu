"""Qwen Omni Realtime handler with a local tool-router fallback.

The handler mirrors the OpenAI/Gemini stream contract used by the app while
keeping Qwen-specific protocol details isolated.  In production it connects to
DashScope's realtime WebSocket endpoint.  In tests, ``QWEN_REALTIME_URL`` can
point at a local fake server so the audio, interruption, and tool fallback
paths are testable without a real API key.
"""

from __future__ import annotations
import json
import uuid
import base64
import random
import asyncio
import logging
from typing import Any, Final, Tuple, Literal, Optional
from datetime import datetime
from urllib.parse import urlencode

import numpy as np
import websockets
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item, audio_to_int16
from numpy.typing import NDArray
from scipy.signal import resample

from reachy_mini_conversation_app.config import QWEN_AVAILABLE_VOICES, config
from reachy_mini_conversation_app.timing import tracker as latency_tracker
from reachy_mini_conversation_app.prompts import get_session_voice, get_session_instructions
from reachy_mini_conversation_app.memory.policy import memory_write_mode, memory_command_writes_enabled
from reachy_mini_conversation_app.memory.runtime import create_default_memory_runtime
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies, get_tool_specs
from reachy_mini_conversation_app.memory.command_router import MemoryCommandRouter
from reachy_mini_conversation_app.tools.background_tool_manager import (
    ToolCallRoutine,
    ToolNotification,
    BackgroundToolManager,
)


logger = logging.getLogger(__name__)

QWEN_INPUT_SAMPLE_RATE: Final[int] = 16000
QWEN_OUTPUT_SAMPLE_RATE: Final[int] = 24000
DEFAULT_QWEN_REALTIME_ENDPOINT: Final[str] = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
DEFAULT_QWEN_TOOL_MODE: Final[Literal["router", "native"]] = "router"
QWEN_35_DEFAULT_VOICE: Final[str] = "Tina"
QWEN_3_DEFAULT_VOICE: Final[str] = "Cherry"
QWEN_35_VOICES: Final[set[str]] = {"Tina", "Cindy", "Liora", "Mira", "Sunnybobi", "Raymond"}
QWEN_CHINESE_INSTRUCTION_SUFFIX: Final[str] = (
    "\n\n请始终使用中文和用户对话，除非用户明确要求使用其他语言。"
    "当用户要求你跳舞、转头、看向某个方向或执行机器人动作时，先简短确认并保持中文表达。"
)
QWEN_MEMORY_POLICY_SUFFIX: Final[str] = (
    "\n\n[记忆使用规则] 最高优先级\n"
    "- Memory context 是本轮对话可用的已确认背景记忆，优先级高于默认人设、寒暄模板和泛化回答。\n"
    "- 当用户直接询问“你记得/还记得/怎么称呼我/我喜欢什么/提醒什么”时，必须先查 Memory context，并使用其中"
    " active/confirmed 信息回答。\n"
    "- 如果 Memory context 明确包含相关 active/confirmed 信息，禁止回答“我还没记住”“还没有确认”“需要你告诉我”。\n"
    "- 如果用户问某个提醒、画像、住址、用药或健康信息是否已确认/是否仍有效，而 active memory context 没有明确列出，"
    "回答“我没有已确认或仍有效的记录”，不要猜测它仍有效。\n"
    "- pending_confirmation、待确认、已取消、已完成、已删除的信息不能当作事实，也不能作为继续提醒或直接使用的依据。"
)


def _default_qwen_voice_for_model(model_name: str | None = None) -> str:
    """Return a safe default voice for the configured Qwen model family."""
    candidate = (model_name or config.MODEL_NAME or "").strip().lower()
    return QWEN_35_DEFAULT_VOICE if candidate.startswith("qwen3.5") else QWEN_3_DEFAULT_VOICE


def _supported_qwen_voices_for_model(model_name: str | None = None) -> set[str]:
    """Return the curated voice set for the configured Qwen model family."""
    candidate = (model_name or config.MODEL_NAME or "").strip().lower()
    if candidate.startswith("qwen3.5"):
        return set(QWEN_35_VOICES)
    return set(QWEN_AVAILABLE_VOICES)


def _resolve_qwen_voice(profile_voice: str | None) -> str:
    """Resolve a profile voice to one supported by Qwen Omni Realtime."""
    supported = _supported_qwen_voices_for_model()
    default_voice = _default_qwen_voice_for_model()
    configured = (config.QWEN_REALTIME_VOICE or "").strip()
    if configured:
        if configured in supported:
            return configured
        logger.warning(
            "QWEN_REALTIME_VOICE=%r is not supported by model %r; falling back to %r",
            configured,
            config.MODEL_NAME,
            default_voice,
        )
        return default_voice

    voice_map = {v.lower(): v for v in QWEN_AVAILABLE_VOICES}
    if profile_voice:
        resolved = voice_map.get(profile_voice.lower())
        if resolved and resolved in supported:
            return resolved
    return default_voice


def _build_qwen_realtime_url() -> str:
    """Build the Qwen realtime WebSocket URL, allowing test override."""
    override = (config.QWEN_REALTIME_URL or "").strip()
    if override:
        return override

    query = urlencode({"model": config.MODEL_NAME})
    return f"{DEFAULT_QWEN_REALTIME_ENDPOINT}?{query}"


def _json_dumps(payload: dict[str, Any]) -> str:
    """Serialize protocol payloads compactly with stable UTF-8 behavior."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _qwen_session_instructions(memory_context: str | None = None) -> str:
    """Return instructions adapted for the Qwen realtime conversation path."""
    instructions = get_session_instructions()
    if "始终使用中文" not in instructions:
        instructions = f"{instructions}{QWEN_CHINESE_INSTRUCTION_SUFFIX}"
    if memory_context:
        instructions = f"{instructions}\n\n{memory_context}"
    if "[记忆使用规则" not in instructions:
        instructions = f"{instructions}{QWEN_MEMORY_POLICY_SUFFIX}"
    return instructions


class LocalToolRouter:
    """Small deterministic fallback router for robot tools.

    This router is intentionally conservative.  It only triggers tools from
    explicit user utterances, so it can preserve core robot actions when a
    realtime provider does not expose OpenAI-compatible custom tool calling.
    """

    def route(self, transcript: str) -> tuple[str, dict[str, Any]] | None:
        """Return a tool call using UTF-8-safe Chinese and English triggers."""
        text = transcript.strip().lower()
        if not text:
            return None

        # Chinese tokens are escaped to keep this file robust across Windows code pages.
        if any(
            token in text
            for token in ("\u505c\u6b62\u8df3\u821e", "\u505c\u4e0b\u8df3\u821e", "stop dance", "stop dancing")
        ):
            return "stop_dance", {}
        if any(token in text for token in ("\u8df3\u821e", "dance", "\u8df3\u4e00\u6bb5\u821e")):
            return "dance", {}
        if any(token in text for token in ("\u505c\u6b62\u8868\u60c5", "stop emotion")):
            return "stop_emotion", {}
        if any(
            token in text
            for token in ("\u8868\u60c5", "\u5f00\u5fc3", "\u96be\u8fc7", "\u751f\u6c14", "emotion")
        ):
            return "play_emotion", {}
        if any(
            token in text
            for token in ("\u62cd\u7167", "\u770b\u4e00\u4e0b", "\u770b\u5230\u4ec0\u4e48", "camera", "picture", "photo")
        ):
            return "camera", {"question": transcript}
        if any(token in text for token in ("\u8ddf\u8e2a", "\u8ffd\u8e2a", "head tracking", "face tracking")):
            enabled = not any(token in text for token in ("\u505c\u6b62", "\u5173\u95ed", "disable", "off"))
            return "head_tracking", {"start": enabled}

        direction: str | None = None
        if any(token in text for token in ("\u5de6", "left")):
            direction = "left"
        elif any(token in text for token in ("\u53f3", "right")):
            direction = "right"
        elif any(token in text for token in ("\u4e0a", "up")):
            direction = "up"
        elif any(token in text for token in ("\u4e0b", "down")):
            direction = "down"
        elif any(token in text for token in ("\u6b63\u9762", "front", "\u4e2d\u95f4", "center")):
            direction = "front"

        if direction and any(token in text for token in ("\u5934", "head", "\u770b", "\u8f6c", "\u62d0")):
            return "move_head", {"direction": direction}

        return None


class QwenOmniRealtimeHandler(AsyncStreamHandler):
    """Qwen Omni Realtime handler for fastrtc Stream."""

    def __init__(self, deps: ToolDependencies, gradio_mode: bool = False, instance_path: Optional[str] = None):
        """Initialize the handler."""
        super().__init__(
            expected_layout="mono",
            output_sample_rate=QWEN_OUTPUT_SAMPLE_RATE,
            input_sample_rate=QWEN_INPUT_SAMPLE_RATE,
        )
        self.deps = deps
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path
        self.websocket: Any = None
        self.output_queue: "asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs]" = asyncio.Queue()
        self.last_activity_time = asyncio.get_event_loop().time()
        self.start_time = asyncio.get_event_loop().time()
        self.is_idle_tool_call = False
        self._voice_override: str | None = None
        self._key_source: Literal["env", "textbox"] = "env"
        self._provided_api_key: str | None = None
        self._connected_event: asyncio.Event = asyncio.Event()
        self._stop_event: asyncio.Event = asyncio.Event()
        self._pending_user_transcript_chunks: list[str] = []
        self._pending_assistant_transcript_chunks: list[str] = []
        self._routed_transcripts: set[str] = set()
        self._memory_routed_transcripts: set[str] = set()
        self._listening_state = False
        self._event_log_counts: dict[str, int] = {}
        self.tool_manager = BackgroundToolManager()
        self.tool_router = LocalToolRouter()
        self.memory_runtime = create_default_memory_runtime(instance_path=instance_path)
        self.memory_command_router = MemoryCommandRouter(self.memory_runtime)
        self.deps.memory_runtime = self.memory_runtime

    def copy(self) -> "QwenOmniRealtimeHandler":
        """Create a copy of the handler."""
        return QwenOmniRealtimeHandler(self.deps, self.gradio_mode, self.instance_path)

    def _tool_mode(self) -> Literal["router", "native"]:
        mode = (config.QWEN_TOOL_MODE or DEFAULT_QWEN_TOOL_MODE).strip().lower()
        return "native" if mode == "native" else "router"

    def _set_listening_state(self, listening: bool) -> None:
        if self._listening_state == listening:
            return
        self._listening_state = listening
        self.deps.movement_manager.set_listening(listening)

    def get_current_voice(self) -> str:
        """Return the Qwen voice currently selected for this handler."""
        return _resolve_qwen_voice(self._voice_override or get_session_voice())

    async def change_voice(self, voice: str) -> str:
        """Change voice and restart the session if connected."""
        self._voice_override = voice
        if self.websocket is not None:
            await self._restart_session()
            return f"Voice changed to {voice}."
        return "Voice changed. Will take effect on next connection."

    async def apply_personality(self, profile: str | None) -> str:
        """Apply a new personality by restarting the Qwen session."""
        try:
            from reachy_mini_conversation_app.config import set_custom_profile

            set_custom_profile(profile)
            self._voice_override = None
            _ = get_session_instructions()
            _ = get_session_voice()
            if self.websocket is not None:
                await self._restart_session()
                return "Applied personality and restarted Qwen session."
            return "Applied personality. Will take effect on next connection."
        except Exception as e:
            logger.error("Error applying personality '%s': %s", profile, e)
            return f"Failed to apply personality: {e}"

    def _build_session_update(self) -> dict[str, Any]:
        """Build the provider session configuration."""
        memory_context = self.memory_runtime.build_memory_context()
        session: dict[str, Any] = {
            "modalities": ["text", "audio"],
            "instructions": _qwen_session_instructions(memory_context),
            "voice": self.get_current_voice(),
            "input_audio_format": "pcm",
            "output_audio_format": "pcm",
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.1,
                "prefix_padding_ms": 500,
                "silence_duration_ms": 500,
                "create_response": True,
                "interrupt_response": True,
            },
            "input_audio_transcription": {"model": "gummy-realtime-v1"},
        }

        if self._tool_mode() == "native":
            session["tools"] = get_tool_specs()
            session["tool_choice"] = "auto"

        logger.info(
            "Qwen Omni Realtime config: model=%r voice=%r tool_mode=%s memory_write_mode=%s",
            config.MODEL_NAME,
            session["voice"],
            self._tool_mode(),
            memory_write_mode(),
        )
        return {"type": "session.update", "session": session}

    def _connect_kwargs(self, api_key: str) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "X-DashScope-Beta": "realtime-v1",
        }
        return {"additional_headers": headers}

    async def _send_json(self, payload: dict[str, Any]) -> None:
        if self.websocket is None:
            return
        payload.setdefault("event_id", f"event_{uuid.uuid4().hex}")
        await self.websocket.send(_json_dumps(payload))

    async def start_up(self) -> None:
        """Start the Qwen handler with retries on unexpected closure."""
        api_key = config.DASHSCOPE_API_KEY
        if self.gradio_mode and not api_key:
            await self.wait_for_args()  # type: ignore[no-untyped-call]
            args = list(self.latest_args)
            textbox_api_key = args[3] if len(args) > 3 and len(args[3]) > 0 else None
            if textbox_api_key is not None:
                api_key = textbox_api_key
                self._key_source = "textbox"
                self._provided_api_key = textbox_api_key
            else:
                api_key = config.DASHSCOPE_API_KEY
        elif not api_key or not api_key.strip():
            logger.warning("DASHSCOPE_API_KEY missing. Proceeding with a placeholder (tests/offline).")
            api_key = "DUMMY"

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                await self._run_realtime_session(api_key)
                return
            except websockets.exceptions.ConnectionClosedOK:
                if self._stop_event.is_set():
                    logger.info("Qwen realtime session closed normally during shutdown.")
                    return
                logger.warning("Qwen realtime session closed normally by server; not retrying.")
                return
            except Exception as e:
                error_msg = str(e)
                if self._stop_event.is_set():
                    logger.info("Qwen realtime session stopped during shutdown: %s", error_msg)
                    return
                logger.warning(
                    "Qwen realtime session closed unexpectedly (attempt %d/%d): %s",
                    attempt,
                    max_attempts,
                    error_msg,
                )
                # Mark reconnection for fault monitoring (only after first failure)
                if attempt > 1:
                    latency_tracker.mark_reconnect(error_msg)
                if attempt < max_attempts:
                    await asyncio.sleep((2 ** (attempt - 1)) + random.uniform(0, 0.5))
                    continue
                raise
            finally:
                self.websocket = None
                self._connected_event.clear()

    async def _restart_session(self) -> None:
        if self.websocket is not None:
            try:
                await self.websocket.close()
            except Exception:
                pass
        self.websocket = None
        self._stop_event.set()
        await asyncio.sleep(0.1)
        self._stop_event.clear()
        asyncio.create_task(self.start_up(), name="qwen-omni-restart")

    async def _connect(self, url: str, api_key: str) -> Any:
        kwargs = self._connect_kwargs(api_key)
        try:
            return websockets.connect(url, **kwargs)
        except TypeError:
            headers = kwargs["additional_headers"]
            return websockets.connect(url, extra_headers=headers)

    async def _run_realtime_session(self, api_key: str) -> None:
        url = _build_qwen_realtime_url()
        # Level 2: R1 WebSocket connection
        latency_tracker.mark_R1_ws_connect_start()
        connect_context = await self._connect(url, api_key)
        async with connect_context as websocket:
            latency_tracker.mark_R1_ws_connect_done()
            self.websocket = websocket
            self._event_log_counts.clear()
            self._connected_event.set()
            self.memory_runtime.start_session(
                {
                    "model": config.MODEL_NAME,
                    "voice": self.get_current_voice(),
                    "tool_mode": self._tool_mode(),
                }
            )
            # Level 2: R2 Session configuration
            latency_tracker.mark_R2_session_config_sent()
            await self._send_json(self._build_session_update())
            logger.info("Qwen Omni Realtime session connected successfully")

            self.tool_manager.start_up(tool_callbacks=[self._handle_tool_result])
            try:
                while not self._stop_event.is_set():
                    raw = await websocket.recv()
                    message = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode("utf-8"))
                    self._log_event_shape(message)
                    await self._handle_message(message)
            finally:
                await self.tool_manager.shutdown()
                self.memory_runtime.end_session_background(reason="realtime_session_closed")

    def _log_event_shape(self, message: dict[str, Any]) -> None:
        """Log a bounded summary of incoming Qwen events without dumping audio."""
        event_type = str(message.get("type") or "<missing>")
        if event_type in {
            "response.audio.delta",
            "response.output_audio.delta",
            "input_audio_buffer.append",
        }:
            return
        count = self._event_log_counts.get(event_type, 0)
        if count >= 3:
            return
        self._event_log_counts[event_type] = count + 1
        logger.info("Qwen event shape: type=%s keys=%s", event_type, sorted(message.keys()))

    async def _flush_transcript_chunks(self, role: str, chunks: list[str]) -> str | None:
        if not chunks:
            return None
        transcript = "".join(chunks).strip()
        chunks.clear()
        if not transcript:
            return None
        await self.output_queue.put(AdditionalOutputs({"role": role, "content": transcript}))
        if role == "assistant":
            self.memory_runtime.record_assistant_transcript(
                transcript,
                metadata={"source": "qwen_realtime", "event": "assistant_transcript_done"},
            )
        return transcript

    async def _handle_user_transcript(self, transcript: str) -> None:
        self._set_listening_state(True)
        self.memory_runtime.record_user_transcript(
            transcript,
            metadata={"source": "qwen_realtime", "event": "user_transcript_completed"},
        )
        if self._tool_mode() != "router":
            logger.info("Qwen user transcript received with native tool mode: %s", transcript)
            return
        if not memory_command_writes_enabled():
            logger.info(
                "Qwen memory command writes disabled; transcript will be handled by session-end extractor: %s",
                transcript,
            )
        elif transcript not in self._memory_routed_transcripts:
            memory_result = await self.memory_command_router.handle(transcript)
            if memory_result is not None:
                self._memory_routed_transcripts.add(transcript)
                logger.info("Qwen memory command router handled transcript=%r result=%s", transcript, memory_result)
                await self.output_queue.put(
                    AdditionalOutputs({"role": "system", "content": f"[memory] {memory_result['type']}"})
                )
        if transcript in self._routed_transcripts:
            logger.debug("Qwen router fallback skipped duplicate transcript: %s", transcript)
            return
        route = self.tool_router.route(transcript)
        if route is None:
            logger.info("Qwen router fallback: transcript=%r no local tool matched", transcript)
            return
        self._routed_transcripts.add(transcript)
        tool_name, args = route
        logger.info("Qwen router fallback: transcript=%r tool=%s args=%s", transcript, tool_name, args)
        bg_tool = await self.tool_manager.start_tool(
            call_id=f"router_{uuid.uuid4().hex}",
            tool_call_routine=ToolCallRoutine(
                tool_name=tool_name,
                args_json_str=json.dumps(args, ensure_ascii=False),
                deps=self.deps,
            ),
            is_idle_tool_call=False,
        )
        logger.info("Qwen router fallback: started local tool %s id=%s", tool_name, bg_tool.tool_id)
        await self.output_queue.put(
            AdditionalOutputs({"role": "system", "content": f"[fallback tool {tool_name}]"})
        )

    async def _handle_native_tool_call(self, message: dict[str, Any]) -> None:
        tool_name = message.get("name") or message.get("tool_name")
        call_id = str(message.get("call_id") or message.get("id") or uuid.uuid4())
        args_value = message.get("arguments") or message.get("args") or {}
        args_json = args_value if isinstance(args_value, str) else json.dumps(args_value, ensure_ascii=False)
        if not isinstance(tool_name, str):
            logger.warning("Ignoring Qwen native tool call without a tool name: %s", message)
            return
        await self.tool_manager.start_tool(
            call_id=call_id,
            tool_call_routine=ToolCallRoutine(tool_name=tool_name, args_json_str=args_json, deps=self.deps),
            is_idle_tool_call=self.is_idle_tool_call,
        )

    async def _handle_tool_result(self, bg_tool: ToolNotification) -> None:
        if bg_tool.error is not None:
            tool_result: dict[str, Any] = {"error": bg_tool.error}
        else:
            tool_result = bg_tool.result or {"status": "completed"}

        if self.websocket is None:
            return

        if self._tool_mode() != "native":
            logger.info("Qwen router fallback: local tool %s completed with %s", bg_tool.tool_name, tool_result)
            return

        await self._send_json(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": bg_tool.id,
                    "output": json.dumps(tool_result, ensure_ascii=False),
                },
            },
        )
        await self._send_json({"type": "response.create"})

    async def _handle_interruption_started(self) -> None:
        if hasattr(self, "_clear_queue") and callable(self._clear_queue):
            self._clear_queue()
        if self.deps.head_wobbler is not None:
            self.deps.head_wobbler.reset()
        self._set_listening_state(True)

    async def _handle_message(self, message: dict[str, Any]) -> None:
        event_type = str(message.get("type") or "")

        # Speech Started - Reset tracker for new turn
        if event_type == "input_audio_buffer.speech_started":
            self._turn_counter = getattr(self, "_turn_counter", 0) + 1
            latency_tracker.reset(self._turn_counter)
            await self._handle_interruption_started()
            return

        # Session Updated (L2: R2)
        if event_type == "session.updated":
            latency_tracker.mark_R2_session_config_done()
            return

        # Speech Stopped (L1: speech_end - 计时起点)
        if event_type == "input_audio_buffer.speech_stopped":
            latency_tracker.mark_speech_end()
            self._set_listening_state(False)
            return

        # Transcription Delta (accumulate)
        if event_type in {"conversation.item.input_audio_transcription.delta", "input_audio_transcription.delta"}:
            delta = str(message.get("delta") or message.get("text") or "")
            if delta:
                self._pending_user_transcript_chunks.append(delta)
            return

        # Transcription Completed (L1: transcript_show)
        if event_type in {
            "conversation.item.input_audio_transcription.completed",
            "input_audio_transcription.completed",
        }:
            transcript = str(message.get("transcript") or message.get("text") or "")
            if not transcript:
                transcript = "".join(self._pending_user_transcript_chunks).strip()
                self._pending_user_transcript_chunks.clear()
            if transcript:
                latency_tracker.mark_transcript_show(transcript)
                logger.info("Qwen user transcript: %s", transcript)
                await self.output_queue.put(AdditionalOutputs({"role": "user", "content": transcript}))
                await self._handle_user_transcript(transcript)
            return

        # Audio Delta (L1: first_audio, L2: Q1 - 首次到达时记录)
        if event_type in {"response.audio.delta", "response.output_audio.delta"}:
            delta = message.get("delta") or message.get("audio")
            if not isinstance(delta, str):
                return
            audio_bytes_len = len(delta)
            # Track first audio for TTFB
            if not any(e["name"] == "first_audio" for e in latency_tracker.events):
                latency_tracker.mark_first_audio(audio_bytes_len)
                latency_tracker.mark_Q1_first_audio_received(audio_bytes_len)
            audio_bytes = base64.b64decode(delta)
            if self.gradio_mode and self.deps.head_wobbler is not None:
                self.deps.head_wobbler.feed(delta)
            self.last_activity_time = asyncio.get_event_loop().time()
            await self.output_queue.put(
                (QWEN_OUTPUT_SAMPLE_RATE, np.frombuffer(audio_bytes, dtype=np.int16).reshape(1, -1))
            )
            return

        # Audio Done
        if event_type in {"response.audio.done", "response.output_audio.done"}:
            if self.deps.head_wobbler is not None:
                self.deps.head_wobbler.reset()
            return

        # Response Transcript Delta (accumulate for display)
        if event_type in {"response.audio_transcript.delta", "response.output_audio_transcript.delta"}:
            delta = str(message.get("delta") or message.get("text") or "")
            if delta:
                self._pending_assistant_transcript_chunks.append(delta)
            return

        # Response Transcript Done (用于显示对话内容)
        if event_type in {"response.audio_transcript.done", "response.output_audio_transcript.done"}:
            transcript = str(message.get("transcript") or message.get("text") or "")
            if transcript:
                self._pending_assistant_transcript_chunks = [transcript]
            # Store for report display
            latency_tracker.mark("Q4_transcript_done", {"transcript": transcript}, level=2)
            await self._flush_transcript_chunks("assistant", self._pending_assistant_transcript_chunks)
            return

        # Tool Call
        if event_type in {"response.function_call_arguments.done", "tool_call"}:
            await self._handle_native_tool_call(message)
            return

        # Response Created (L1: response_start, L2: P2)
        if event_type == "response.created":
            latency_tracker.mark_response_start()
            latency_tracker.mark_P2_response_created()
            return

        # Response Done - Print summary
        if event_type == "response.done":
            latency_tracker.print_summary()
            return

        # Error
        if event_type == "error":
            error = message.get("error") or {}
            msg = error.get("message") if isinstance(error, dict) else str(error)
            logger.error("Qwen realtime error: %s", msg)
            await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": f"[error] {msg}"}))
            return

        if "transcript" in event_type or "transcription" in event_type:
            logger.info("Unhandled Qwen transcript event: %s", message)

    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive microphone audio and send it to Qwen."""
        if self.websocket is None:
            return

        input_sample_rate, audio_frame = frame
        if audio_frame.ndim == 2:
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            if audio_frame.shape[1] > 1:
                audio_frame = audio_frame[:, 0]

        if QWEN_INPUT_SAMPLE_RATE != input_sample_rate:
            audio_frame = resample(audio_frame, int(len(audio_frame) * QWEN_INPUT_SAMPLE_RATE / input_sample_rate))

        audio_frame = audio_to_int16(audio_frame)
        audio_message = base64.b64encode(audio_frame.tobytes()).decode("utf-8")
        try:
            await self._send_json({"type": "input_audio_buffer.append", "audio": audio_message})
        except Exception as e:
            logger.debug("Dropping audio frame: websocket not ready (%s)", e)

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit audio or UI messages to fastrtc."""
        idle_duration = asyncio.get_event_loop().time() - self.last_activity_time
        if idle_duration > 15.0 and self.deps.movement_manager.is_idle():
            try:
                await self.send_idle_signal(idle_duration)
            except Exception as e:
                logger.warning("Idle signal skipped: %s", e)
                return None
            self.last_activity_time = asyncio.get_event_loop().time()
        return await wait_for_item(self.output_queue)  # type: ignore[no-any-return]

    async def shutdown(self) -> None:
        """Shutdown the handler."""
        self._stop_event.set()
        await self.tool_manager.shutdown()
        self.memory_runtime.end_session_background(reason="shutdown")
        if self.websocket is not None:
            try:
                await self.websocket.close()
            except Exception:
                pass
            self.websocket = None
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def format_timestamp(self) -> str:
        """Format current timestamp with date, time, and elapsed seconds."""
        loop_time = asyncio.get_event_loop().time()
        elapsed_seconds = loop_time - self.start_time
        dt = datetime.now()
        return f"[{dt.strftime('%Y-%m-%d %H:%M:%S')} | +{elapsed_seconds:.1f}s]"

    async def send_idle_signal(self, idle_duration: float) -> None:
        """Send an idle signal into the Qwen session."""
        self.is_idle_tool_call = True
        timestamp_msg = (
            f"[Idle time update: {self.format_timestamp()} - No activity for {idle_duration:.1f}s] "
            "Choose a small robot action if appropriate, or stay still."
        )
        await self._send_json(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": timestamp_msg}],
                },
            },
        )
        await self._send_json({"type": "response.create"})

    async def get_available_voices(self) -> list[str]:
        """Return the curated Qwen voice list."""
        return list(QWEN_AVAILABLE_VOICES)
