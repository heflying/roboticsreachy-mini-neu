"""Qwen realtime ASR provider over WebSocket."""

from __future__ import annotations
import json
import base64
import asyncio
import logging
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
from typing import Any, Optional

from .audio_utils import wav_to_pcm_int16
from .base_streaming import StreamingASRProvider


logger = logging.getLogger(__name__)


async def _connect_websocket(url: str, headers: dict[str, str]) -> Any:
    """Connect to a WebSocket lazily so tests can run without websockets installed."""
    import websockets

    try:
        return await websockets.connect(url, additional_headers=headers)
    except TypeError:
        return await websockets.connect(url, extra_headers=headers)


class QwenRealtimeASR(StreamingASRProvider):
    """Qwen realtime ASR using DashScope WebSocket events.

    The provider keeps the current local VAD turn-taking model: audio is sent
    while the user speaks, and end_stream commits the buffered audio and waits
    for the final transcript.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "qwen3-asr-flash-realtime",
        websocket_url: str = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
        sample_rate: int = 16000,
        language: str = "zh",
        enable_itn: bool = True,
        wait_timeout_s: float = 20.0,
    ) -> None:
        """Initialize Qwen realtime ASR."""
        self.api_key = api_key
        self.model = model
        self.websocket_url = websocket_url
        self.sample_rate = sample_rate
        self.language = language
        self.enable_itn = enable_itn
        self.wait_timeout_s = wait_timeout_s
        self.last_cost = 0.0

        self._ws: Any | None = None
        self._receiver_task: asyncio.Task[None] | None = None
        self._event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._partial_transcript: str | None = None
        self._final_transcript: str | None = None
        self._stream_lock = asyncio.Lock()

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "X-DashScope-DataInspection": "enable",
        }

    def _websocket_url_with_model(self) -> str:
        """Return WebSocket URL with the model query parameter required by ASR realtime."""
        parsed = urlparse(self.websocket_url)
        query = dict(parse_qsl(parsed.query))
        query.setdefault("model", self.model)
        return urlunparse(parsed._replace(query=urlencode(query)))

    async def _send_event(self, event: dict[str, Any]) -> None:
        if self._ws is None:
            raise RuntimeError("Qwen ASR stream is not started")
        await self._ws.send(json.dumps(event))

    async def _receiver_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                event = json.loads(raw) if isinstance(raw, str) else raw
                if not isinstance(event, dict):
                    continue
                text = self._extract_text(event)
                event_type = str(event.get("type") or event.get("event") or "").lower()
                if text:
                    if any(marker in event_type for marker in ("partial", "delta", "updated")):
                        self._partial_transcript = text
                    elif any(marker in event_type for marker in ("final", "completed", "done")):
                        self._final_transcript = text
                    else:
                        self._partial_transcript = text
                await self._event_queue.put(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("Qwen ASR receiver stopped: %s", exc)

    @staticmethod
    async def _cleanup_connection(receiver_task: asyncio.Task[None] | None, ws: Any | None) -> None:
        """Close the ASR WebSocket in the background after final text is available."""
        if receiver_task:
            receiver_task.cancel()
            try:
                await receiver_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.debug("Qwen ASR receiver cleanup ignored error: %s", exc)

        if ws is not None:
            try:
                await ws.close()
            except Exception as exc:
                logger.debug("Qwen ASR websocket cleanup ignored error: %s", exc)

    @staticmethod
    def _extract_text(event: dict[str, Any]) -> str | None:
        """Extract transcript text from several DashScope-style shapes."""
        candidates: list[Any] = [
            event.get("transcript"),
            event.get("text"),
            event.get("sentence", {}).get("text") if isinstance(event.get("sentence"), dict) else None,
            event.get("output", {}).get("text") if isinstance(event.get("output"), dict) else None,
            event.get("output", {}).get("transcript") if isinstance(event.get("output"), dict) else None,
        ]
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return None

    @staticmethod
    def _is_final_event(event: dict[str, Any]) -> bool:
        event_type = str(event.get("type") or event.get("event") or "").lower()
        return any(marker in event_type for marker in ("final", "completed", "done"))

    async def start_stream(self) -> None:
        """Start a WebSocket ASR session."""
        from reachy_mini_conversation_app.cascade.timing import tracker

        self._partial_transcript = None
        self._final_transcript = None
        async with self._stream_lock:
            if self._ws is not None:
                tracker.mark("asr_ws_reused")
                return

            self._event_queue = asyncio.Queue()
            ws_url = self._websocket_url_with_model()
            logger.info(f"Connecting to Qwen ASR WebSocket: {ws_url}")
            tracker.mark("asr_ws_connect_start")
            self._ws = await _connect_websocket(ws_url, self._headers())
            tracker.mark("asr_ws_connected")
            await self._send_event(
                {
                    "type": "session.update",
                    "session": {
                        "input_audio_format": "pcm",
                        "sample_rate": self.sample_rate,
                        "language": self.language,
                        "enable_itn": self.enable_itn,
                        "turn_detection": None,
                    },
                }
            )
            tracker.mark("asr_session_update_sent")
            self._receiver_task = asyncio.create_task(self._receiver_loop())

    async def prepare_stream(self) -> None:
        """Pre-connect a streaming ASR session before the user starts speaking."""
        if self._ws is None:
            await self.start_stream()

    async def send_audio_chunk(self, audio_chunk: bytes) -> None:
        """Send one WAV/PCM chunk to the realtime ASR session.

        Raises:
            RuntimeError: If the WebSocket stream is not started.
        """
        # Check WebSocket exists before recording timing
        if self._ws is None:
            logger.warning("send_audio_chunk called but WebSocket not ready")
            raise RuntimeError("Qwen ASR stream is not started")

        # 只在首次发送时记录 B3 start（避免大量日志）
        if not hasattr(self, "_audio_send_started"):
            from reachy_mini_conversation_app.cascade.timing import tracker
            tracker.mark("asr_audio_send_start", {"chunk_bytes": len(audio_chunk)})
            self._audio_send_started = True

        pcm = wav_to_pcm_int16(audio_chunk, self.sample_rate)
        await self._send_event(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode("ascii"),
            }
        )

    async def get_partial_transcript(self) -> Optional[str]:
        """Return the latest partial transcript, if any."""
        return self._partial_transcript

    async def end_stream(self) -> str:
        """Commit audio and wait for final transcript."""
        from reachy_mini_conversation_app.cascade.timing import tracker

        # B3: 音频发送完成（在 commit 之前）
        if hasattr(self, "_audio_send_started"):
            tracker.mark("asr_audio_send_complete")
            del self._audio_send_started  # 清除标志

        tracker.mark("asr_commit_start")
        await self._send_event({"type": "input_audio_buffer.commit"})
        await self._send_event({"type": "session.finish"})
        tracker.mark("asr_commit_sent")
        deadline = asyncio.get_running_loop().time() + self.wait_timeout_s

        while True:
            if self._final_transcript:
                break
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                logger.warning("Timed out waiting for Qwen ASR final transcript")
                break
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            text = self._extract_text(event)
            event_type = str(event.get("type") or event.get("event") or "")
            if text:
                tracker.mark("asr_text_event", {"event_type": event_type, "text_len": len(text)})
            if text and self._is_final_event(event):
                self._final_transcript = text
                tracker.mark("asr_final_received", {"event_type": event_type, "text_len": len(text)})
                break

        cleanup_task = self._receiver_task
        cleanup_ws = self._ws
        self._receiver_task = None
        self._ws = None

        # B5: ASR 结果返回时间点
        final_text = (self._final_transcript or self._partial_transcript or "").strip()
        tracker.mark("asr_result_delivered", {"transcript_len": len(final_text)})

        tracker.mark("asr_cleanup_scheduled")
        asyncio.create_task(self._cleanup_connection(cleanup_task, cleanup_ws))
        return final_text
