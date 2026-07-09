"""Qwen realtime TTS provider over WebSocket."""

from __future__ import annotations
import json
import base64
import asyncio
import logging
import time
import os
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
from typing import Any, Optional, AsyncIterator

from .base import TTSProvider


logger = logging.getLogger(__name__)

KOKORO_VOICE_PREFIXES = ("af_", "am_", "bf_", "bm_")


def _connect_websocket(url: str, headers: dict[str, str]) -> Any:
    """Connect to a WebSocket lazily so tests can run without websockets installed."""
    import websockets

    try:
        return websockets.connect(url, additional_headers=headers)
    except TypeError:
        return websockets.connect(url, extra_headers=headers)


class QwenRealtimeTTS(TTSProvider):
    """Qwen realtime TTS with streaming audio output.

    Session Tracking (方案 A):
    - _session_id: 递增计数，每次 synthesize 递增
    - _stale_session_ids: 记录已取消的 session
    - cancel_current(): 标记当前 session 为 stale，关闭 WebSocket
    - cancel_current_from_thread(): 从任意线程调用
    """

    prefer_single_request = True

    def __init__(
        self,
        api_key: str,
        model: str = "qwen3-tts-flash-realtime",
        voice: str = "Ethan",
        websocket_url: str = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
        response_format: str = "pcm",
        sample_rate: int = 24000,
        mode: str = "commit",
        language_type: str = "Chinese",
        wait_timeout_s: float = 30.0,
    ) -> None:
        """Initialize Qwen realtime TTS."""
        self.api_key = api_key
        self.model = model
        self.default_voice = voice
        self.websocket_url = websocket_url
        self.response_format = response_format
        self._sample_rate = sample_rate
        self.mode = mode
        self.language_type = language_type
        self.wait_timeout_s = wait_timeout_s
        self.last_cost = 0.0

        # Session tracking for interrupt isolation (方案 A)
        self._session_id: int = 0
        self._stale_session_ids: set[int] = set()
        self._current_ws: Any | None = None

        self._prepared_ws: Any | None = None
        self._prepared_cm: Any | None = None
        self._prepared_voice: str | None = None
        self._prepared_at: float | None = None
        self._preparing: bool = False  # 预连接是否正在进行
        self._prepare_task: asyncio.Task[None] | None = None  # 预连接任务引用
        self.prepared_max_age_s = float(os.getenv("QWEN_TTS_PREPARED_MAX_AGE_S", "8.0"))
        self.reuse_first_audio_timeout_s = float(os.getenv("QWEN_TTS_REUSE_FIRST_AUDIO_TIMEOUT_S", "3.0"))
        self.wait_preconnect_s = float(os.getenv("QWEN_TTS_WAIT_PRECONNECT_S", "0.5"))  # 等待预连接的最大时间

    @property
    def sample_rate(self) -> int:
        """Audio sample rate in Hz."""
        return self._sample_rate

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "X-DashScope-DataInspection": "enable",
        }

    def _websocket_url_with_model(self) -> str:
        """Return WebSocket URL with the model query parameter required by TTS realtime."""
        parsed = urlparse(self.websocket_url)
        query = dict(parse_qsl(parsed.query))
        query.setdefault("model", self.model)
        return urlunparse(parsed._replace(query=urlencode(query)))

    def _voice_for_request(self, voice: Optional[str]) -> str:
        """Return a Qwen voice, ignoring Kokoro-only autotest voice overrides."""
        if voice and not voice.startswith(KOKORO_VOICE_PREFIXES):
            return voice
        return self.default_voice

    async def _send_session_update(self, ws: Any, voice: str) -> None:
        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "mode": self.mode,
                        "voice": voice,
                        "response_format": self.response_format,
                        "sample_rate": self.sample_rate,
                        "language_type": self.language_type,
                    },
                }
            )
        )

    async def prepare_stream(self, voice: Optional[str] = None) -> None:
        """Pre-connect a Qwen TTS WebSocket before text is ready.

        状态追踪：
        - 设置 _preparing=True 标记预连接正在进行
        - 保存 _prepare_task 用于 synthesize 等待
        """
        from reachy_mini_conversation_app.cascade.timing import tracker

        voice_to_use = self._voice_for_request(voice)

        # 如果已有相同 voice 的可用预连接，直接返回
        if self._prepared_ws is not None and self._prepared_voice == voice_to_use:
            tracker.mark("tts_ws_preconnect_reused")
            return

        # 如果预连接正在进行中，等待它完成
        if self._preparing and self._prepare_task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._prepare_task), timeout=self.wait_preconnect_s)
            except asyncio.TimeoutError:
                logger.info("Previous prepare_stream still pending; will start new one")
            except Exception:
                pass

        # 清理旧预连接
        await self._close_prepared()

        # 标记预连接正在进行
        self._preparing = True

        tracker.mark("tts_ws_preconnect_start")
        cm = _connect_websocket(self._websocket_url_with_model(), self._headers())
        ws = await cm.__aenter__()
        tracker.mark("tts_ws_preconnected")
        await self._send_session_update(ws, voice_to_use)
        tracker.mark("tts_preconnect_session_update_sent")

        # 设置预连接状态
        self._prepared_cm = cm
        self._prepared_ws = ws
        self._prepared_voice = voice_to_use
        self._prepared_at = time.monotonic()
        self._preparing = False
        self._prepare_task = None

    async def _close_prepared(self) -> None:
        """Close any prepared TTS WebSocket and cancel pending prepare task."""
        # 取消正在进行的预连接任务
        if self._prepare_task is not None and not self._prepare_task.done():
            self._prepare_task.cancel()
            try:
                await self._prepare_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.debug("Cancelled prepare_task cleanup ignored error: %s", exc)

        cm = self._prepared_cm
        ws = self._prepared_ws
        self._prepared_cm = None
        self._prepared_ws = None
        self._prepared_voice = None
        self._prepared_at = None
        self._preparing = False
        self._prepare_task = None

        if cm is not None:
            try:
                await cm.__aexit__(None, None, None)
                return
            except Exception as exc:
                logger.debug("Prepared Qwen TTS context cleanup ignored error: %s", exc)
        if ws is not None:
            try:
                await ws.close()
            except Exception as exc:
                logger.debug("Prepared Qwen TTS websocket cleanup ignored error: %s", exc)

    async def _run_synthesis_on_ws(
        self,
        ws: Any,
        text: str,
        first_audio_timeout_s: float | None = None,
    ) -> AsyncIterator[bytes]:
        from reachy_mini_conversation_app.cascade.timing import tracker

        await ws.send(json.dumps({"type": "input_text_buffer.append", "text": text}))
        tracker.mark("tts_text_append_sent", {"text_len": len(text)})
        await ws.send(json.dumps({"type": "input_text_buffer.commit"}))
        await ws.send(json.dumps({"type": "session.finish"}))
        tracker.mark("tts_commit_sent")

        first_chunk = True
        chunk_count = 0
        audio_bytes = 0
        request_started_at = time.perf_counter()
        first_audio_at: float | None = None
        last_audio_at: float | None = None
        max_audio_gap_ms = 0.0
        while True:
            try:
                timeout_s = first_audio_timeout_s if first_chunk and first_audio_timeout_s else self.wait_timeout_s
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
            except asyncio.TimeoutError:
                if first_chunk:
                    raise TimeoutError(
                        f"Timed out waiting for Qwen realtime TTS first audio after "
                        f"{first_audio_timeout_s or self.wait_timeout_s:.1f}s"
                    )
                logger.warning("Timed out waiting for Qwen realtime TTS completion; ending current synthesis")
                break
            event = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(event, dict):
                continue

            event_type = str(event.get("type") or event.get("event") or "").lower()
            if "error" in event_type:
                raise RuntimeError(f"Qwen realtime TTS error: {event}")

            audio_b64 = self._extract_audio(event)
            if audio_b64:
                chunk = base64.b64decode(audio_b64)
                now = time.perf_counter()
                chunk_count += 1
                audio_bytes += len(chunk)
                if first_chunk:
                    first_audio_at = now
                    tracker.mark(
                        "tts_first_chunk_ready",
                        {"event_type": event_type, "chunk_bytes": len(chunk)},
                    )
                    first_chunk = False
                elif last_audio_at is not None:
                    gap_ms = (now - last_audio_at) * 1000
                    max_audio_gap_ms = max(max_audio_gap_ms, gap_ms)
                    if gap_ms > 1000:
                        tracker.mark(
                            "tts_audio_chunk_gap",
                            {"chunk": chunk_count, "gap_ms": round(gap_ms, 1)},
                        )
                        logger.warning("Qwen TTS audio chunk gap %.1fms before chunk %s", gap_ms, chunk_count)
                last_audio_at = now
                yield chunk

            if any(marker in event_type for marker in ("done", "completed", "finished")):
                done_at = time.perf_counter()
                tracker.mark(
                    "tts_finish_event_received",
                    {
                        "event_type": event_type,
                        "chunks": chunk_count,
                        "audio_bytes": audio_bytes,
                        "stream_ms": round((done_at - request_started_at) * 1000, 1),
                        "first_to_done_ms": round((done_at - first_audio_at) * 1000, 1)
                        if first_audio_at is not None
                        else None,
                        "max_gap_ms": round(max_audio_gap_ms, 1),
                    },
                )
                break

    async def _close_context_or_ws(self, cm: Any | None, ws: Any) -> None:
        if cm is not None:
            try:
                await cm.__aexit__(None, None, None)
            except Exception as exc:
                logger.debug("Qwen TTS context close ignored error: %s", exc)
        else:
            try:
                await ws.close()
            except Exception as exc:
                logger.debug("Qwen TTS websocket close ignored error: %s", exc)

    async def _synthesize_fresh(self, text: str, voice: str) -> AsyncIterator[bytes]:
        from reachy_mini_conversation_app.cascade.timing import tracker

        tracker.mark("tts_ws_connect_start")
        async with _connect_websocket(self._websocket_url_with_model(), self._headers()) as ws:
            tracker.mark("tts_ws_connected")
            await self._send_session_update(ws, voice)
            tracker.mark("tts_session_update_sent")
            async for chunk in self._run_synthesis_on_ws(ws, text):
                yield chunk

    async def synthesize(self, text: str, voice: Optional[str] = None) -> AsyncIterator[bytes]:
        """Synthesize text and yield PCM chunks as they arrive.

        场景覆盖：
        - 如果预连接正在进行，等待最多 wait_preconnect_s
        - 如果预连接已完成且未过期，reuse
        - 否则启动新连接
        """
        if not text.strip():
            return

        from reachy_mini_conversation_app.cascade.timing import tracker

        tracker.mark("tts_start", {"text_len": len(text)})
        voice_to_use = self._voice_for_request(voice)

        # 场景 B/C/D: 预连接正在进行中，等待它完成
        if self._preparing and self._prepare_task is not None:
            tracker.mark("tts_wait_preconnect_start")
            try:
                await asyncio.wait_for(asyncio.shield(self._prepare_task), timeout=self.wait_preconnect_s)
                tracker.mark("tts_wait_preconnect_success")
            except asyncio.TimeoutError:
                tracker.mark("tts_wait_preconnect_timeout")
                logger.info(
                    "TTS pre-connect still pending after %.1fs; proceeding with fresh connection",
                    self.wait_preconnect_s,
                )
            except Exception as exc:
                tracker.mark("tts_wait_preconnect_failed", {"error": str(exc)})
                logger.warning("TTS pre-connect task failed: %s", exc)

        # 场景 E/G/H: 预连接已完成，检查是否可 reuse
        if self._prepared_ws is not None and self._prepared_voice == voice_to_use:
            prepared_age_s = (
                time.monotonic() - self._prepared_at
                if self._prepared_at is not None
                else self.prepared_max_age_s + 1
            )
            if prepared_age_s > self.prepared_max_age_s:
                logger.info(
                    "Prepared Qwen TTS websocket is %.1fs old; reconnecting instead of reusing",
                    prepared_age_s,
                )
                tracker.mark("tts_ws_prepared_stale", {"age_s": round(prepared_age_s, 1)})
                await self._close_prepared()
            else:
                ws = self._prepared_ws
                cm = self._prepared_cm
                self._prepared_ws = None
                self._prepared_cm = None
                self._prepared_voice = None
                self._prepared_at = None
                tracker.mark("tts_ws_reused", {"age_s": round(prepared_age_s, 1)})
                yielded_audio = False
                try:
                    async for chunk in self._run_synthesis_on_ws(
                        ws,
                        text,
                        first_audio_timeout_s=self.reuse_first_audio_timeout_s,
                    ):
                        yielded_audio = True
                        yield chunk
                    tracker.mark("tts_api_complete")
                    return
                except Exception as exc:
                    if yielded_audio:
                        raise
                    logger.warning(
                        "Prepared Qwen TTS websocket failed before audio; retrying with a fresh connection: %s",
                        exc,
                    )
                    tracker.mark("tts_ws_reuse_failed")
                finally:
                    await self._close_context_or_ws(cm, ws)

        # 场景 A/I: 无法 reuse，启动新连接
        # 先清理残留的预连接状态（包括后台正在进行的任务）
        await self._close_prepared()

        async for chunk in self._synthesize_fresh(text, voice_to_use):
            yield chunk

        tracker.mark("tts_api_complete")

    @staticmethod
    def _extract_audio(event: dict[str, Any]) -> str | None:
        """Extract base64 audio from several DashScope-style event shapes."""
        candidates: list[Any] = [
            event.get("audio"),
            event.get("delta"),
            event.get("data"),
            event.get("output", {}).get("audio") if isinstance(event.get("output"), dict) else None,
            event.get("output", {}).get("audio", {}).get("data")
            if isinstance(event.get("output"), dict) and isinstance(event.get("output", {}).get("audio"), dict)
            else None,
        ]
        for candidate in candidates:
            if isinstance(candidate, str) and candidate:
                return candidate
        return None

    async def cancel_current(self) -> None:
        """打断当前 TTS session.

        方案 A 策略：
        1. 标记当前 session_id 为 stale
        2. 关闭当前 WebSocket（不复用）
        3. 清空 current_ws 和 prepared_ws

        这是异步方法，需要在 event loop 内调用或通过 run_coroutine_threadsafe。
        """
        current_sid = self._session_id
        if current_sid > 0:
            self._stale_session_ids.add(current_sid)
            logger.info("TTS session %s marked as stale", current_sid)

        # 关闭当前 WebSocket（异步关闭）
        if self._current_ws is not None:
            ws_to_close = self._current_ws
            self._current_ws = None
            try:
                await self._close_ws_async(ws_to_close)
                logger.info("Closed current TTS WebSocket for session %s", current_sid)
            except Exception as e:
                logger.warning("Failed to close TTS WS: %s", e)

        # 清空 prepared WebSocket
        await self._close_prepared()

        # 清理过旧的 stale session IDs
        self._cleanup_stale_sessions()

    def cancel_current_from_thread(self, event_loop: asyncio.AbstractEventLoop) -> None:
        """从任意线程调用 cancel_current.

        使用 asyncio.run_coroutine_threadsafe 调度异步关闭。
        用于 VAD 线程触发打断时。

        Args:
            event_loop: 运行中的 asyncio event loop
        """
        # 标记当前 session 为 stale（同步操作）
        current_sid = self._session_id
        if current_sid > 0:
            self._stale_session_ids.add(current_sid)
            logger.info("TTS session %s marked as stale from thread", current_sid)

        # 清空 current_ws（同步清理引用）
        self._current_ws = None

        # 异步关闭（fire-and-forget）
        if event_loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(
                    self._close_prepared(),
                    event_loop
                )
                logger.info("Scheduled TTS WebSocket close from thread")
            except Exception as e:
                logger.warning("Failed to schedule TTS cancel from thread: %s", e)
        else:
            logger.warning("Event loop not running, cannot cancel TTS safely")

    async def _close_ws_async(self, ws: Any) -> None:
        """Async helper to close WebSocket."""
        try:
            await ws.close()
        except Exception as e:
            logger.debug("TTS WS close error (ignored): %s", e)

    def _is_session_stale(self, session_id: int) -> bool:
        """检查 session 是否已 stale."""
        return session_id in self._stale_session_ids

    def _cleanup_stale_sessions(self, keep_recent: int = 5) -> None:
        """清理过旧的 stale session 记录."""
        if len(self._stale_session_ids) > keep_recent:
            # 只保留最近的 keep_recent 个
            sorted_ids = sorted(self._stale_session_ids)
            to_remove = sorted_ids[:-keep_recent]
            for sid in to_remove:
                self._stale_session_ids.discard(sid)
            logger.debug("Cleaned up %s stale session IDs", len(to_remove))
