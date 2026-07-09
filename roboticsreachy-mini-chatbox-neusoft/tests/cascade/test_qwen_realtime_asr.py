"""Tests for Qwen realtime ASR provider."""

from __future__ import annotations

import asyncio
import json
from typing import Callable

from reachy_mini_conversation_app.cascade.asr.audio_utils import pcm_to_wav
from reachy_mini_conversation_app.cascade.asr.qwen_realtime import QwenRealtimeASR


NI = "\u4F60"
NI_HAO = "\u4F60\u597D"


class _FakeASRWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.closed = False
        self._events = [
            {"type": "response.audio_transcript.delta", "transcript": NI},
            {"type": "response.audio_transcript.completed", "transcript": NI_HAO},
        ]

    async def send(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def close(self) -> None:
        self.closed = True

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        await asyncio.sleep(0)
        if not self._events:
            await asyncio.sleep(10)
        return json.dumps(self._events.pop(0))


async def _wait_until(predicate: Callable[[], bool], timeout_s: float = 1.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError("condition was not met before timeout")
        await asyncio.sleep(0.01)


def test_qwen_realtime_asr_stream_lifecycle(monkeypatch):
    """ASR provider starts a session, sends audio, and returns final text."""
    async def run():
        fake_ws = _FakeASRWebSocket()

        async def fake_connect(url, additional_headers=None, extra_headers=None):
            assert url == "wss://example.test/asr?model=qwen3-asr-flash-realtime"
            headers = additional_headers or extra_headers
            assert headers["Authorization"] == "Bearer test-key"
            return fake_ws

        monkeypatch.setattr(
            "reachy_mini_conversation_app.cascade.asr.qwen_realtime._connect_websocket",
            fake_connect,
        )

        asr = QwenRealtimeASR(
            api_key="test-key",
            websocket_url="wss://example.test/asr",
            wait_timeout_s=1,
        )
        await asr.start_stream()
        await asyncio.sleep(0)

        wav = pcm_to_wav((b"\x00\x00" * 512), 16000)
        await asr.send_audio_chunk(wav)
        final_text = await asr.end_stream()

        assert final_text == NI_HAO
        assert fake_ws.sent[0]["type"] == "session.update"
        assert fake_ws.sent[0]["session"]["input_audio_format"] == "pcm"
        assert fake_ws.sent[0]["session"]["turn_detection"] is None
        assert fake_ws.sent[1]["type"] == "input_audio_buffer.append"
        assert fake_ws.sent[2]["type"] == "input_audio_buffer.commit"
        assert fake_ws.sent[3]["type"] == "session.finish"
        await _wait_until(lambda: fake_ws.closed)

    asyncio.run(run())


def test_qwen_realtime_asr_extracts_text_shapes():
    """Transcript extraction supports several event shapes."""
    assert QwenRealtimeASR._extract_text({"text": "hello"}) == "hello"
    assert QwenRealtimeASR._extract_text({"output": {"transcript": "hi"}}) == "hi"
    assert QwenRealtimeASR._extract_text({"sentence": {"text": "ok"}}) == "ok"


def test_qwen_realtime_asr_prepare_stream_reuses_connection(monkeypatch):
    """Prepared ASR sessions are reused when speech starts."""
    async def run():
        fake_ws = _FakeASRWebSocket()
        connect_count = 0

        async def fake_connect(url, additional_headers=None, extra_headers=None):
            nonlocal connect_count
            connect_count += 1
            return fake_ws

        monkeypatch.setattr(
            "reachy_mini_conversation_app.cascade.asr.qwen_realtime._connect_websocket",
            fake_connect,
        )

        asr = QwenRealtimeASR(api_key="test-key", websocket_url="wss://example.test/asr")
        await asr.prepare_stream()
        await asr.start_stream()

        assert connect_count == 1
        assert fake_ws.sent[0]["type"] == "session.update"

    asyncio.run(run())


def test_qwen_realtime_asr_start_waits_for_inflight_prepare(monkeypatch):
    """Concurrent pre-connect and start share one WebSocket session."""
    async def run():
        fake_ws = _FakeASRWebSocket()
        connect_count = 0
        connect_started = asyncio.Event()
        allow_connect = asyncio.Event()

        async def fake_connect(url, additional_headers=None, extra_headers=None):
            nonlocal connect_count
            connect_count += 1
            connect_started.set()
            await allow_connect.wait()
            return fake_ws

        monkeypatch.setattr(
            "reachy_mini_conversation_app.cascade.asr.qwen_realtime._connect_websocket",
            fake_connect,
        )

        asr = QwenRealtimeASR(api_key="test-key", websocket_url="wss://example.test/asr")
        prepare_task = asyncio.create_task(asr.prepare_stream())
        await connect_started.wait()
        start_task = asyncio.create_task(asr.start_stream())
        await asyncio.sleep(0)
        allow_connect.set()
        await asyncio.gather(prepare_task, start_task)

        assert connect_count == 1
        assert fake_ws.sent[0]["type"] == "session.update"
        assert len([event for event in fake_ws.sent if event["type"] == "session.update"]) == 1

    asyncio.run(run())
