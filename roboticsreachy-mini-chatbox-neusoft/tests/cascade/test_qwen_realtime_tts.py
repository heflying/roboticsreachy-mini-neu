"""Tests for Qwen realtime TTS provider."""

from __future__ import annotations
import json
import base64
import asyncio

from reachy_mini_conversation_app.cascade.tts.qwen_realtime import QwenRealtimeTTS


class _FakeTTSWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.closed = False
        self.exits = 0
        self._events = [
            {"type": "response.audio.delta", "audio": base64.b64encode(b"\x01\x02").decode("ascii")},
            {"type": "response.audio.completed"},
        ]

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.exits += 1
        self.closed = True
        return False

    async def close(self) -> None:
        self.closed = True

    async def send(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def recv(self) -> str:
        return json.dumps(self._events.pop(0))


class _FailingPreparedTTSWebSocket(_FakeTTSWebSocket):
    async def send(self, data: str) -> None:
        payload = json.loads(data)
        self.sent.append(payload)
        if payload["type"] == "input_text_buffer.append":
            raise ConnectionError("stale prepared websocket")


def test_qwen_realtime_tts_yields_audio(monkeypatch):
    """TTS provider sends text and yields decoded PCM chunks."""
    async def run():
        fake_ws = _FakeTTSWebSocket()

        def fake_connect(url, additional_headers=None, extra_headers=None):
            assert url == "wss://example.test/tts?model=qwen3-tts-flash-realtime"
            headers = additional_headers or extra_headers
            assert headers["Authorization"] == "Bearer test-key"
            return fake_ws

        monkeypatch.setattr(
            "reachy_mini_conversation_app.cascade.tts.qwen_realtime._connect_websocket",
            fake_connect,
        )

        tts = QwenRealtimeTTS(
            api_key="test-key",
            websocket_url="wss://example.test/tts",
            voice="Cherry",
            sample_rate=24000,
        )
        chunks = [chunk async for chunk in tts.synthesize("你好")]

        assert chunks == [b"\x01\x02"]
        assert tts.sample_rate == 24000
        assert fake_ws.sent[0]["type"] == "session.update"
        assert fake_ws.sent[0]["session"]["voice"] == "Cherry"
        assert fake_ws.sent[0]["session"]["mode"] == "commit"
        assert fake_ws.sent[1] == {"type": "input_text_buffer.append", "text": "你好"}
        assert fake_ws.sent[2] == {"type": "input_text_buffer.commit"}
        assert fake_ws.sent[3] == {"type": "session.finish"}

    asyncio.run(run())


def test_qwen_realtime_tts_ignores_kokoro_autotest_voice():
    """Qwen TTS falls back when autotest passes a Kokoro-only voice name."""
    tts = QwenRealtimeTTS(api_key="test-key", voice="Cherry")

    assert tts._voice_for_request("af_heart") == "Cherry"
    assert tts._voice_for_request("Ethan") == "Ethan"


def test_qwen_realtime_tts_extracts_audio_shapes():
    """Audio extraction supports several event shapes."""
    assert QwenRealtimeTTS._extract_audio({"audio": "a"}) == "a"
    assert QwenRealtimeTTS._extract_audio({"output": {"audio": "b"}}) == "b"
    assert QwenRealtimeTTS._extract_audio({"output": {"audio": {"data": "c"}}}) == "c"


def test_qwen_realtime_tts_prepare_stream_reuses_connection(monkeypatch):
    """Prepared TTS WebSocket is reused for the next synthesis request."""
    async def run():
        fake_ws = _FakeTTSWebSocket()
        connect_count = 0

        def fake_connect(url, additional_headers=None, extra_headers=None):
            nonlocal connect_count
            connect_count += 1
            return fake_ws

        monkeypatch.setattr(
            "reachy_mini_conversation_app.cascade.tts.qwen_realtime._connect_websocket",
            fake_connect,
        )

        tts = QwenRealtimeTTS(api_key="test-key", websocket_url="wss://example.test/tts", voice="Cherry")
        await tts.prepare_stream()
        chunks = [chunk async for chunk in tts.synthesize("你好")]

        assert chunks == [b"\x01\x02"]
        assert connect_count == 1
        assert fake_ws.sent[0]["type"] == "session.update"
        assert fake_ws.sent[1] == {"type": "input_text_buffer.append", "text": "你好"}
        assert fake_ws.sent[2] == {"type": "input_text_buffer.commit"}
        assert fake_ws.sent[3] == {"type": "session.finish"}
        assert fake_ws.closed
        assert fake_ws.exits == 1

    asyncio.run(run())


def test_qwen_realtime_tts_retries_when_prepared_connection_is_stale(monkeypatch):
    """A stale prepared TTS connection falls back to a fresh WebSocket before yielding audio."""
    async def run():
        stale_ws = _FailingPreparedTTSWebSocket()
        fresh_ws = _FakeTTSWebSocket()
        sockets = [stale_ws, fresh_ws]

        def fake_connect(url, additional_headers=None, extra_headers=None):
            return sockets.pop(0)

        monkeypatch.setattr(
            "reachy_mini_conversation_app.cascade.tts.qwen_realtime._connect_websocket",
            fake_connect,
        )

        tts = QwenRealtimeTTS(api_key="test-key", websocket_url="wss://example.test/tts", voice="Cherry")
        await tts.prepare_stream()
        chunks = [chunk async for chunk in tts.synthesize("你好")]

        assert chunks == [b"\x01\x02"]
        assert stale_ws.closed
        assert fresh_ws.sent[0]["type"] == "session.update"
        assert fresh_ws.sent[1] == {"type": "input_text_buffer.append", "text": "你好"}
        assert sockets == []

    asyncio.run(run())


def test_qwen_realtime_tts_discards_old_prepared_connection(monkeypatch):
    """An old prepared TTS connection is closed before synthesis starts."""
    async def run():
        old_ws = _FakeTTSWebSocket()
        fresh_ws = _FakeTTSWebSocket()
        sockets = [old_ws, fresh_ws]

        def fake_connect(url, additional_headers=None, extra_headers=None):
            return sockets.pop(0)

        monkeypatch.setattr(
            "reachy_mini_conversation_app.cascade.tts.qwen_realtime._connect_websocket",
            fake_connect,
        )

        tts = QwenRealtimeTTS(api_key="test-key", websocket_url="wss://example.test/tts", voice="Cherry")
        await tts.prepare_stream()
        tts.prepared_max_age_s = 0.0
        tts._prepared_at = 0.0

        chunks = [chunk async for chunk in tts.synthesize("hello")]

        assert chunks == [b"\x01\x02"]
        assert old_ws.closed
        assert len(old_ws.sent) == 1
        assert old_ws.sent[0]["type"] == "session.update"
        assert fresh_ws.sent[0]["type"] == "session.update"
        assert fresh_ws.sent[1] == {"type": "input_text_buffer.append", "text": "hello"}
        assert sockets == []

    asyncio.run(run())
