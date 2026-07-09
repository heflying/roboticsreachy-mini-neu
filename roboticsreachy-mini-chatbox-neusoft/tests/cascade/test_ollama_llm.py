"""Tests for Ollama LLM provider."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from reachy_mini_conversation_app.cascade.interrupt_coordinator import TurnCancellationToken
from reachy_mini_conversation_app.cascade.llm.ollama import OllamaLLM


class _FakeChatCompletions:
    def __init__(self) -> None:
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return _fake_stream()


class _FakeAsyncOpenAI:
    def __init__(self, api_key: str, base_url: str) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self._completions = _FakeChatCompletions()
        self.chat = SimpleNamespace(completions=self._completions)


class _ClosableStream:
    def __init__(self, chunks):
        self._chunks = iter(chunks)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.closed:
            raise StopAsyncIteration
        try:
            return next(self._chunks)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    def close(self):
        self.closed = True


async def _fake_stream():
    yield SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content="Hello", tool_calls=None),
                finish_reason=None,
            )
        ],
        usage=None,
    )
    yield SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=" Ollama", tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=None,
    )
    yield SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3),
    )


def test_ollama_llm_streams_text(monkeypatch):
    """OllamaLLM yields text deltas and a done chunk."""

    async def run():
        fake_client = _FakeAsyncOpenAI(api_key="ollama", base_url="http://127.0.0.1:11434/v1")
        monkeypatch.setattr(
            "reachy_mini_conversation_app.cascade.llm.ollama._make_async_openai_client",
            lambda api_key, base_url: fake_client,
        )

        llm = OllamaLLM(
            model="qwen3:8b",
            base_url="http://127.0.0.1:11434/v1",
            system_instructions="system",
        )

        chunks = [chunk async for chunk in llm.generate([{"role": "user", "content": "Say hi"}])]

        assert [c.type for c in chunks] == ["text_delta", "text_delta", "done"]
        assert "".join(c.content or "" for c in chunks) == "Hello Ollama"
        assert fake_client._completions.kwargs["model"] == "qwen3:8b"
        assert fake_client._completions.kwargs["extra_body"] == {"enable_thinking": False}
        assert fake_client._completions.kwargs["messages"][0] == {"role": "system", "content": "system"}

    asyncio.run(run())


def test_ollama_llm_closes_stream_when_token_cancelled(monkeypatch):
    """OllamaLLM closes the provider stream promptly after cancellation."""

    async def run():
        chunks = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="Hello", tool_calls=None),
                        finish_reason=None,
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=" Ollama", tool_calls=None),
                        finish_reason=None,
                    )
                ],
                usage=None,
            ),
        ]
        stream = _ClosableStream(chunks)
        fake_completions = SimpleNamespace(create=AsyncMock(return_value=stream))
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=fake_completions))
        monkeypatch.setattr(
            "reachy_mini_conversation_app.cascade.llm.ollama._make_async_openai_client",
            lambda api_key, base_url: fake_client,
        )

        token = TurnCancellationToken(turn_id=11)
        llm = OllamaLLM()
        seen = []
        async for chunk in llm.generate([{"role": "user", "content": "hi"}], token=token):
            seen.append(chunk)
            token.cancel()

        assert [chunk.content for chunk in seen if chunk.content] == ["Hello"]
        assert stream.closed is True

    asyncio.run(run())


def test_ollama_entry_is_registered_in_static_config():
    """The repository config advertises the Ollama LLM provider."""

    cascade_yaml = Path("cascade.yaml").read_text(encoding="utf-8")

    assert "ollama:" in cascade_yaml
    assert "module: ollama" in cascade_yaml
    assert "class: OllamaLLM" in cascade_yaml
    assert "base_url: http://127.0.0.1:11434/v1" in cascade_yaml
    assert "requires: []" in cascade_yaml
