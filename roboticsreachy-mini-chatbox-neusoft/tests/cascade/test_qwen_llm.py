"""Tests for Qwen LLM provider."""

from __future__ import annotations
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from reachy_mini_conversation_app.cascade.interrupt_coordinator import TurnCancellationToken
from reachy_mini_conversation_app.cascade.llm.qwen import QwenLLM


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
                delta=SimpleNamespace(content=" world", tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=None,
    )
    yield SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3),
    )


def test_qwen_llm_streams_text(monkeypatch):
    """QwenLLM yields text deltas and a done chunk."""
    async def run():
        fake_client = _FakeAsyncOpenAI(api_key="test", base_url="https://example.test/v1")
        monkeypatch.setattr(
            "reachy_mini_conversation_app.cascade.llm.qwen._make_async_openai_client",
            lambda api_key, base_url: fake_client,
        )

        llm = QwenLLM(
            api_key="test",
            model="qwen-plus",
            base_url="https://example.test/v1",
            system_instructions="system",
            input_cost_per_1m=1,
            output_cost_per_1m=2,
        )

        chunks = [chunk async for chunk in llm.generate([{"role": "user", "content": "Say hi"}])]

        assert [c.type for c in chunks] == ["text_delta", "text_delta", "done"]
        assert "".join(c.content or "" for c in chunks) == "Hello world"
        assert fake_client._completions.kwargs["model"] == "qwen-plus"
        assert fake_client._completions.kwargs["extra_body"] == {"enable_thinking": False}
        assert fake_client._completions.kwargs["messages"][0] == {"role": "system", "content": "system"}
        assert llm.last_cost > 0

    asyncio.run(run())


def test_qwen_llm_accumulates_tool_calls(monkeypatch):
    """QwenLLM converts streamed tool-call deltas to LLMChunk tool calls."""
    async def run():
        async def stream_with_tool():
            tool_delta_1 = SimpleNamespace(
                index=0,
                id="call_1",
                function=SimpleNamespace(name="speak", arguments='{"message":'),
            )
            tool_delta_2 = SimpleNamespace(
                index=0,
                id=None,
                function=SimpleNamespace(name=None, arguments='"hello"}'),
            )
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=[tool_delta_1]), finish_reason=None)],
                usage=None,
            )
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=[tool_delta_2]), finish_reason="tool_calls")],
                usage=None,
            )

        fake_completions = SimpleNamespace(create=AsyncMock(return_value=stream_with_tool()))
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=fake_completions))
        monkeypatch.setattr(
            "reachy_mini_conversation_app.cascade.llm.qwen._make_async_openai_client",
            lambda api_key, base_url: fake_client,
        )

        llm = QwenLLM(api_key="test")
        chunks = [chunk async for chunk in llm.generate([{"role": "user", "content": "hi"}], tools=[])]

        tool_chunks = [c for c in chunks if c.type == "tool_call"]
        assert len(tool_chunks) == 1
        assert tool_chunks[0].tool_call["function"]["name"] == "speak"
        assert tool_chunks[0].tool_call["function"]["arguments"] == '{"message":"hello"}'

    asyncio.run(run())


def test_qwen_llm_closes_stream_when_token_cancelled(monkeypatch):
    """QwenLLM closes the provider stream promptly after cancellation."""

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
                        delta=SimpleNamespace(content=" world", tool_calls=None),
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
            "reachy_mini_conversation_app.cascade.llm.qwen._make_async_openai_client",
            lambda api_key, base_url: fake_client,
        )

        token = TurnCancellationToken(turn_id=7)
        llm = QwenLLM(api_key="test")
        seen = []
        async for chunk in llm.generate([{"role": "user", "content": "hi"}], token=token):
            seen.append(chunk)
            token.cancel()

        assert [chunk.content for chunk in seen if chunk.content] == ["Hello"]
        assert stream.closed is True

    asyncio.run(run())
