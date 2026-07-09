"""Tests for direct streaming dialog output."""

from __future__ import annotations

import asyncio
import importlib
import sys
import types
from typing import Any, AsyncIterator

from reachy_mini_conversation_app.cascade.interrupt_coordinator import TurnCancellationToken
from reachy_mini_conversation_app.cascade.llm.base import LLMChunk, LLMProvider
from reachy_mini_conversation_app.cascade.streaming_text import SentenceChunker
from reachy_mini_conversation_app.cascade.turn_result import PipelineResult


HELLO = "\u4F60\u597D"
I_AM = "\u6211\u662F"
SMALL_ROBOT = "\u5C0F\u673A\u5668\u4EBA"
NEXT = "\u540E\u9762"
WHO_ARE_YOU = "\u4F60\u662F\u8C01\uFF1F"


class _FakeLLM(LLMProvider):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.last_cost = 0.0
        self.system_instructions = "system"

    async def generate(self, messages, tools=None, temperature=0.7, token=None):
        self.calls.append({"messages": messages, "tools": tools, "temperature": temperature, "token": token})
        yield LLMChunk(type="text_delta", content=f"{HELLO}\uFF0C")
        yield LLMChunk(type="text_delta", content=f"{I_AM} Reachy Mini\u3002")
        yield LLMChunk(type="done")


class _FakeSpeechOutput:
    def __init__(self) -> None:
        self.streamed_chunks: list[str] = []

    async def speak(self, text: str, token: Any = None, turn_id: int = 0) -> None:
        self.streamed_chunks.append(text)

    async def speak_stream(
        self,
        text_chunks: AsyncIterator[str],
        token: Any = None,
        turn_id: int = 0,
    ) -> str:
        full_text = ""
        async for chunk in text_chunks:
            self.streamed_chunks.append(chunk)
            full_text += chunk
        return full_text


class _CancellingSpeechOutput(_FakeSpeechOutput):
    async def speak(self, text: str, token: Any = None, turn_id: int = 0) -> None:
        self.streamed_chunks.append(text)
        if token is not None:
            token.cancel()


def test_sentence_chunker_flushes_chinese_segments():
    chunker = SentenceChunker(min_chars=4, max_chars=12)

    assert chunker.push(f"{HELLO}\uFF0C{I_AM}") == []
    assert chunker.push(f"{SMALL_ROBOT}\u3002{NEXT}") == [f"{HELLO}\uFF0C{I_AM}{SMALL_ROBOT}\u3002"]
    assert chunker.flush() == NEXT


def test_streaming_dialog_response_bypasses_speak_tool():
    async def run():
        fake_config_module = types.ModuleType("reachy_mini_conversation_app.cascade.config")
        fake_config_module.get_config = lambda: type("C", (), {"llm_temperature": 0.1})()
        fake_tools_module = types.ModuleType("reachy_mini_conversation_app.tools.core_tools")
        fake_tools_module.ToolDependencies = object

        async def fake_dispatch_tool_call(*args, **kwargs):
            return {}

        fake_tools_module.dispatch_tool_call = fake_dispatch_tool_call

        saved_modules = {
            name: sys.modules.get(name)
            for name in (
                "reachy_mini_conversation_app.cascade.config",
                "reachy_mini_conversation_app.tools.core_tools",
                "reachy_mini_conversation_app.cascade.pipeline",
            )
        }
        sys.modules["reachy_mini_conversation_app.cascade.config"] = fake_config_module
        sys.modules["reachy_mini_conversation_app.tools.core_tools"] = fake_tools_module
        sys.modules.pop("reachy_mini_conversation_app.cascade.pipeline", None)
        pipeline = importlib.import_module("reachy_mini_conversation_app.cascade.pipeline")

        fake_llm = _FakeLLM()
        fake_speech = _FakeSpeechOutput()
        history = [{"role": "user", "content": "\u8BF7\u8BF4\u4E00\u53E5\u95EE\u5019"}]
        ctx = pipeline.PipelineContext(
            llm=fake_llm,
            tts=None,  # type: ignore[arg-type]
            speech_output=fake_speech,
            conversation_history=history,
            tool_specs=[{"type": "function", "function": {"name": "speak"}}],
            deps=None,  # type: ignore[arg-type]
            result=PipelineResult(),
        )

        try:
            result = await pipeline.process_streaming_dialog_response(ctx)
        finally:
            for name, module in saved_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

        expected_text = f"{HELLO}\uFF0C{I_AM} Reachy Mini\u3002"
        assert fake_llm.calls[0]["tools"] is None
        assert fake_speech.streamed_chunks == [f"{HELLO}\uFF0C", f"{I_AM} Reachy Mini\u3002"]
        assert history[-1] == {"role": "assistant", "content": expected_text}
        assert result.turn_items[0].kind == "speak"
        assert result.turn_items[0].text == expected_text

    asyncio.run(run())


def test_streaming_dialog_uses_quick_reply_for_simple_identity_question():
    async def run():
        fake_config_module = types.ModuleType("reachy_mini_conversation_app.cascade.config")
        fake_config_module.get_config = lambda: type("C", (), {"llm_temperature": 0.1})()
        fake_tools_module = types.ModuleType("reachy_mini_conversation_app.tools.core_tools")
        fake_tools_module.ToolDependencies = object

        async def fake_dispatch_tool_call(*args, **kwargs):
            return {}

        fake_tools_module.dispatch_tool_call = fake_dispatch_tool_call

        saved_modules = {
            name: sys.modules.get(name)
            for name in (
                "reachy_mini_conversation_app.cascade.config",
                "reachy_mini_conversation_app.tools.core_tools",
                "reachy_mini_conversation_app.cascade.pipeline",
            )
        }
        sys.modules["reachy_mini_conversation_app.cascade.config"] = fake_config_module
        sys.modules["reachy_mini_conversation_app.tools.core_tools"] = fake_tools_module
        sys.modules.pop("reachy_mini_conversation_app.cascade.pipeline", None)
        pipeline = importlib.import_module("reachy_mini_conversation_app.cascade.pipeline")

        fake_llm = _FakeLLM()
        fake_speech = _FakeSpeechOutput()
        history = [{"role": "user", "content": WHO_ARE_YOU}]
        ctx = pipeline.PipelineContext(
            llm=fake_llm,
            tts=None,  # type: ignore[arg-type]
            speech_output=fake_speech,
            conversation_history=history,
            tool_specs=[{"type": "function", "function": {"name": "speak"}}],
            deps=None,  # type: ignore[arg-type]
            result=PipelineResult(),
        )

        try:
            result = await pipeline.process_streaming_dialog_response(ctx)
        finally:
            for name, module in saved_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

        assert fake_llm.calls == []
        assert fake_speech.streamed_chunks == ["我是 Reachy Mini，可以和你语音聊天的小机器人。"]
        assert history[-1]["role"] == "assistant"
        assert result.turn_items[0].text == "我是 Reachy Mini，可以和你语音聊天的小机器人。"

    asyncio.run(run())


def test_cancelled_speak_rolls_back_partial_history():
    async def run():
        fake_config_module = types.ModuleType("reachy_mini_conversation_app.cascade.config")
        fake_config_module.get_config = lambda: type("C", (), {"llm_temperature": 0.1})()
        fake_tools_module = types.ModuleType("reachy_mini_conversation_app.tools.core_tools")
        fake_tools_module.ToolDependencies = object

        async def fake_dispatch_tool_call(*args, **kwargs):
            return {"message": "从前有一只小机器人。"}

        fake_tools_module.dispatch_tool_call = fake_dispatch_tool_call

        saved_modules = {
            name: sys.modules.get(name)
            for name in (
                "reachy_mini_conversation_app.cascade.config",
                "reachy_mini_conversation_app.tools.core_tools",
                "reachy_mini_conversation_app.cascade.pipeline",
            )
        }
        sys.modules["reachy_mini_conversation_app.cascade.config"] = fake_config_module
        sys.modules["reachy_mini_conversation_app.tools.core_tools"] = fake_tools_module
        sys.modules.pop("reachy_mini_conversation_app.cascade.pipeline", None)
        pipeline = importlib.import_module("reachy_mini_conversation_app.cascade.pipeline")

        class _ToolCallLLM(LLMProvider):
            last_cost = 0.0
            system_instructions = "system"

            async def generate(self, messages, tools=None, temperature=0.7, token=None):
                yield LLMChunk(
                    type="tool_call",
                    tool_call={
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "speak", "arguments": '{"message":"从前有一只小机器人。"}'},
                    },
                )
                yield LLMChunk(type="done")

        token = TurnCancellationToken(turn_id=1)
        history = [{"role": "user", "content": "讲一个故事"}]
        ctx = pipeline.PipelineContext(
            llm=_ToolCallLLM(),
            tts=None,  # type: ignore[arg-type]
            speech_output=_CancellingSpeechOutput(),
            conversation_history=history,
            tool_specs=[{"type": "function", "function": {"name": "speak"}}],
            deps=None,  # type: ignore[arg-type]
            result=PipelineResult(),
            token=token,
            turn_id=1,
        )

        try:
            await pipeline.process_llm_response(ctx)
        finally:
            for name, module in saved_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

        assert history == [{"role": "user", "content": "讲一个故事"}]
        assert ctx.result.turn_items == []

    asyncio.run(run())
