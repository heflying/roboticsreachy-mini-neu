"""OpenAI LLM implementation."""

from __future__ import annotations
import time
import asyncio
import base64
import logging
from typing import Any, Dict, List, Optional, AsyncIterator
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

from .base import LLMChunk, LLMProvider, close_stream_resource

if TYPE_CHECKING:
    from reachy_mini_conversation_app.cascade.interrupt_coordinator import TurnCancellationToken


logger = logging.getLogger(__name__)


class OpenAILLM(LLMProvider):
    """OpenAI GPT implementation for LLM."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: Optional[str] = None,
        system_instructions: Optional[str] = None,
        input_cost_per_1m: float = 0.0,
        output_cost_per_1m: float = 0.0
    ):
        """Initialize OpenAI LLM.

        Args:
            api_key: OpenAI API key
            model: Model name
            base_url: Base URL
            system_instructions: System prompt
            input_cost_per_1m: Cost per 1M input tokens (from cascade.yaml)
            output_cost_per_1m: Cost per 1M output tokens (from cascade.yaml)

        """
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.system_instructions = system_instructions
        self.input_cost_per_1m = input_cost_per_1m
        self.output_cost_per_1m = output_cost_per_1m
        self.last_cost: float = 0.0
        logger.info(f"Initialized OpenAI LLM with model: {model}")

    def _build_extra_create_chat_param(self) -> Optional[Dict[str, Any]]:
        """Build extra parameters for chat completion request.

        Override in subclasses to add provider-specific parameters
        (e.g. Qwen's enable_thinking).
        """
        return None

    def _convert_messages_for_openai(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert internal message format to OpenAI Chat Completions format.

        Handles image content which is stored as raw bytes internally
        but needs to be base64 data URLs for OpenAI.
        """
        converted = []
        for msg in messages:
            if "content" not in msg or not isinstance(msg["content"], list):
                converted.append(msg)
                continue

            new_content = []
            for part in msg["content"]:
                if isinstance(part, dict) and part.get("type") == "image":
                    # Convert raw bytes to OpenAI image_url format
                    image_bytes = part.get("image")
                    if image_bytes:
                        b64_str = base64.b64encode(image_bytes).decode("utf-8")
                        new_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}
                        })
                else:
                    new_content.append(part)

            converted.append({**msg, "content": new_content})
        return converted

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
        token: TurnCancellationToken | None = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[LLMChunk]:
        """Generate streaming response from OpenAI.

        Args:
            messages: Conversation history
            tools: Available tools
            temperature: Sampling temperature

        Yields:
            LLMChunk with text deltas or tool calls

        """
        from reachy_mini_conversation_app.cascade.timing import tracker

        # Prepend system message if provided
        full_messages = []
        if self.system_instructions:
            full_messages.append({"role": "system", "content": self.system_instructions})
        full_messages.extend(messages)

        # Convert internal image format to OpenAI format
        full_messages = self._convert_messages_for_openai(full_messages)

        logger.debug(f"Generating with {len(full_messages)} messages, {len(tools) if tools else 0} tools")

        try:
            # Build request parameters
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": full_messages,
                "temperature": temperature,
                "stream": True,
                "stream_options": {"include_usage": True},  # Enable usage in streaming
            }

            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens

            extra = self._build_extra_create_chat_param()
            if extra:
                kwargs.update(extra)

            # Stream response
            tracker.mark("llm_request_sending")
            request_start = time.perf_counter()

            if token and token.cancelled:
                logger.info("OpenAI generation skipped for cancelled turn %s", token.turn_id)
                return

            stream = await self.client.chat.completions.create(**kwargs)

            stream_open_ms = (time.perf_counter() - request_start) * 1000
            tracker.mark("llm_stream_opened", {"stream_open_ms": round(stream_open_ms, 1)})

            accumulated_text = ""
            accumulated_tool_calls: Dict[int, Dict[str, Any]] = {}
            usage_data: Any = None
            first_token = True
            chunk_count = 0

            stream_open_timestamp = time.perf_counter()
            try:
                async for chunk in stream:
                    if token and token.cancelled:
                        logger.info("OpenAI stream cancelled for turn %s", token.turn_id)
                        return

                    chunk_count += 1

                    # Capture usage data from final chunk (sent after finish_reason with empty choices)
                    if hasattr(chunk, "usage") and chunk.usage is not None:
                        usage_data = chunk.usage

                    if not chunk.choices:
                        continue

                    delta = chunk.choices[0].delta

                    # Track first token
                    if first_token and (delta.content or delta.tool_calls):
                        tracker.mark("llm_first_token")
                        first_token = False

                    # Handle text content
                    if delta.content:
                        accumulated_text += delta.content
                        yield LLMChunk(type="text_delta", content=delta.content)

                    # Handle tool calls
                    if delta.tool_calls:
                        for tool_call_delta in delta.tool_calls:
                            idx = tool_call_delta.index

                            # Initialize or update tool call accumulation
                            if idx not in accumulated_tool_calls:
                                accumulated_tool_calls[idx] = {
                                    "id": tool_call_delta.id or "",
                                    "type": "function",
                                    "function": {
                                        "name": tool_call_delta.function.name or "",
                                        "arguments": "",
                                    },
                                }

                            # Accumulate function arguments
                            if tool_call_delta.function.arguments:
                                accumulated_tool_calls[idx]["function"]["arguments"] += tool_call_delta.function.arguments

                            # Update name if provided
                            if tool_call_delta.function.name:
                                accumulated_tool_calls[idx]["function"]["name"] = tool_call_delta.function.name

                    # Yield accumulated tool calls when streaming content is done
                    if chunk.choices[0].finish_reason:
                        for tool_call in accumulated_tool_calls.values():
                            logger.info(f"Tool call: {tool_call['function']['name']}")
                            yield LLMChunk(type="tool_call", tool_call=tool_call)
            finally:
                await close_stream_resource(stream)

            if token and token.cancelled:
                logger.info("OpenAI generation aborted before completion for turn %s", token.turn_id)
                return

            cur_time = time.perf_counter()
            tracker.mark(
               "llm_complete",
                {
                    "text_len": len(accumulated_text),
                    "tool_calls": len(accumulated_tool_calls),
                    "chunks": chunk_count,
                    "total_ms": round((cur_time - request_start) * 1000, 1),
                    "stream_output_ms": round((cur_time - stream_open_timestamp) * 1000, 1),
                    "chars/sec": round(len(accumulated_text) / (cur_time - stream_open_timestamp), 1),
                    "chunks/sec": round(chunk_count / (cur_time - stream_open_timestamp), 1),
                },
            )

            # Calculate cost from usage data (after stream fully consumed)
            if usage_data and (self.input_cost_per_1m > 0 or self.output_cost_per_1m > 0):
                prompt_tokens = getattr(usage_data, "prompt_tokens", 0)
                completion_tokens = getattr(usage_data, "completion_tokens", 0)
                self.last_cost = (
                    prompt_tokens * self.input_cost_per_1m / 1e6
                    + completion_tokens * self.output_cost_per_1m / 1e6
                )
                logger.info(f"LLM Cost: ${self.last_cost:.6f} (in={prompt_tokens}, out={completion_tokens})")

            # Yield "done" after stream is fully consumed and cost is calculated
            yield LLMChunk(type="done")

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

    async def warmup(
        self,
        messages: List[Dict[str, Any]] | None = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 1.0,
    ) -> None:
        """Warm up the LLM by making a request with full context.

        This sends a complete request (including conversation history and tools)
        to trigger prompt processing and KV cache population, then discards
        the response after the first token.

        Args:
            messages: Full conversation history for warmup. If None, only system message is sent.
            tools: Available tools
            temperature: Sampling temperature

        Note:
            The warmup uses max_tokens=1 to minimize cost while still triggering
            the prompt processing. The generated token is discarded.
        """
        try:
            if messages is None:
                # Fallback to simple warmup — generate() will prepend system_instructions
                messages = []

            logger.info(f"Warming up LLM with {len(messages)} messages, {len(tools) if tools else 0} tools")
            async for chunk in self.generate(
                messages=messages,
                tools=tools,
                temperature=temperature,
                token=None,
                max_tokens=1,
            ):
                # Stop after first valid chunk
                if chunk.type in ("text_delta", "tool_call"):
                    break
            logger.info("LLM warmup successful")
        except asyncio.TimeoutError:
            logger.warning("LLM warmup timed out")
        except Exception as e:
            logger.warning("LLM warmup failed: %s", e)