"""Ollama LLM provider via native /api/chat endpoint.

Uses Ollama's native API (not OpenAI-compatible) to support the `think`
parameter for controlling thinking mode output. Tool calling uses Ollama's
native ``payload["tools"]`` parameter — Ollama returns structured
``tool_calls`` in the response message, no inline JSON parsing needed.

API docs: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-response
"""

from __future__ import annotations

import base64
import json
import logging
import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional

import aiohttp

from .base import LLMChunk, LLMProvider, close_stream_resource


logger = logging.getLogger(__name__)

# Ollama default base URL
DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaLLM(LLMProvider):
    """Ollama LLM implementation using the native /api/chat endpoint.

    Supports `think` parameter for thinking mode and uses Ollama's native
    tool calling via ``payload["tools"]`` (no prompt-based tool injection).
    """

    def __init__(
        self,
        model: str = "qwen2.5:0.5b",
        base_url: str = DEFAULT_BASE_URL,
        system_instructions: Optional[str] = None,
        input_cost_per_1m: float = 0.0,
        output_cost_per_1m: float = 0.0,
        think: Optional[bool] = False,
    ) -> None:
        """Initialize Ollama LLM.

        Args:
            model: Ollama model name (e.g. "qwen2.5:0.5b", "qwen3:1.7b").
            base_url: Ollama server URL (default: http://localhost:11434).
            system_instructions: System prompt.
            input_cost_per_1m: Cost per 1M input tokens (always 0 for local).
            output_cost_per_1m: Cost per 1M output tokens (always 0 for local).
            think: Enable thinking mode output. Set False to disable thinking.
                  Only effective for models that support thinking (e.g. Qwen3).
                  None = don't send the parameter (use Ollama default).
        """
        self.model = model
        # Strip /v1 suffix if present (OpenAI-compatible → native Ollama)
        self.base_url = base_url.replace("/v1", "").rstrip("/")
        self.system_instructions = system_instructions
        self.input_cost_per_1m = input_cost_per_1m
        self.output_cost_per_1m = output_cost_per_1m
        self.think = think
        self.last_cost: float = 0.0
        self._session: Optional[aiohttp.ClientSession] = None
        logger.info(
            "Initialized OllamaLLM: model=%s, think=%s", model, think
        )

    def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=120),
            )
        return self._session

    async def _close_session(self) -> None:
        """Close aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def _build_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build Ollama-format messages from internal OpenAI-format messages.

        Tool calls in assistant messages are preserved as Ollama's native
        ``tool_calls`` field, and tool messages use Ollama's ``tool_name``
        + content format.
        """
        ollama_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")

            # Handle tool messages (Ollama uses `tool_name` + content)
            if role == "tool":
                tool_name = msg.get("name", "")
                ollama_messages.append({
                    "role": "tool",
                    "tool_name": tool_name,
                    "content": content or "",
                })
                continue

            # Handle assistant with tool calls — preserve as native format
            if role == "assistant" and msg.get("tool_calls"):
                assistant_msg = {"role": "assistant"}
                assistant_msg["content"] = content or ""
                # Convert internal format to Ollama native tool_calls format
                ollama_tool_calls = []
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    ollama_tc = {
                        "function": {
                            "name": fn.get("name", ""),
                            "arguments": fn.get("arguments", {}),
                        },
                    }
                    if tc.get("id"):
                        ollama_tc["id"] = tc["id"]
                    ollama_tool_calls.append(ollama_tc)
                assistant_msg["tool_calls"] = ollama_tool_calls
                ollama_messages.append(assistant_msg)
                continue

            # Normal message — extract text from content list if needed
            if isinstance(content, list):
                text_parts = [
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                content = " ".join(text_parts)

            ollama_messages.append({"role": role, "content": content or ""})

        return ollama_messages

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
        token: Any = None,
        max_tokens: Optional[int] = None,
        images: Optional[List[str]] = None,
    ) -> AsyncIterator[LLMChunk]:
        """Generate streaming response from Ollama via native /api/chat.

        Args:
            messages: Conversation history.
            tools: Available tools (OpenAI-format tool definitions).
            temperature: Sampling temperature.
            token: Turn cancellation token.
            max_tokens: Max tokens.
            images: Optional list of image file paths to attach to the last user
                message. Each file is read and base64-encoded for Ollama's
                multimodal /api/chat endpoint (supports qwen-vl, gemma3, etc.).

        Yields:
            LLMChunk with text deltas or tool calls.
        """
        from reachy_mini_conversation_app.cascade.timing import tracker

        # Build Ollama-format messages
        ollama_messages = self._build_messages(messages)
        system_content = self.system_instructions or ""
        if system_content:
            ollama_messages.insert(0, {"role": "system", "content": system_content})

        # Attach images to the last user message (Ollama multimodal support)
        if images:
            # Read and base64-encode all image files
            encoded_images: List[str] = []
            for img_path in images:
                try:
                    with open(img_path, "rb") as f:
                        encoded_images.append(base64.b64encode(f.read()).decode("utf-8"))
                except FileNotFoundError:
                    raise FileNotFoundError(f"Image file not found: {img_path}")

            # Find the last user message to attach images to
            last_user_msg = None
            for msg in reversed(ollama_messages):
                if msg.get("role") == "user":
                    last_user_msg = msg
                    break

            if last_user_msg is None:
                raise ValueError("Cannot attach images: no user message found in conversation")

            last_user_msg["images"] = encoded_images
            logger.debug("Attached %d image(s) to last user message", len(encoded_images))

        # Build request payload
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                # Context window size in tokens. Native tool calling mode adds
                # an internal tools prompt that can be thousands of tokens long
                # (not counted in prompt_eval_count). Without expanding num_ctx,
                # the model may run out of context and truncate the response
                # (done_reason="length") even with plenty of user-side tokens.
                # 8192 is the default for most small models; 16384 gives enough
                # headroom for tools + conversation history + output.
                "num_ctx": 16384,
                # Maximum number of tokens the model can generate in response.
                # -1 means unlimited (up to context window).
                "num_predict": -1,
            },
        }

        # Pass tools via payload["tools"] so Ollama handles tool calling natively.
        # Ollama returns structured tool_calls in the response message instead of
        # inline JSON in text content.
        if tools:
            ollama_tools = []
            for spec in tools:
                func = spec.get("function", spec)
                ollama_tools.append({
                    "type": "function",
                    "function": {
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "parameters": func.get("parameters", {}),
                    },
                })
            payload["tools"] = ollama_tools

        # Add think parameter (controls thinking mode output)
        if self.think is not None:
            payload["think"] = self.think

        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        logger.debug(
            "Ollama request: model=%s, messages=%d, think=%s, tools=%d",
            self.model,
            len(ollama_messages),
            self.think,
            len(tools) if tools else 0,
        )


        tracker.mark("llm_request_sending")

        try:
            session = self._get_session()
            async with session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(
                        f"Ollama API error {resp.status}: {error_text}"
                    )

                tracker.mark("llm_stream_opened", {"stream_open_ms": 0})

                accumulated_text = ""
                tool_calls_found: list[Dict[str, Any]] = []
                first_token = True
                chunk_count = 0
                eval_count = 0
                eval_duration_ns = 0
                stream_open_timestamp = asyncio.get_event_loop().time()

                # Read NDJSON stream (newline-delimited JSON)
                async for line in resp.content:
                    if token and getattr(token, "cancelled", False):
                        logger.info("Ollama generation cancelled")
                        break

                    line = line.strip()
                    if not line:
                        continue

                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse Ollama chunk: %s", line[:100])
                        continue

                    chunk_count += 1

                    # Extract content from Ollama response
                    message = chunk.get("message", {})
                    content = message.get("content", "")

                    # Also check for thinking content (if think=True)
                    thinking = chunk.get("thinking", "")

                    # ── Native tool calls ────────────────────────────
                    # Ollama returns structured tool_calls in message["tool_calls"]
                    # instead of inline text JSON. Yield them immediately.
                    native_tool_calls = message.get("tool_calls")
                    if native_tool_calls:
                        if first_token:
                            tracker.mark("llm_first_token")
                            first_token = False
                        for tc in native_tool_calls:
                            func_data = tc.get("function", {})
                            tool_call_chunk = {
                                "type": "function",
                                "function": {
                                    "name": func_data.get("name", ""),
                                    "arguments": func_data.get("arguments", {}),
                                },
                            }
                            tool_calls_found.append(tool_call_chunk)
                            yield LLMChunk(type="tool_call", tool_call=tool_call_chunk)
                        # When tool_calls are present, content is empty
                        content = ""

                    if first_token and (content or thinking):
                        tracker.mark("llm_first_token")
                        first_token = False

                    # Yield thinking content first (if present)
                    if thinking:
                        yield LLMChunk(type="text_delta", content=thinking)

                    # Yield text content directly (no JSON buffer needed in native mode)
                    if content:
                        accumulated_text += content
                        yield LLMChunk(type="text_delta", content=content)

                    # Check if done — capture usage stats from final chunk
                    if chunk.get("done", False):
                        eval_count = chunk.get("eval_count", 0)
                        eval_duration_ns = chunk.get("eval_duration", 0)
                        break

                # Calculate and log performance stats
                cur_time = asyncio.get_event_loop().time()
                elapsed = cur_time - stream_open_timestamp
                if elapsed > 0:
                    chars_per_sec = round(len(accumulated_text) / elapsed, 1)
                    chunks_per_sec = round(chunk_count / elapsed, 1)
                    tokens_per_sec = (
                        round(eval_count / (eval_duration_ns / 1e9), 1)
                        if eval_duration_ns > 0
                        else round(eval_count / elapsed, 1)
                    )
                else:
                    chars_per_sec = chunks_per_sec = tokens_per_sec = 0.0

                tracker.mark(
                    "llm_complete",
                    {
                        "text_len": len(accumulated_text),
                        "tool_calls": len(tool_calls_found),
                        "chunks": chunk_count,
                        "total_ms": round(elapsed * 1000, 1),
                        "chars/sec": chars_per_sec,
                        "chunks/sec": chunks_per_sec,
                        "tokens/sec": tokens_per_sec,
                        "eval_count": eval_count,
                        "eval_duration_ms": round(eval_duration_ns / 1e6, 1) if eval_duration_ns else 0,
                    },
                )
                logger.info(
                    "Ollama stats: chars/sec=%.1f, chunks/sec=%.1f, tokens/sec=%.1f (eval_count=%d, eval_duration=%.0fms)",
                    chars_per_sec,
                    chunks_per_sec,
                    tokens_per_sec,
                    eval_count,
                    eval_duration_ns / 1e6 if eval_duration_ns else 0,
                )

                # Calculate cost (always 0 for local Ollama)
                self.last_cost = 0.0

                yield LLMChunk(type="done")

        except aiohttp.ClientError as e:
            logger.error("Ollama connection error: %s", e)
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Make sure Ollama is running (ollama serve)."
            ) from e
        except Exception as e:
            logger.error("Ollama generation failed: %s", e)
            raise

    async def warmup(
        self,
        messages: List[Dict[str, Any]] | None = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 1.0,
    ) -> None:
        """Warm up Ollama by making a minimal request with tools in system prompt.

        Ollama loads model into memory on first request. Tools are embedded
        in the system prompt to warm up the KV cache prefix as well.
        """
        try:
            logger.info("Warming up Ollama with model: %s", self.model)
            async for chunk in self.generate(
                messages=messages or [],
                tools=tools,
                temperature=temperature,
                max_tokens=1,
            ):
                if chunk.type in ("text_delta", "tool_call"):
                    break
            logger.info("Ollama warmup successful")
        except Exception as e:
            logger.warning("Ollama warmup failed: %s", e)

    def parse_tool_call(self, tool_call: Dict[str, Any]) -> tuple[str, str, Dict[str, Any]]:
        """Parse an Ollama-format tool call into (call_id, tool_name, arguments_dict).

        Ollama tool_calls have ``function.name`` and ``function.arguments``
        where ``arguments`` is already a parsed dict (unlike OpenAI's JSON string).
        """
        call_id = tool_call.get("id", "")
        function_data = tool_call.get("function", {})
        tool_name = function_data.get("name", "")

        args = function_data.get("arguments", {})
        if isinstance(args, str):
            try:
                arguments: Dict[str, Any] = json.loads(args)
            except json.JSONDecodeError:
                logger.error("Failed to parse tool arguments: %s", args)
                arguments = {}
        elif isinstance(args, dict):
            arguments = args
        else:
            arguments = {}

        return call_id, tool_name, arguments

    async def close(self) -> None:
        """Close the HTTP session."""
        await self._close_session()
