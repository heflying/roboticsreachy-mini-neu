"""科大讯飞星火大模型 WebSocket LLM provider."""

from __future__ import annotations
import json
import logging
import asyncio
import time
from typing import Any, Dict, List, Optional, AsyncIterator
from typing import TYPE_CHECKING

from .base import LLMChunk, LLMProvider

if TYPE_CHECKING:
    from reachy_mini_conversation_app.cascade.interrupt_coordinator import TurnCancellationToken

logger = logging.getLogger(__name__)

# WebSocket endpoints for different Spark model versions
SPARK_ENDPOINTS = {
    "ultra": "wss://spark-api.xf-yun.com/v4.0/chat",
    "max-32k": "wss://spark-api.xf-yun.com/chat/max-32k",
    "max": "wss://spark-api.xf-yun.com/v3.5/chat",
    "pro-128k": "wss://spark-api.xf-yun.com/chat/pro-128k",
    "pro": "wss://spark-api.xf-yun.com/v3.1/chat",
    "lite": "wss://spark-api.xf-yun.com/v1.1/chat",
}

# Domain parameters for each version
SPARK_DOMAINS = {
    "ultra": "Ultra",
    "max-32k": "max-32k",
    "max": "generalv3.5",
    "pro-128k": "pro-128k",
    "pro": "generalv3",
    "lite": "lite",
}


def _connect_websocket(url: str) -> Any:
    """Connect to WebSocket lazily so tests can run without websockets installed."""
    import websockets

    return websockets.connect(url)


def _generate_auth_url(api_key: str, api_secret: str, base_url: str) -> str:
    """Generate authenticated WebSocket URL with signature.

    Spark API requires HMAC-SHA256 signature in URL query parameters.
    Reference: https://www.xfyun.cn/doc/spark/Web.html
    """
    import hashlib
    import hmac
    import base64
    from datetime import datetime, timezone
    from urllib.parse import urlparse, urlencode

    # Parse base URL
    parsed = urlparse(base_url)
    host = parsed.hostname or ""
    path = parsed.path or "/"

    # Generate RFC1123 timestamp
    now = datetime.now(timezone.utc)
    date = now.strftime("%a, %d %b %Y %H:%M:%S GMT")

    # Build signature origin string (must be exactly this format)
    signature_origin = f"host: {host}\ndate: {date}\nGET {path} HTTP/1.1"

    # HMAC-SHA256 signature
    signature_sha = hmac.new(
        api_secret.encode("utf-8"),
        signature_origin.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    signature_sha_base64 = base64.b64encode(signature_sha).decode("utf-8")

    # Build authorization origin string (no quotes around values except signature)
    authorization_origin = f"api_key=\"{api_key}\", algorithm=\"hmac-sha256\", headers=\"host date request-line\", signature=\"{signature_sha_base64}\""
    authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode("utf-8")

    # Build final URL - all three params must be in the URL for signature validation
    # The signature covers host, date, and request-line, so all must be present
    query_params = {
        "authorization": authorization,
        "date": date,
        "host": host,
    }
    return f"{base_url}?{urlencode(query_params)}"


class SparkLLM(LLMProvider):
    """科大讯飞星火大模型 WebSocket streaming LLM provider.

    Features:
    - WebSocket streaming for lower latency
    - Multi-turn conversation support
    - Tool call support
    """

    def __init__(
        self,
        api_key: str,
        api_secret: Optional[str] = None,
        app_id: Optional[str] = None,
        model: str = "ultra",
        system_instructions: Optional[str] = None,
        input_cost_per_1m: float = 0.0,
        output_cost_per_1m: float = 0.0,
    ) -> None:
        """Initialize Spark LLM provider.

        Args:
            api_key: Spark API key (from SPARK_API_KEY env var, format: "key:secret")
            api_secret: Spark API secret (extracted from api_key if format is "key:secret")
            app_id: Spark application ID (optional, for billing)
            model: Model version: "ultra", "max", "pro", "lite", etc.
            system_instructions: System prompt
            input_cost_per_1m: Input token cost per million
            output_cost_per_1m: Output token cost per million
        """
        # Parse API key format "key:secret" if not provided separately
        if api_secret is None and ":" in api_key:
            parts = api_key.split(":", 1)
            self._api_key = parts[0]
            self._api_secret = parts[1]
        else:
            self._api_key = api_key
            self._api_secret = api_secret or api_key

        # Spark API requires app_id and uid to be non-empty strings
        # If not provided, use API key as app_id (common pattern for Spark)
        self.app_id = app_id or self._api_key[:8] if self._api_key else "default"
        self.model = model
        self.system_instructions = system_instructions
        self.input_cost_per_1m = input_cost_per_1m
        self.output_cost_per_1m = output_cost_per_1m
        self.last_cost = 0.0

        # WebSocket state (must be initialized before _connect_ws is called)
        self._ws: Any | None = None
        self._ws_url: str | None = None

        logger.info("Initialized Spark LLM: model=%s, app_id=%s", model, self.app_id)

    def _get_endpoint(self) -> str:
        """Get WebSocket endpoint for current model."""
        endpoint = SPARK_ENDPOINTS.get(self.model)
        if not endpoint:
            available = ", ".join(SPARK_ENDPOINTS.keys())
            raise ValueError(f"Unknown Spark model '{self.model}'. Available: {available}")
        return endpoint

    def _get_domain(self) -> str:
        """Get domain parameter for current model."""
        domain = SPARK_DOMAINS.get(self.model)
        if not domain:
            raise ValueError(f"Unknown Spark model '{self.model}'")
        return domain

    async def _connect_ws(self, force_new: bool = False) -> Any:
        """Connect to WebSocket with authentication.

        Args:
            force_new: If True, always create new connection even if one exists.
                       Spark WebSocket doesn't support persistent connections,
                       so each generate() call needs a fresh connection.
        """
        # Spark WebSocket closes after idle, so we need fresh connection each time
        if force_new and self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        if self._ws is not None:
            return self._ws

        base_url = self._get_endpoint()
        auth_url = _generate_auth_url(self._api_key, self._api_secret, base_url)

        # Debug: log auth params (without full auth string for security)
        logger.info("Spark WebSocket connecting: host=%s, path=%s, api_key_len=%d, api_secret_len=%d",
                    base_url.split("//")[1].split("/")[0] if "//" in base_url else "unknown",
                    base_url.split("/")[-1] if "/" in base_url else "/",
                    len(self._api_key),
                    len(self._api_secret))
        logger.debug("Spark auth_url (truncated): %s...", auth_url[:100])
        self._ws = await _connect_websocket(auth_url)
        self._ws_url = auth_url
        logger.info("Spark WebSocket connected: model=%s", self.model)
        return self._ws

    async def _close_ws(self) -> None:
        """Close WebSocket connection."""
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception as e:
                logger.warning("Error closing Spark WebSocket: %s", e)
            self._ws = None
            self._ws_url = None

    def _build_request(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Build Spark WebSocket request payload."""
        # Convert messages to Spark format
        text_messages: List[Dict[str, Any]] = []

        # Add system instructions as first message if present
        if self.system_instructions:
            text_messages.append({"role": "system", "content": self.system_instructions})

        # Convert OpenAI format messages to Spark format
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Handle tool results
            if role == "tool":
                tool_name = msg.get("name", "")
                text_messages.append({
                    "role": "tool",
                    "content": content,
                    "name": tool_name,
                })
            # Handle assistant with tool calls
            elif role == "assistant" and "tool_calls" in msg:
                # Spark expects tool_calls in different format
                text_content = content or ""
                tool_calls = msg.get("tool_calls", [])
                text_messages.append({
                    "role": "assistant",
                    "content": text_content,
                    "tool_calls": tool_calls,
                })
            # Handle user with image (skip for now - Spark has different vision API)
            elif role == "user" and isinstance(content, list):
                # Extract text parts only
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                text_messages.append({"role": "user", "content": " ".join(text_parts)})
            else:
                text_messages.append({"role": role, "content": str(content)})

        # Build request payload
        # Spark API requires app_id and uid to be non-empty strings
        import uuid
        request: Dict[str, Any] = {
            "header": {
                "app_id": self.app_id,
                "uid": f"user_{uuid.uuid4().hex[:8]}",  # Generate unique user ID
            },
            "parameter": {
                "chat": {
                    "domain": self._get_domain(),
                    "temperature": temperature,
                    "max_tokens": max_tokens if max_tokens is not None else 2048,
                    "top_k": 5,
                },
            },
            "payload": {
                "message": {
                    "text": text_messages,
                },
            },
        }

        # Add tools if provided
        if tools:
            # Convert OpenAI tools format to Spark format
            spark_tools = []
            for tool in tools:
                if tool.get("type") == "function":
                    func = tool.get("function", {})
                    spark_tools.append({
                        "type": "function",
                        "function": {
                            "name": func.get("name", ""),
                            "description": func.get("description", ""),
                            "parameters": func.get("parameters", {}),
                        },
                    })
            request["payload"]["message"]["tools"] = spark_tools

        return request

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        token: TurnCancellationToken | None = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[LLMChunk]:
        """Generate streaming response from Spark via WebSocket.

        Note: Spark WebSocket requires a fresh connection for each request.
        Unlike HTTP APIs, the WebSocket is closed after each response completes.
        """
        from reachy_mini_conversation_app.cascade.timing import tracker

        request_start = time.perf_counter()
        tracker.mark("llm_request_sending")

        if token and token.cancelled:
            logger.info("Spark generation skipped for cancelled turn %s", token.turn_id)
            return

        # Build request
        request = self._build_request(messages, tools, temperature, max_tokens)

        # Connect WebSocket - force new connection each time (Spark doesn't support persistent WS)
        ws = await self._connect_ws(force_new=True)
        tracker.mark("llm_stream_opened", {"stream_open_ms": round((time.perf_counter() - request_start) * 1000, 1)})

        # Send request
        await ws.send(json.dumps(request))

        accumulated_text = ""
        accumulated_tool_calls: Dict[int, Dict[str, Any]] = {}
        usage_data: Dict[str, int] = {}
        first_token = True
        chunk_count = 0

        try:
            async for raw in ws:
                if token and token.cancelled:
                    logger.info("Spark stream cancelled for turn %s", token.turn_id)
                    return

                chunk_count += 1
                response = json.loads(raw) if isinstance(raw, str) else raw

                if not isinstance(response, dict):
                    continue

                # Debug: log raw response structure (first few chunks)
                if chunk_count <= 3:
                    logger.debug("Spark response chunk %d: %s", chunk_count, json.dumps(response, ensure_ascii=False)[:500])

                # Check response code
                header = response.get("header", {})
                code = header.get("code", 0)
                if code != 0:
                    error_msg = header.get("message", "Unknown error")
                    logger.error("Spark API error: code=%d, message=%s", code, error_msg)
                    raise RuntimeError(f"Spark API error: {error_msg}")

                # Process payload - Spark uses "payload.choices.text" format
                payload = response.get("payload", {})
                choices = payload.get("choices", {})

                # Spark Lite uses "text" array format, other versions may use "content" directly
                # Format 1: choices.text[0].content (Spark Lite)
                # Format 2: choices.content (other versions)
                text_array = choices.get("text", [])
                if text_array and isinstance(text_array, list):
                    # Extract from text array format
                    for text_item in text_array:
                        if isinstance(text_item, dict):
                            content = text_item.get("content", "")
                            if content:
                                if first_token:
                                    tracker.mark("llm_first_token")
                                    first_token = False
                                accumulated_text += content
                                yield LLMChunk(type="text_delta", content=content)
                else:
                    # Direct content format
                    content = choices.get("content", "")
                    if content:
                        if first_token:
                            tracker.mark("llm_first_token")
                            first_token = False
                        accumulated_text += content
                        yield LLMChunk(type="text_delta", content=content)

                status = choices.get("status", 0)

                # Extract tool calls (if present)
                tool_calls = choices.get("tool_calls", [])
                if tool_calls:
                    for tc in tool_calls:
                        idx = tc.get("index", 0)
                        if idx not in accumulated_tool_calls:
                            accumulated_tool_calls[idx] = {
                                "id": tc.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": tc.get("function", {}).get("name", ""),
                                    "arguments": tc.get("function", {}).get("arguments", ""),
                                },
                            }
                        else:
                            # Append to existing tool call
                            if tc.get("function", {}).get("arguments"):
                                accumulated_tool_calls[idx]["function"]["arguments"] += tc["function"]["arguments"]

                # Extract usage (only in final response)
                usage = payload.get("usage", {})
                if usage:
                    usage_data = usage

                # Check for completion (status 2 = last result)
                if status == 2:
                    # Yield tool calls if present
                    for tool_call in accumulated_tool_calls.values():
                        yield LLMChunk(type="tool_call", tool_call=tool_call)
                    break

        except asyncio.CancelledError:
            logger.info("Spark LLM generation cancelled")
            raise
        finally:
            # Close WebSocket after generation completes
            await self._close_ws()

        if token and token.cancelled:
            logger.info("Spark generation aborted before completion for turn %s", token.turn_id)
            return

        tracker.mark(
            "llm_complete",
            {
                "text_len": len(accumulated_text),
                "tool_calls": len(accumulated_tool_calls),
                "chunks": chunk_count,
                "total_ms": round((time.perf_counter() - request_start) * 1000, 1),
            },
        )

        # Calculate cost if pricing is configured
        if usage_data and (self.input_cost_per_1m > 0 or self.output_cost_per_1m > 0):
            prompt_tokens = usage_data.get("prompt_tokens", 0)
            completion_tokens = usage_data.get("completion_tokens", 0)
            self.last_cost = (
                prompt_tokens * self.input_cost_per_1m / 1e6
                + completion_tokens * self.output_cost_per_1m / 1e6
            )

        yield LLMChunk(type="done")

    async def warmup(
        self,
        messages: List[Dict[str, Any]] | None = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 1.0,
    ) -> None:
        """Warm up the LLM with full context (conversation history + tools).

        Args:
            messages: Full conversation history for warmup. If None, only system message is sent.
            tools: Available tools
            temperature: Sampling temperature

        Note:
            Spark WebSocket doesn't support persistent connections, so warmup
            creates a fresh connection and closes it immediately after.
        """
        try:
            if messages is None:
                messages = []

            logger.info(f"Warming up Spark with {len(messages)} messages, {len(tools) if tools else 0} tools")

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

            logger.info("Spark LLM warmup successful")
        except Exception as e:
            logger.warning("Spark LLM warmup failed: %s", e)
