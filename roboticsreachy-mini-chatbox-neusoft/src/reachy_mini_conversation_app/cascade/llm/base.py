"""LLM abstraction for cascade pipeline."""

from __future__ import annotations
import inspect
import abc
import json
import logging
from typing import Any, Dict, List, Literal, Optional, AsyncIterator
from typing import TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from reachy_mini_conversation_app.cascade.interrupt_coordinator import TurnCancellationToken


logger = logging.getLogger(__name__)


@dataclass
class LLMChunk:
    """Represents a chunk from LLM streaming response."""

    type: Literal["text_delta", "tool_call", "done"]
    content: Optional[str] = None  # Text content for text_delta
    tool_call: Optional[Dict[str, Any]] = None  # Tool call data for tool_call type


class LLMProvider(abc.ABC):
    """Abstract base class for LLM providers."""

    @abc.abstractmethod
    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        token: TurnCancellationToken | None = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[LLMChunk]:
        """Generate streaming response from LLM.

        Args:
            messages: Conversation history in OpenAI format
            tools: Optional list of available tools
            temperature: Sampling temperature
            token: Optional turn cancellation token for early provider-side abort
            max_tokens: Optional maximum tokens to generate (useful for warmup)

        Yields:
            LLMChunk objects with text deltas, tool calls, or completion signal

        """
        raise NotImplementedError

    async def warmup(
        self,
        messages: List[Dict[str, Any]] | None = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 1.0,
    ) -> None:
        """Warm up the LLM provider with full context.

        Args:
            messages: Full conversation history for warmup. If None, only system message is sent.
            tools: Available tools
            temperature: Sampling temperature

        Note:
            Default implementation does nothing. Override in provider-specific
            implementations to enable warmup with conversation history and tools.
        """

    def parse_tool_call(self, tool_call: Dict[str, Any]) -> tuple[str, str, Dict[str, Any]]:
        """Parse a tool call into its components.

        Default implementation handles OpenAI-style tool call format.

        Args:
            tool_call: Tool call dictionary

        Returns:
            Tuple of (call_id, tool_name, arguments_dict)

        """
        call_id = tool_call.get("id", "")
        function_data = tool_call.get("function", {})
        tool_name = function_data.get("name", "")
        args_json = function_data.get("arguments", "{}")

        try:
            arguments = json.loads(args_json) if isinstance(args_json, str) else args_json
        except json.JSONDecodeError:
            logger.error(f"Failed to parse tool arguments: {args_json}")
            arguments = {}

        return call_id, tool_name, arguments


async def close_stream_resource(stream: Any) -> None:
    """Best-effort close for sync/async streaming resources."""
    if stream is None:
        return

    for method_name in ("aclose", "close"):
        close_method = getattr(stream, method_name, None)
        if not callable(close_method):
            continue
        try:
            result = close_method()
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            logger.debug("Ignoring %s() error while closing stream resource: %s", method_name, exc)
        return
