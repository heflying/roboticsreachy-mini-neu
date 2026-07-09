"""Router abstraction for cascade pipeline — decides which LLM path to take."""

from __future__ import annotations
import abc
from typing import Any, Dict, List
from dataclasses import dataclass


@dataclass
class RouteResult:
    """Result of a routing decision."""

    decision: str
    reason: str = ""


class Router(abc.ABC):
    """Abstract base class for router providers."""

    @abc.abstractmethod
    async def route(self, messages: List[Dict[str, Any]], show_reason: bool = False) -> RouteResult:
        """Determine the routing decision for user input.

        Args:
            messages: Conversation history in OpenAI Chat Completions format, including the latest user message.
            show_reason: If True, the router will also populate the reason field in RouteResult.

        Returns:
            RouteResult with the routing decision.

        """
        raise NotImplementedError
