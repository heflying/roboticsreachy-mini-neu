"""Composite router that combines multiple router strategies."""

from __future__ import annotations
import logging
from typing import Any, Dict, List

from reachy_mini_conversation_app.cascade.router.llm import LLMRouter
from reachy_mini_conversation_app.cascade.router.base import Router, RouteResult
from reachy_mini_conversation_app.cascade.router.bert import BertRouter


logger = logging.getLogger(__name__)


class CompositeRouter(Router):
    """Router that delegates to BertRouter, with LLMRouter kept initialized but unused."""

    def __init__(self) -> None:
        """Initialize CompositeRouter with BertRouter as primary and LLMRouter as reserved fallback."""
        # Initialize BertRouter as the primary router
        try:
            self._bert_privacy_router = BertRouter("models/Router/privacy/BaseBert", id2label={0: "no_privacy", 1: "privacy"})
        except Exception:
            logger.exception("Failed to initialize BertRouter, privacy routing will not use BERT")
            self._bert_privacy_router = None  # type: ignore[assignment]
        try:
            self._bert_complex_router = BertRouter("models/Router/complex/BaseBert", id2label={0: "simple", 1: "complex"})
        except Exception:
            logger.exception("Failed to initialize BertRouter, complex routing will not use BERT")
            self._bert_complex_router = None  # type: ignore[assignment]

        # Initialize LLMRouter — kept for future fallback but not currently used
        try:
            self._llm_router = LLMRouter()
        except Exception:
            logger.warning("Failed to initialize LLMRouter (currently unused, non-critical)", exc_info=True)
            self._llm_router = None  # type: ignore[assignment]

    async def route(self, messages: List[Dict[str, Any]], show_reason: bool = False) -> RouteResult:
        """Determine routing decision via BertRouter.

        LLMRouter is initialized but not currently delegated to.

        Args:
            messages: Conversation history in OpenAI Chat Completions format, including the latest user message.
            show_reason: If True, the router will also populate the reason field in RouteResult.

        Returns:
            RouteResult with the routing decision, or "unknown" if the router is unavailable.

        """
        if self._bert_privacy_router is None or self._bert_complex_router is None:
            return RouteResult(decision="unknown")
        privacy_result = await self._bert_privacy_router.route(messages, show_reason=show_reason)
        complex_result = await self._bert_complex_router.route(messages, show_reason=show_reason)
        result = RouteResult(decision=privacy_result.decision + " " + complex_result.decision,
                             reason=privacy_result.reason + " " + complex_result.reason)
        return result
