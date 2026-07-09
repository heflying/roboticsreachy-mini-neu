"""Router module — decides which LLM path to take based on user input."""

from reachy_mini_conversation_app.cascade.router.llm import LLMRouter
from reachy_mini_conversation_app.cascade.router.base import Router, RouteResult
from reachy_mini_conversation_app.cascade.router.bert import BertRouter
from reachy_mini_conversation_app.cascade.router.composite import CompositeRouter


__all__ = ["BertRouter", "CompositeRouter", "LLMRouter", "RouteResult", "Router"]
