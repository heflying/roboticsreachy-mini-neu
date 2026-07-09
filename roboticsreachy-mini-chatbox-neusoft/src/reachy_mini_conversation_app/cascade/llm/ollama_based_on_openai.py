"""Ollama LLM provider via local OpenAI-compatible API."""

from __future__ import annotations
import logging
from typing import Any, Dict, Optional

from .openai import OpenAILLM


logger = logging.getLogger(__name__)


class OllamaLLM(OpenAILLM):
    """Ollama chat model implementation using the local OpenAI-compatible endpoint."""

    def __init__(
        self,
        api_key: str = "",
        model: str = "qwen3:8b",
        base_url: str = "http://127.0.0.1:11434/v1",
        system_instructions: Optional[str] = None,
        input_cost_per_1m: float = 0.0,
        output_cost_per_1m: float = 0.0,
        enable_thinking: Optional[bool] = False
    ) -> None:
        """Initialize Ollama LLM provider."""
        super().__init__(
            api_key=api_key or "ollama",
            model=model,
            base_url=base_url,
            system_instructions=system_instructions,
            input_cost_per_1m=input_cost_per_1m,
            output_cost_per_1m=output_cost_per_1m,
        )
        self.enable_thinking = enable_thinking
        logger.info("Initialized Ollama LLM with model: %s", model)

    def _build_extra_create_chat_param(self) -> Optional[Dict[str, Any]]:
        """Add Ollama-specific think parameter."""
        if self.enable_thinking is not None:
            return {"extra_body": {
                "enable_thinking": self.enable_thinking,
                "think": self.enable_thinking,
                "thinking": {"type": "disabled" if self.enable_thinking else "enabled"}
            }}
        return None
