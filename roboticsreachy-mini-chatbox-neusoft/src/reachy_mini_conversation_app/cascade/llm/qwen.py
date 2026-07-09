"""Qwen LLM provider via Alibaba Cloud Model Studio OpenAI-compatible API."""

from __future__ import annotations
import logging
from typing import Any, Dict, Optional, override

from .openai import OpenAILLM


logger = logging.getLogger(__name__)


class QwenLLM(OpenAILLM):
    """Qwen chat model implementation using DashScope OpenAI-compatible mode."""

    def __init__(
        self,
        api_key: str,
        model: str = "qwen-plus",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        system_instructions: Optional[str] = None,
        input_cost_per_1m: float = 0.0,
        output_cost_per_1m: float = 0.0,
        enable_thinking: Optional[bool] = False,
    ) -> None:
        """Initialize Qwen LLM provider."""
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            system_instructions=system_instructions,
            input_cost_per_1m=input_cost_per_1m,
            output_cost_per_1m=output_cost_per_1m,
        )
        self.enable_thinking = enable_thinking
        logger.info("Initialized Qwen LLM with model: %s", model)

    @override
    def _build_extra_create_chat_param(self) -> Optional[Dict[str, Any]]:
        """Add Qwen-specific extra_body for thinking mode."""
        if self.enable_thinking is not None:
            return {"extra_body": {"enable_thinking": self.enable_thinking}}
        return None
