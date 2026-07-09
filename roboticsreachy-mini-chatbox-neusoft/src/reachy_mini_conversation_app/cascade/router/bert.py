"""BERT-based router implementation — uses a fine-tuned BERT classifier to classify user input."""

from __future__ import annotations
import asyncio
import logging
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from reachy_mini_conversation_app.cascade.router.base import Router, RouteResult


logger = logging.getLogger(__name__)


class BertRouter(Router):
    """Router that uses a fine-tuned BERT model to classify user input."""

    def __init__(self, model_path: str, id2label: Dict[int, str]) -> None:
        """Initialize BertRouter with a BERT sequence classification model.

        Args:
            model_path: Path to the directory containing the safetensors model
                        and tokenizer files.
            id2label: Mapping from class id to decision string,
                      e.g. {0: "no_privacy", 1: "privacy"}.

        """
        if not model_path:
            raise ValueError("BertRouter requires a non-empty model_path.")

        logger.info("BertRouter loading model from: %s", model_path)

        self._id2label = id2label

        self._tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        self._model.eval()
        self._model.to("cpu")

        logger.info("BertRouter model loaded successfully (device=cpu)")

        # Warmup: run a dummy inference to trigger lazy initialization
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._warmup())
        except RuntimeError:
            try:
                asyncio.run(self._warmup())
            except RuntimeError:
                logger.debug("Cannot run warmup: event loop conflict")

    async def _warmup(self) -> None:
        """Warm up the BERT model with a dummy inference to avoid first-call latency."""
        try:
            dummy_text = "你好"
            inputs = self._tokenizer(dummy_text, return_tensors="pt", truncation=True, max_length=512)
            await asyncio.to_thread(self._model, **dict(inputs.items()))
            logger.info("BertRouter warmup completed")
        except Exception:
            logger.warning("BertRouter warmup failed", exc_info=True)

    def _extract_last_user_message(self, messages: List[Dict[str, Any]]) -> str:
        """Extract the last user message from conversation history.

        Args:
            messages: Conversation history in OpenAI Chat Completions format.

        Returns:
            The content of the last user message, or empty string if not found.

        """
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                # Handle list-style content (e.g. multimodal messages)
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return str(part.get("text", ""))
        return ""

    async def route(self, messages: List[Dict[str, Any]], show_reason: bool = False) -> RouteResult:
        """Determine routing decision using BERT classification.

        Extracts the last user message and runs it through the BERT model
        for classification based on the id2label mapping.

        Args:
            messages: Conversation history in OpenAI Chat Completions format,
                      including the latest user message.
            show_reason: If True, populate reason with raw logits/probabilities.

        Returns:
            RouteResult with decision mapped from id2label, or "unknown".

        """
        text = self._extract_last_user_message(messages)
        if not text:
            return RouteResult(decision="unknown")

        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            outputs = await asyncio.to_thread(self._model, **dict(inputs.items()))

        logits = outputs.logits
        predicted_class_id = int(torch.argmax(logits, dim=-1).item())

        decision = self._id2label.get(predicted_class_id, "unknown")

        reason = ""
        if show_reason:
            probs = torch.softmax(logits, dim=-1)
            predicted_prob = probs[0][predicted_class_id].item()
            reason = f"probability={predicted_prob:.4f}"

        logger.info("BertRouter decision: %s (predicted_class_id=%d)", decision, predicted_class_id)
        return RouteResult(decision=decision, reason=reason)
