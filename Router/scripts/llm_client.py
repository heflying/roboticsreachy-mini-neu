"""
scripts/llm_client.py

Synchronous LLM client supporting Ollama, Qwen (DashScope), and Spark (讯飞星火 HTTP API).
Reads API keys and model names from .env file via python-dotenv.
"""

from __future__ import annotations

import hashlib
import hmac
import base64
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse, urlencode

import requests
from openai import OpenAI
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env from project root
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_project_root, ".env"))


class LLMClient:
    """Synchronous LLM client with multi-backend support."""

    def __init__(self, backend: str = "ollama", **overrides):
        """Initialize LLM client.

        Args:
            backend: One of 'ollama', 'qwen', 'spark'.
            **overrides: Override any env-var-derived setting.
        """
        self.backend = backend.lower()
        if self.backend not in ("ollama", "qwen", "spark"):
            raise ValueError(f"Unknown backend '{backend}'. Supported: ollama, qwen, spark")
        self._init_backend(**overrides)

    # ------------------------------------------------------------------
    # Backend initialization
    # ------------------------------------------------------------------

    def _init_backend(self, **overrides):
        if self.backend == "ollama":
            self._init_ollama(**overrides)
        elif self.backend == "qwen":
            self._init_qwen(**overrides)
        elif self.backend == "spark":
            self._init_spark(**overrides)

    def _init_ollama(self, **overrides):
        self.api_key = overrides.get("api_key", os.getenv("OLLAMA_API_KEY", "ollama"))
        self.model = overrides.get("model", os.getenv("OLLAMA_MODEL", "qwen2.5-1.5b-instruct"))
        self.base_url = overrides.get("base_url", os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1"))
        self._openai_client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _init_qwen(self, **overrides):
        self.api_key = overrides.get("api_key", os.getenv("QWEN_API_KEY", ""))
        if not self.api_key:
            raise ValueError("QWEN_API_KEY is required for qwen backend. Set it in .env or pass api_key=.")
        self.model = overrides.get("model", os.getenv("QWEN_MODEL", "qwen-plus"))
        self.base_url = overrides.get("base_url", os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
        self._openai_client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _init_spark(self, **overrides):
        self.model = overrides.get("model", os.getenv("SPARK_MODEL", "lite"))
        # HTTP API uses Bearer APIPassword (from console), not HMAC
        self.api_key = overrides.get("api_key", os.getenv("SPARK_API_PASSWORD", os.getenv("SPARK_API_KEY", "")))
        if not self.api_key:
            raise ValueError("SPARK_API_PASSWORD (or SPARK_API_KEY) is required for spark backend.")
        self.base_url = overrides.get("base_url", os.getenv("SPARK_BASE_URL", "https://spark-api-open.xf-yun.com/v1"))
        self._openai_client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    # ------------------------------------------------------------------
    # Spark HTTP endpoints and domain mapping
    # ------------------------------------------------------------------

    SPARK_HTTP_URLS = {
        "ultra": "https://spark-api.xf-yun.com/v4.0/chat",
        "max-32k": "https://spark-api.xf-yun.com/chat/max-32k",
        "max": "https://spark-api.xf-yun.com/v3.5/chat",
        "pro-128k": "https://spark-api.xf-yun.com/chat/pro-128k",
        "pro": "https://spark-api.xf-yun.com/v3.1/chat",
        "lite": "https://spark-api.xf-yun.com/v1.1/chat",
        "generalv3.5": "https://spark-api.xf-yun.com/v3.5/chat",
        "generalv3": "https://spark-api.xf-yun.com/v3.1/chat",
        "4.0Ultra": "https://spark-api.xf-yun.com/v4.0/chat",
    }

    # lite uses Bearer token auth; others use HMAC signature
    SPARK_BEARER_MODELS = {"lite"}

    SPARK_DOMAINS = {
        "ultra": "Ultra",
        "max-32k": "max-32k",
        "max": "generalv3.5",
        "pro-128k": "pro-128k",
        "pro": "generalv3",
        "lite": "lite",
    }

    def _generate_spark_auth_url(self, base_url: str) -> str:
        """Generate authenticated URL with HMAC-SHA256 signature for Spark HTTP API."""
        parsed = urlparse(base_url)
        host = parsed.hostname or ""
        path = parsed.path or "/"

        now = datetime.now(timezone.utc)
        date = now.strftime("%a, %d %b %Y %H:%M:%S GMT")

        signature_origin = f"host: {host}\ndate: {date}\nGET {path} HTTP/1.1"
        signature_sha = hmac.new(
            self._spark_api_secret.encode("utf-8"),
            signature_origin.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        signature_b64 = base64.b64encode(signature_sha).decode("utf-8")

        authorization_origin = (
            f'api_key="{self._spark_api_key}", algorithm="hmac-sha256", '
            f'headers="host date request-line", signature="{signature_b64}"'
        )
        authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode("utf-8")

        params = {"authorization": authorization, "date": date, "host": host}
        return f"{base_url}?{urlencode(params)}"

    # ------------------------------------------------------------------
    # Generate (sync)
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Send a prompt and return the full text response synchronously.

        Args:
            prompt: User message text.
            system_prompt: Optional system message.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            The full text content of the LLM response.
        """
        if self.backend in ("ollama", "qwen", "spark"):
            return self._generate_openai(prompt, system_prompt, temperature, max_tokens)
        raise ValueError(f"Unknown backend: {self.backend}")

    def _generate_openai(self, prompt, system_prompt, temperature, max_tokens):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""

    def _generate_spark(self, prompt, system_prompt, temperature, max_tokens):
        base_url = self.SPARK_HTTP_URLS.get(self.model)
        if not base_url:
            available = ", ".join(self.SPARK_HTTP_URLS.keys())
            raise ValueError(f"Unknown Spark model '{self.model}'. Available: {available}")

        domain = self.SPARK_DOMAINS.get(self.model, self.model)

        text_messages = []
        if system_prompt:
            text_messages.append({"role": "system", "content": system_prompt})
        text_messages.append({"role": "user", "content": prompt})

        import uuid
        payload = {
            "header": {
                "app_id": self._spark_app_id,
                "uid": f"user_{uuid.uuid4().hex[:8]}",
            },
            "parameter": {
                "chat": {
                    "domain": domain,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            },
            "payload": {
                "message": {
                    "text": text_messages,
                },
            },
        }

        # lite model uses Bearer token auth; others use HMAC signature
        if self.model in self.SPARK_BEARER_MODELS:
            bearer_token = base64.b64encode(
                f"{self._spark_api_key}:{self._spark_api_secret}".encode("utf-8")
            ).decode("utf-8")
            headers = {"Authorization": f"Bearer {bearer_token}", "Content-Type": "application/json"}
            resp = requests.post(base_url, json=payload, headers=headers, timeout=120)
        else:
            auth_url = self._generate_spark_auth_url(base_url)
            resp = requests.post(auth_url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        header = data.get("header", {})
        code = header.get("code", 0)
        if code != 0:
            raise RuntimeError(f"Spark API error: code={code}, message={header.get('message', 'Unknown')}")

        choices = data.get("payload", {}).get("choices", {})
        text_array = choices.get("text", [])
        if text_array and isinstance(text_array, list):
            parts = [item.get("content", "") for item in text_array if isinstance(item, dict)]
            return "".join(parts).strip()

        content = choices.get("content", "")
        return content.strip() if content else ""


def create_client(backend: str = "ollama", **overrides) -> LLMClient:
    """Convenience factory function."""
    return LLMClient(backend=backend, **overrides)
