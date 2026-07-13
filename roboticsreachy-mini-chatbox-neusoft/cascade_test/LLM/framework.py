"""LLM evaluation test framework for Reachy Mini cascade pipeline.

Evaluates LLM providers across 8 categories per the evaluation criteria:
  01-事实问答, 02-任务执行, 03-工具调用, 04-多轮记忆,
  05-记忆总结, 06-ASR错字输入, 07-陪伴安抚, 08-安全风险

Usage:
    cd project_root
    python -m cascade_test.LLM.runner --provider qwen-flash
    python -m cascade_test.LLM.runner --provider ollama-qwen2.5-0.5b --category 01-事实问答
"""

from __future__ import annotations

import csv
import json
import time
import asyncio
import logging
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict

from reachy_mini_conversation_app.cascade.llm.base import LLMChunk, LLMProvider

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent

CATEGORIES = [
    "01-事实问答",
    "02-任务执行",
    "03-工具调用",
    "04-多轮记忆",
    "05-记忆总结",
    "06-ASR错字输入",
    "07-陪伴安抚",
    "08-安全风险",
]

# Speed thresholds per evaluation doc
TTFT_THRESHOLDS = {"excellent": 300, "good": 800, "acceptable": 1500}
TOKENS_PER_SEC_THRESHOLDS = {"excellent": 20, "good": 10, "acceptable": 5}
HALLUCINATION_THRESHOLDS = {"excellent": 3, "good": 8, "acceptable": 15}

# Weights for final score per evaluation doc section 5.3
SCORE_WEIGHTS = {
    "safety": 0.25,
    "multi_turn_recall": 0.15,
    "memory_summary": 0.10,
    "asr_robustness": 0.10,
    "tool_call_accuracy": 0.10,
    "oral_style": 0.10,
    "hallucination": 0.10,
    "ttft": 0.05,
    "tokens_per_sec": 0.05,
}

# Tool definitions for tool-call tests
TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前时间",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                    "date": {"type": "string", "description": "日期，如'今天'、'明天'"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": "设置提醒事项",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "提醒内容"},
                    "time": {"type": "string", "description": "提醒时间"},
                },
                "required": ["content", "time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "control_device",
            "description": "控制智能家居设备",
            "parameters": {
                "type": "object",
                "properties": {
                    "device": {"type": "string", "description": "设备名称，如'客厅灯'"},
                    "action": {"type": "string", "description": "操作，如'打开'、'关闭'", "enum": ["打开", "关闭"]},
                },
                "required": ["device", "action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember_user_fact",
            "description": "记住用户的关键个人信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "信息类别",
                        "enum": ["name", "preference", "health", "family", "schedule", "other"],
                    },
                    "fact": {"type": "string", "description": "要记住的事实内容"},
                },
                "required": ["category", "fact"],
            },
        },
    },
]


@dataclass
class TestCase:
    """A single test case loaded from JSON."""

    case_id: str
    category: str
    input: str
    expected_points: List[str]
    risk_level: str = "low"
    notes: str = ""
    # For multi-turn cases: list of (role, content) pairs before the final input
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    # For ASR noise cases: the original correct text
    original_text: str = ""
    # For tool-call cases: expected tool name
    expected_tool: str = ""
    # For tool-call cases: expected parameter keys
    expected_params: Dict[str, str] = field(default_factory=dict)


@dataclass
class SpeedMetrics:
    """Speed metrics for a single response."""

    ttft_ms: Optional[float] = None
    total_ms: Optional[float] = None
    tokens_per_sec: Optional[float] = None
    total_tokens: int = 0
    text_length: int = 0


@dataclass
class CaseResult:
    """Result for a single test case."""

    case_id: str
    category: str
    full_text: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    speed: SpeedMetrics = field(default_factory=SpeedMetrics)
    # Scoring results
    expected_points_hit: List[str] = field(default_factory=list)
    expected_points_miss: List[str] = field(default_factory=list)
    hallucination: bool = False
    oral_style_score: float = 0.0  # 0-1
    safety_score: float = 0.0  # 0-1
    asr_intent_correct: Optional[bool] = None
    tool_call_correct: Optional[bool] = None
    memory_recall_correct: Optional[bool] = None
    memory_summary_complete: Optional[bool] = None
    passed: bool = False
    error: str = ""


@dataclass
class CategoryReport:
    """Aggregated report for one category."""

    category: str
    total_cases: int = 0
    passed_cases: int = 0
    avg_ttft_ms: Optional[float] = None
    avg_tokens_per_sec: Optional[float] = None
    hallucination_rate: Optional[float] = None
    tool_call_accuracy: Optional[float] = None
    multi_turn_recall_rate: Optional[float] = None
    memory_summary_completeness: Optional[float] = None
    asr_robustness_rate: Optional[float] = None
    avg_oral_style_score: Optional[float] = None
    avg_safety_score: Optional[float] = None
    case_results: List[CaseResult] = field(default_factory=list)


def load_test_cases(category: Optional[str] = None) -> List[TestCase]:
    """Load test cases from JSON files in category folders.

    Args:
        category: If provided, only load cases from this category folder.

    Returns:
        List of TestCase objects.
    """
    cases: List[TestCase] = []
    categories = [category] if category else CATEGORIES

    for cat in categories:
        cat_dir = BASE_DIR / cat
        if not cat_dir.is_dir():
            logger.warning("Category directory not found: %s", cat_dir)
            continue

        for json_file in sorted(cat_dir.glob("*.json")):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                tc = TestCase(
                    case_id=data.get("case_id", json_file.stem),
                    category=cat,
                    input=data["input"],
                    expected_points=data.get("expected_points", []),
                    risk_level=data.get("risk_level", "low"),
                    notes=data.get("notes", ""),
                    conversation_history=data.get("conversation_history", []),
                    original_text=data.get("original_text", ""),
                    expected_tool=data.get("expected_tool", ""),
                    expected_params=data.get("expected_params", {}),
                )
                cases.append(tc)
            except (json.JSONDecodeError, KeyError) as e:
                logger.error("Failed to load test case %s: %s", json_file, e)

    return cases


def build_messages(tc: TestCase) -> List[Dict[str, Any]]:
    """Build OpenAI-format message list from a TestCase.

    Does NOT prepend system instructions — those are handled by the LLM provider
    itself (via init_llm_provider / cascade.yaml configuration).
    """
    messages: List[Dict[str, Any]] = []

    # Add conversation history
    for turn in tc.conversation_history:
        messages.append({"role": turn.get("role", "user"), "content": turn.get("content", "")})

    # Add the test input
    messages.append({"role": "user", "content": tc.input})

    return messages


async def run_single_case(
    llm: LLMProvider,
    tc: TestCase,
    provide_tools: bool = False,
) -> CaseResult:
    """Run a single test case against an LLM provider.

    Args:
        llm: The LLM provider instance.
        tc: The test case.
        provide_tools: Whether to pass tool definitions.

    Returns:
        CaseResult with response text, speed metrics, and tool calls.
    """
    result = CaseResult(case_id=tc.case_id, category=tc.category)
    messages = build_messages(tc)
    tools = TOOL_DEFINITIONS if provide_tools else None

    accumulated_text = ""
    tool_calls: List[Dict[str, Any]] = []
    first_token_time: Optional[float] = None
    token_count = 0
    request_start = time.perf_counter()

    try:
        async for chunk in llm.generate(
            messages=messages,
            tools=tools,
            temperature=0.7,
            token=None,
        ):
            if chunk.type == "text_delta" and chunk.content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                accumulated_text += chunk.content
                token_count += 1
            elif chunk.type == "tool_call" and chunk.tool_call:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                tool_calls.append(chunk.tool_call)
            elif chunk.type == "done":
                pass

        total_time = time.perf_counter() - request_start

        result.full_text = accumulated_text
        result.tool_calls = tool_calls
        result.speed = SpeedMetrics(
            ttft_ms=round((first_token_time - request_start) * 1000, 1) if first_token_time else None,
            total_ms=round(total_time * 1000, 1),
            tokens_per_sec=round(token_count / total_time, 1) if total_time > 0 else None,
            total_tokens=token_count,
            text_length=len(accumulated_text),
        )

    except Exception as e:
        result.error = str(e)
        logger.error("Case %s failed: %s", tc.case_id, e)

    return result


def rate_ttft(ttft_ms: Optional[float]) -> str:
    """Rate TTFT against evaluation thresholds."""
    if ttft_ms is None:
        return "N/A"
    if ttft_ms <= TTFT_THRESHOLDS["excellent"]:
        return "优秀"
    if ttft_ms <= TTFT_THRESHOLDS["good"]:
        return "良好"
    if ttft_ms <= TTFT_THRESHOLDS["acceptable"]:
        return "可接受"
    return "偏差"


def rate_tokens_per_sec(tokens_per_sec: Optional[float]) -> str:
    """Rate generation speed against evaluation thresholds."""
    if tokens_per_sec is None:
        return "N/A"
    if tokens_per_sec > TOKENS_PER_SEC_THRESHOLDS["excellent"]:
        return "优秀"
    if tokens_per_sec > TOKENS_PER_SEC_THRESHOLDS["good"]:
        return "良好"
    if tokens_per_sec > TOKENS_PER_SEC_THRESHOLDS["acceptable"]:
        return "可接受"
    return "偏差"


def create_llm_provider(provider_name: str) -> LLMProvider:
    """Create an LLM provider instance using the cascade module's factory.

    Uses cascade.yaml configuration and init_llm_provider() to create the
    provider, ensuring the test environment matches the real runtime setup
    (same system instructions, API keys, model params, etc.).

    The provider_name must match a key under llm.providers in cascade.yaml.
    It overrides the default provider via the CASCADE_LLM_PROVIDER env var.

    Args:
        provider_name: Provider name from cascade.yaml (e.g. "qwen-flash",
            "ollama-qwen2.5-0.5b", "spark-ultra").

    Returns:
        Initialized LLMProvider instance.

    Raises:
        RuntimeError: If the provider cannot be initialized (missing API key,
            unknown provider name, hardware mismatch, etc.).
    """
    import os
    from reachy_mini_conversation_app.cascade.config import get_config, set_config
    from reachy_mini_conversation_app.cascade.provider_factory import init_llm_provider

    # Override LLM provider via env var so CascadeConfig picks it up
    os.environ["CASCADE_LLM_PROVIDER"] = provider_name

    # Also set safe ASR/TTS providers to avoid validation failures from
    # hardware-incompatible defaults (e.g. parakeet_mlx on non-Apple Silicon).
    # We only need the LLM; ASR/TTS won't actually be used during testing.
    if "CASCADE_ASR_PROVIDER" not in os.environ:
        os.environ["CASCADE_ASR_PROVIDER"] = "zipformer_sherpa"
    if "CASCADE_TTS_PROVIDER" not in os.environ:
        os.environ["CASCADE_TTS_PROVIDER"] = "piper_zh"

    # Reset the config singleton so it re-reads the env vars
    set_config(None)

    try:
        config = get_config()
        llm = init_llm_provider()
        logger.info(
            "Created LLM provider '%s' (module=%s, model=%s)",
            provider_name,
            config.get_llm_provider_info(provider_name).get("module", "?"),
            getattr(llm, "model", "?"),
        )
        return llm
    except Exception as e:
        # Provide a clear error message about what's needed
        # Fall back to reading available providers from YAML directly
        # in case config validation failed before populating llm_providers
        available = get_available_llm_providers()
        raise RuntimeError(
            f"Failed to initialize LLM provider '{provider_name}'.\n"
            f"Error: {e}\n"
            f"Available providers in cascade.yaml: {', '.join(available)}\n"
            f"Make sure required API keys are set in .env or environment variables."
        ) from e


def get_available_llm_providers() -> List[str]:
    """Get list of available LLM provider names from cascade.yaml."""
    import yaml

    config_file = Path("cascade.yaml")
    if not config_file.exists():
        return []

    with open(config_file, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    llm_section = data.get("llm", {})
    return list(llm_section.get("providers", {}).keys())
