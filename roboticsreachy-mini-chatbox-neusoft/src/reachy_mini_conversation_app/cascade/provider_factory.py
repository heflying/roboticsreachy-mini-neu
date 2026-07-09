"""Factory functions for initializing cascade providers (ASR, LLM, TTS, transcript analysis)."""

from __future__ import annotations
import logging
import importlib
import os
from typing import Any, Dict

from reachy_mini_conversation_app.prompts import (
    CASCADE_EXTRA_INSTRUCTIONS,
    CASCADE_STREAMING_DIALOG_EXTRA_INSTRUCTIONS,
    get_session_instructions,
)
from reachy_mini_conversation_app.cascade.asr import ASRProvider
from reachy_mini_conversation_app.cascade.llm import LLMProvider
from reachy_mini_conversation_app.cascade.tts import TTSProvider
from reachy_mini_conversation_app.cascade.config import get_config
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.cascade.transcript_analysis import (
    NoOpTranscriptManager,
    TranscriptAnalysisManager,
)


logger = logging.getLogger(__name__)


def init_provider(provider_type: str, extra_kwargs: Dict[str, Any] | None = None, name: str | None = None) -> Any:
    """Initialize a provider (ASR/LLM/TTS) from cascade.yaml config.

    Args:
        provider_type: One of "asr", "llm", "tts"
        extra_kwargs: Additional kwargs to pass to provider constructor
        name: Provider name to use instead of the default from config

    Returns:
        Initialized provider instance

    """
    config = get_config()

    # All API keys that any provider might need
    api_key_map = {
        "OPENAI_API_KEY": config.OPENAI_API_KEY,
        "DEEPGRAM_API_KEY": config.DEEPGRAM_API_KEY,
        "GEMINI_API_KEY": config.GEMINI_API_KEY,
        "ELEVENLABS_API_KEY": config.ELEVENLABS_API_KEY,
        "GRADIUM_API_KEY": config.GRADIUM_API_KEY,
        "DASHSCOPE_API_KEY": config.DASHSCOPE_API_KEY,
        # Spark: combine key and secret into "key:secret" format if separate
        "SPARK_API_KEY": (
            f"{config.SPARK_API_KEY}:{config.SPARK_API_SECRET}"
            if config.SPARK_API_SECRET
            else config.SPARK_API_KEY or ""
        ),
    }

    # Get provider name, info, and settings using dynamic attribute access
    provider_name = name if name is not None else getattr(config, f"{provider_type}_provider")
    info = getattr(config, f"get_{provider_type}_provider_info")(provider_name)
    kwargs = getattr(config, f"get_{provider_type}_settings")(provider_name)

    # Add API key (validated at config load time)
    requires = info["requires"]
    if len(requires) == 1:
        kwargs["api_key"] = api_key_map[requires[0]]
    elif requires:
        raise ValueError(f"Multi-key providers not supported: {requires}")

    # Merge extra kwargs if provided
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    # Dynamic import and instantiate
    module = importlib.import_module(f"reachy_mini_conversation_app.cascade.{provider_type}.{info['module']}")
    ProviderClass = getattr(module, info["class"])

    # Log with provider-specific details
    extra_info = f", streaming={info['streaming']}" if "streaming" in info else ""
    logger.info(f"Initializing {provider_type.upper()}: {provider_name} (location={info['location']}{extra_info})")

    return ProviderClass(**kwargs)


def init_asr_provider() -> ASRProvider:
    """Initialize ASR provider from cascade.yaml config."""
    return init_provider("asr")  # type: ignore[no-any-return]


def init_llm_provider() -> LLMProvider:
    """Initialize LLM provider from cascade.yaml config."""
    streaming_dialog_requested = (
        os.getenv("CASCADE_STREAMING_DIALOG", "1").strip().lower() not in {"0", "false", "no", "off"}
        and os.getenv("CASCADE_DIALOG_ONLY_TOOLS", "1").strip().lower() not in {"0", "false", "no", "off"}
    )
    extra_instructions = (
        CASCADE_STREAMING_DIALOG_EXTRA_INSTRUCTIONS
        if streaming_dialog_requested
        else CASCADE_EXTRA_INSTRUCTIONS
    )
    cascade_instructions = get_session_instructions() + extra_instructions

    # Add app_id for Spark provider if configured
    extra_kwargs: Dict[str, Any] = {"system_instructions": cascade_instructions}
    config = get_config()
    if config.SPARK_APP_ID and "spark" in config.llm_provider.lower():
        extra_kwargs["app_id"] = config.SPARK_APP_ID

    return init_provider("llm", extra_kwargs)  # type: ignore[no-any-return]


def init_tts_provider() -> TTSProvider:
    """Initialize TTS provider from cascade.yaml config."""
    return init_provider("tts")  # type: ignore[no-any-return]


def init_transcript_analysis(deps: ToolDependencies) -> TranscriptAnalysisManager | NoOpTranscriptManager:
    """Initialize transcript analysis from profile reactions."""
    from reachy_mini_conversation_app.cascade.transcript_analysis import get_profile_reactions

    reactions = get_profile_reactions()
    if not reactions:
        logger.info("No profile reactions configured, transcript analysis disabled")
        return NoOpTranscriptManager()

    return TranscriptAnalysisManager(reactions=reactions, deps=deps)
