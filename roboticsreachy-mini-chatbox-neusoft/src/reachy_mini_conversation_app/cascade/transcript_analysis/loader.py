"""Load transcript reaction configurations from profile."""

from __future__ import annotations
import logging
import importlib
from typing import Any, Callable, Optional
from pathlib import Path

import yaml

from .base import TriggerConfig, ReactionConfig
from reachy_mini_conversation_app.config import config


logger = logging.getLogger(__name__)

PROFILES_DIRECTORY = Path(__file__).parent.parent.parent / "profiles"


def get_profile_reactions() -> list[ReactionConfig] | None:
    """Load reactions from current profile's reactions.yaml.

    Returns:
        List of ReactionConfig, or None if no profile or no reactions.yaml

    """
    profile = config.REACHY_MINI_CUSTOM_PROFILE
    if not profile:
        logger.debug("No profile set, transcript reactions disabled")
        return None

    reactions_file = PROFILES_DIRECTORY / profile / "reactions.yaml"
    if not reactions_file.exists():
        logger.debug(f"No reactions.yaml in profile '{profile}'")
        return None

    try:
        with open(reactions_file) as f:
            yaml_config = yaml.safe_load(f)

        if not yaml_config or not isinstance(yaml_config, list):
            return None

        reactions: list[ReactionConfig] = []

        for entry in yaml_config:
            name = entry.get("name")
            callback_name = entry.get("callback")
            if not name or not callback_name:
                logger.warning(f"Skipping reaction entry missing name or callback: {entry}")
                continue

            callback = _import_callback(profile, callback_name)
            if not callback:
                continue

            trigger_raw = entry.get("trigger", {})
            trigger = _parse_trigger(trigger_raw)

            params = entry.get("params", {})

            reactions.append(ReactionConfig(
                name=name,
                callback=callback,
                trigger=trigger,
                params=params,
                repeatable=entry.get("repeatable", False),
            ))

        logger.info(f"Loaded {len(reactions)} reactions from profile '{profile}'")
        return reactions if reactions else None

    except Exception as e:
        logger.warning(f"Failed to load reactions from profile '{profile}': {e}")
        return None


def _parse_trigger(raw: dict[str, Any]) -> TriggerConfig:
    """Parse a trigger dict from YAML into a TriggerConfig."""
    all_groups = [
        TriggerConfig(words=group.get("words", []), entities=group.get("entities", []))
        for group in raw.get("all", [])
    ]
    return TriggerConfig(
        words=raw.get("words", []),
        entities=raw.get("entities", []),
        all=all_groups,
    )


def _import_callback(profile: str, callback_name: str) -> Optional[Callable[..., Any]]:
    """Import a callback function from profile module."""
    try:
        module_path = f"reachy_mini_conversation_app.profiles.{profile}.{callback_name}"
        module = importlib.import_module(module_path)
        return getattr(module, callback_name, None)
    except (ImportError, AttributeError) as e:
        logger.warning(f"Failed to import callback '{callback_name}' from profile '{profile}': {e}")
        return None
