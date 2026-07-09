"""Reaction callback: trigger the groovy_sway_and_roll dance."""

import logging

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.dance_emotion_moves import DanceQueueMove
from reachy_mini_conversation_app.cascade.transcript_analysis.base import TriggerMatch


logger = logging.getLogger(__name__)


async def do_groovy_dance(deps: ToolDependencies, match: TriggerMatch, **kwargs: object) -> None:
    """Queue the groovy_sway_and_roll dance when both dance + groove words are detected."""
    logger.info(f"Groovy dance triggered! Matched: {match.words}")
    deps.movement_manager.queue_move(DanceQueueMove("groovy_sway_and_roll"))
