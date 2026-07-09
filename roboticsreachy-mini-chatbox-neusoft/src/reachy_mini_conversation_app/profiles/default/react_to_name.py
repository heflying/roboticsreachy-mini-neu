"""Reaction callback: play an emotion when someone says the robot's name."""

import logging

from reachy_mini.motion.recorded_move import RecordedMoves
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.dance_emotion_moves import EmotionQueueMove
from reachy_mini_conversation_app.cascade.transcript_analysis.base import TriggerMatch


logger = logging.getLogger(__name__)

RECORDED_MOVES = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")


async def react_to_name(deps: ToolDependencies, match: TriggerMatch, emotion: str = "enthusiastic1", **kwargs: object) -> None:
    """Play an emotion when someone mentions Reachy's name."""
    logger.info(f"Someone said my name! Matched: {match.words} — playing {emotion}")
    deps.movement_manager.queue_move(EmotionQueueMove(emotion, RECORDED_MOVES))
