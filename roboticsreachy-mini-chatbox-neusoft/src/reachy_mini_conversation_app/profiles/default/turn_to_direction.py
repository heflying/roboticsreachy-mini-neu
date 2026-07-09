"""Reaction callback: turn the robot 90 degrees left or right, then return to center."""

import logging

import numpy as np

from reachy_mini.utils import create_head_pose
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove
from reachy_mini_conversation_app.cascade.transcript_analysis.base import TriggerMatch


logger = logging.getLogger(__name__)

TURN_ANGLE = np.pi / 2  # 90 degrees
TURN_DURATION = 1.5  # seconds to reach target
HOLD_DURATION = 1.0  # seconds to hold at target


async def turn_to_direction(
    deps: ToolDependencies, match: TriggerMatch, direction: str = "left", **kwargs: object
) -> None:
    """Turn robot 90 degrees in the given direction, hold, then return to center."""
    sign = 1.0 if direction == "left" else -1.0
    angle = sign * TURN_ANGLE
    logger.info(f"Turning {direction}! Matched: {match.words}")

    current_head_pose = deps.reachy_mini.get_current_head_pose()
    head_joints, antenna_joints = deps.reachy_mini.get_current_joint_positions()
    current_body_yaw = head_joints[0]
    antennas = (antenna_joints[0], antenna_joints[1])

    target_head_pose = create_head_pose(0, 0, 0, 0, 0, angle, degrees=False)
    center_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=False)

    # Turn to target
    deps.movement_manager.queue_move(GotoQueueMove(
        target_head_pose=target_head_pose,
        start_head_pose=current_head_pose,
        target_antennas=antennas,
        start_antennas=antennas,
        target_body_yaw=current_body_yaw + angle,
        start_body_yaw=current_body_yaw,
        duration=TURN_DURATION,
    ))

    # Hold
    deps.movement_manager.queue_move(GotoQueueMove(
        target_head_pose=target_head_pose,
        start_head_pose=target_head_pose,
        target_antennas=antennas,
        start_antennas=antennas,
        target_body_yaw=current_body_yaw + angle,
        start_body_yaw=current_body_yaw + angle,
        duration=HOLD_DURATION,
    ))

    # Return to center
    deps.movement_manager.queue_move(GotoQueueMove(
        target_head_pose=center_head_pose,
        start_head_pose=target_head_pose,
        target_antennas=antennas,
        start_antennas=antennas,
        target_body_yaw=current_body_yaw,
        start_body_yaw=current_body_yaw + angle,
        duration=TURN_DURATION,
    ))

    total_duration = TURN_DURATION + HOLD_DURATION + TURN_DURATION
    deps.movement_manager.set_moving_state(total_duration)
