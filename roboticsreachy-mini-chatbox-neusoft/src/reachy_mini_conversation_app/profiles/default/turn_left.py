"""Tool: turn the robot 90 degrees to the left and stay turned."""

import logging
from typing import Any, Dict

import numpy as np

from reachy_mini.utils import create_head_pose
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove


logger = logging.getLogger(__name__)

TURN_ANGLE = np.pi / 2  # 90 degrees
TURN_DURATION = 1.5


class TurnLeft(Tool):
    """Turn the robot 90 degrees to the left."""

    name = "turn_left"
    description = (
        "Turn your whole body 90 degrees to the left and stay facing that direction. "
        "Use center_position to come back to the original orientation."
    )
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Turn left and stay."""
        logger.info("Tool call: turn_left")

        current_head_pose = deps.reachy_mini.get_current_head_pose()
        head_joints, antenna_joints = deps.reachy_mini.get_current_joint_positions()
        current_body_yaw = head_joints[0]
        antennas = (antenna_joints[0], antenna_joints[1])

        target_head_pose = create_head_pose(0, 0, 0, 0, 0, TURN_ANGLE, degrees=False)

        deps.movement_manager.queue_move(GotoQueueMove(
            target_head_pose=target_head_pose,
            start_head_pose=current_head_pose,
            target_antennas=antennas,
            start_antennas=antennas,
            target_body_yaw=current_body_yaw + TURN_ANGLE,
            start_body_yaw=current_body_yaw,
            duration=TURN_DURATION,
        ))
        deps.movement_manager.set_moving_state(TURN_DURATION)

        return {"status": "movement queued"}
