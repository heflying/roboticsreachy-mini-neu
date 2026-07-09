"""Tool: return the robot to center position."""

import logging
from typing import Any, Dict

from reachy_mini.utils import create_head_pose
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.dance_emotion_moves import GotoQueueMove


logger = logging.getLogger(__name__)

CENTER_DURATION = 1.5


class CenterPosition(Tool):
    """Return the robot to its center position."""

    name = "center_position"
    description = (
        "Return your body and head to the center (forward-facing) position. "
        "Use after turn_left or turn_right to face forward again."
    )
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Move back to center."""
        logger.info("Tool call: center_position")

        current_head_pose = deps.reachy_mini.get_current_head_pose()
        head_joints, antenna_joints = deps.reachy_mini.get_current_joint_positions()
        current_body_yaw = head_joints[0]
        antennas = (antenna_joints[0], antenna_joints[1])

        center_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=False)

        deps.movement_manager.queue_move(GotoQueueMove(
            target_head_pose=center_head_pose,
            start_head_pose=current_head_pose,
            target_antennas=antennas,
            start_antennas=antennas,
            target_body_yaw=0,
            start_body_yaw=current_body_yaw,
            duration=CENTER_DURATION,
        ))
        deps.movement_manager.set_moving_state(CENTER_DURATION)

        return {"status": "movement queued"}
