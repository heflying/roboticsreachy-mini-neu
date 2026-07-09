"""Tool: describe_camera_image — capture a frame and describe it with a local vision model."""

import asyncio
import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class DescribeCameraImage(Tool):
    """Take a picture and answer a question about it using the local vision model."""

    name = "describe_camera_image"
    description = "Ask a question about what is visible to you through your eyes right now. A vision model will analyze your latest camera frame and answer your question about it."
    parameters_schema = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask about the picture",
            },
        },
        "required": ["question"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Capture a frame and describe it with the local vision model."""
        image_query = (kwargs.get("question") or "").strip()
        if not image_query:
            logger.warning("describe_camera_image: empty question")
            return {"error": "question must be a non-empty string"}

        logger.info("Tool call: describe_camera_image question=%s", image_query[:120])

        # Wait for any in-progress movement to complete before capturing
        if deps.movement_manager and deps.movement_manager.has_pending_moves():
            logger.info("Waiting for movement to complete before capturing...")
            for _ in range(30):  # up to 3s
                await asyncio.sleep(0.1)
                if not deps.movement_manager.has_pending_moves():
                    break

        if deps.camera_worker is None:
            logger.error("Camera worker not available")
            return {"error": "Camera worker not available"}

        if deps.vision_manager is None:
            logger.error("Vision manager not available (required for describe_camera_image)")
            return {"error": "Vision manager not available — use see_image_through_camera instead"}

        frame = deps.camera_worker.get_latest_frame()
        if frame is None:
            logger.error("No frame available from camera worker")
            return {"error": "No frame available"}

        vision_result = await asyncio.to_thread(
            deps.vision_manager.processor.process_image, frame, image_query,
        )
        if isinstance(vision_result, dict) and "error" in vision_result:
            return vision_result
        if isinstance(vision_result, str):
            logger.info("Vision model response: %s", vision_result[:500])
            return {"description": vision_result}
        return {"error": "vision returned non-string"}
