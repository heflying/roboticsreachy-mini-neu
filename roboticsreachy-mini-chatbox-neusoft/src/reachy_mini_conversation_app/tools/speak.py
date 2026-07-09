"""Speak tool - allows the LLM to speak text through TTS."""

from typing import Any, Dict

from .core_tools import Tool, ToolDependencies


class SpeakTool(Tool):
    """Tool for speaking text to the user via TTS."""

    name = "speak"
    description = "Speak the given message to the user. Use this tool for ALL verbal responses."
    parameters_schema = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The text to speak to the user",
            }
        },
        "required": ["message"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Return the message for TTS processing by the handler."""
        message = kwargs.get("message", "")
        return {"message": message}
