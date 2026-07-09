import re
import sys
import logging
from pathlib import Path

from reachy_mini_conversation_app.config import DEFAULT_PROFILES_DIRECTORY, config, get_default_voice_for_backend


logger = logging.getLogger(__name__)


PROMPTS_LIBRARY_DIRECTORY = Path(__file__).parent / "prompts"
INSTRUCTIONS_FILENAME = "instructions.txt"
VOICE_FILENAME = "voice.txt"


def _expand_prompt_includes(content: str) -> str:
    """Expand [<name>] placeholders with content from prompts library files.

    Args:
        content: The template content with [<name>] placeholders

    Returns:
        Expanded content with placeholders replaced by file contents

    """
    # Pattern to match [<name>] where name is a valid file stem (alphanumeric, underscores, hyphens)
    # pattern = re.compile(r'^\[([a-zA-Z0-9_-]+)\]$')
    # Allow slashes for subdirectories
    pattern = re.compile(r'^\[([a-zA-Z0-9/_-]+)\]$')

    lines = content.split('\n')
    expanded_lines = []

    for line in lines:
        stripped = line.strip()
        match = pattern.match(stripped)

        if match:
            # Extract the name from [<name>]
            template_name = match.group(1)
            template_file = PROMPTS_LIBRARY_DIRECTORY / f"{template_name}.txt"

            try:
                if template_file.exists():
                    template_content = template_file.read_text(encoding="utf-8").rstrip()
                    expanded_lines.append(template_content)
                    logger.debug("Expanded template: [%s]", template_name)
                else:
                    logger.warning("Template file not found: %s, keeping placeholder", template_file)
                    expanded_lines.append(line)
            except Exception as e:
                logger.warning("Failed to read template '%s': %s, keeping placeholder", template_name, e)
                expanded_lines.append(line)
        else:
            expanded_lines.append(line)

    return '\n'.join(expanded_lines)


CASCADE_EXTRA_INSTRUCTIONS = """\n\n**IMPORTANT:**
## SPEAKING TO THE USER
- For normal dialog, answer directly with natural text. The runtime will send your text to speech automatically.
- Do not wrap speech in JSON or use a 'speak' tool — just reply with plain text.
- Keep ordinary replies to one short Chinese sentence, ideally 15-25 Chinese characters.
- For explanations, use at most two short Chinese sentences unless the user explicitly asks for detail.

## ISSUING SEVERAL TOOLS IN ONE RESPONSE
- You can always issue several tools in one response if needed.
- All the tool calls will be executed in sequence, so you can combine them to create complex behaviors.
- Do not hesitate to use multiple tools if the situation requires it, especially for complex tasks.
"""


CASCADE_STREAMING_DIALOG_EXTRA_INSTRUCTIONS = """\n\n**IMPORTANT:**
## SPEAKING TO THE USER
- For normal dialog, answer directly with natural text. The runtime will stream your text to speech.
- Do not wrap normal speech in JSON and do not mention internal tools.
- Keep ordinary replies to one short Chinese sentence, ideally 15-25 Chinese characters.
- For explanations, use at most two short Chinese sentences unless the user explicitly asks for detail.
- Do not add extra jokes, commentary, or follow-up questions unless the user asks.
- Prefer concise Mandarin phrasing suitable for spoken conversation.

## TOOLS
- If tools are available for non-speech actions, use them only when the user request requires them.
- Speech output itself is handled by the runtime pipeline in this mode.
"""


def get_session_instructions() -> str:
    """Get session instructions, loading from REACHY_MINI_CUSTOM_PROFILE if set."""
    profile = config.REACHY_MINI_CUSTOM_PROFILE
    if not profile:
        logger.info(f"Loading default prompt from {PROMPTS_LIBRARY_DIRECTORY / 'default_prompt.txt'}")
        instructions_file = PROMPTS_LIBRARY_DIRECTORY / "default_prompt.txt"
    else:
        if config.PROFILES_DIRECTORY != DEFAULT_PROFILES_DIRECTORY:
            logger.info(
                "Loading prompt from external profile '%s' (root=%s)",
                profile,
                config.PROFILES_DIRECTORY,
            )
        else:
            logger.info(f"Loading prompt from profile '{profile}'")
        instructions_file = config.PROFILES_DIRECTORY / profile / INSTRUCTIONS_FILENAME

    try:
        if instructions_file.exists():
            instructions = instructions_file.read_text(encoding="utf-8").strip()
            if instructions:
                # Expand [<name>] placeholders with content from prompts library
                expanded_instructions = _expand_prompt_includes(instructions)
                return expanded_instructions
            logger.error(f"Profile '{profile}' has empty {INSTRUCTIONS_FILENAME}")
            sys.exit(1)
        logger.error(f"Profile {profile} has no {INSTRUCTIONS_FILENAME}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load instructions from profile '{profile}': {e}")
        sys.exit(1)


def get_session_voice(default: str | None = None) -> str:
    """Resolve the voice to use for the session.

    If a custom profile is selected and contains a voice.txt, return its
    trimmed content; otherwise return the provided default or the active
    backend default voice.
    """
    fallback = get_default_voice_for_backend() if default is None else default
    profile = config.REACHY_MINI_CUSTOM_PROFILE
    if not profile:
        return fallback
    try:
        voice_file = config.PROFILES_DIRECTORY / profile / VOICE_FILENAME
        if voice_file.exists():
            voice = voice_file.read_text(encoding="utf-8").strip()
            return voice or fallback
    except Exception:
        pass
    return fallback
