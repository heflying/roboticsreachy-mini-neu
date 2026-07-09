import os
import sys
import logging
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


# Locked profile: set to a profile name (e.g., "astronomer") to lock the app
# to that profile and disable all profile switching. Leave as None for normal behavior.
LOCKED_PROFILE: str | None = None
DEFAULT_PROFILES_DIRECTORY = Path(__file__).parent / "profiles"

logger = logging.getLogger(__name__)

AVAILABLE_VOICES: list[str] = [
    "alloy",
    "ash",
    "ballad",
    "cedar",
    "coral",
    "echo",
    "marin",
    "sage",
    "shimmer",
    "verse",
]
QWEN_AVAILABLE_VOICES: list[str] = [
    "Tina",
    "Cindy",
    "Liora",
    "Mira",
    "Sunnybobi",
    "Raymond",
    "Cherry",
    "Serena",
    "Ethan",
    "Chelsie",
]

OPENAI_BACKEND = "openai"
QWEN_OMNI_BACKEND = "qwen_omni"
DEFAULT_BACKEND_PROVIDER = OPENAI_BACKEND
DEFAULT_MODEL_NAME_BY_BACKEND = {
    OPENAI_BACKEND: "gpt-realtime",
    QWEN_OMNI_BACKEND: "qwen3.5-omni-flash-realtime",
}
DEFAULT_VOICE_BY_BACKEND = {
    OPENAI_BACKEND: "cedar",
    QWEN_OMNI_BACKEND: "Tina",
}


def _is_qwen_omni_model_name(model_name: str | None) -> bool:
    """Return True when the provided model name targets Qwen Omni Realtime."""
    candidate = (model_name or "").strip().lower()
    return candidate.startswith("qwen") and "realtime" in candidate


def _normalize_backend_provider(
    backend_provider: str | None = None,
    model_name: str | None = None,
) -> str:
    """Normalize backend selection, falling back to MODEL_NAME for compatibility."""
    candidate = (backend_provider or "").strip().lower()
    aliases = {
        "qwen": QWEN_OMNI_BACKEND,
        "qwen-omni": QWEN_OMNI_BACKEND,
        "qwen_omni_realtime": QWEN_OMNI_BACKEND,
        "dashscope": QWEN_OMNI_BACKEND,
    }
    candidate = aliases.get(candidate, candidate)
    if candidate in DEFAULT_MODEL_NAME_BY_BACKEND:
        return candidate
    if _is_qwen_omni_model_name(model_name):
        return QWEN_OMNI_BACKEND
    return DEFAULT_BACKEND_PROVIDER


def _resolve_model_name(
    backend_provider: str | None = None,
    model_name: str | None = None,
) -> str:
    """Return a model name that matches the selected backend provider."""
    normalized_backend = _normalize_backend_provider(backend_provider, model_name)
    candidate = (model_name or "").strip()
    if candidate:
        if normalized_backend == QWEN_OMNI_BACKEND and _is_qwen_omni_model_name(candidate):
            return candidate
        if normalized_backend == OPENAI_BACKEND and not _is_qwen_omni_model_name(candidate):
            return candidate
        logger.warning(
            "MODEL_NAME=%r does not match BACKEND_PROVIDER=%r, using default %r",
            candidate,
            normalized_backend,
            DEFAULT_MODEL_NAME_BY_BACKEND[normalized_backend],
        )
    return DEFAULT_MODEL_NAME_BY_BACKEND[normalized_backend]


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean environment flag.

    Accepted truthy values: 1, true, yes, on
    Accepted falsy values: 0, false, no, off
    """
    raw = os.getenv(name)
    if raw is None:
        return default

    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False

    logger.warning("Invalid boolean value for %s=%r, using default=%s", name, raw, default)
    return default


def _collect_profile_names(profiles_root: Path) -> set[str]:
    """Return profile folder names from a profiles root directory."""
    if not profiles_root.exists() or not profiles_root.is_dir():
        return set()
    return {p.name for p in profiles_root.iterdir() if p.is_dir()}


def _collect_tool_module_names(tools_root: Path) -> set[str]:
    """Return tool module names from a tools directory."""
    if not tools_root.exists() or not tools_root.is_dir():
        return set()
    ignored = {"__init__", "core_tools"}
    return {
        p.stem
        for p in tools_root.glob("*.py")
        if p.is_file() and p.stem not in ignored
    }


def _raise_on_name_collisions(
    *,
    label: str,
    external_root: Path,
    internal_root: Path,
    external_names: set[str],
    internal_names: set[str],
) -> None:
    """Raise with a clear message when external/internal names collide."""
    collisions = sorted(external_names & internal_names)
    if not collisions:
        return

    raise RuntimeError(
        f"Config.__init__(): Ambiguous {label} names found in both external and built-in libraries: {collisions}. "
        f"External {label} root: {external_root}. Built-in {label} root: {internal_root}. "
        f"Please rename the conflicting external {label}(s) to continue."
    )


# Validate LOCKED_PROFILE at startup
if LOCKED_PROFILE is not None:
    _profiles_dir = DEFAULT_PROFILES_DIRECTORY
    _profile_path = _profiles_dir / LOCKED_PROFILE
    _instructions_file = _profile_path / "instructions.txt"
    if not _profile_path.is_dir():
        print(f"Error: LOCKED_PROFILE '{LOCKED_PROFILE}' does not exist in {_profiles_dir}", file=sys.stderr)
        sys.exit(1)
    if not _instructions_file.is_file():
        print(f"Error: LOCKED_PROFILE '{LOCKED_PROFILE}' has no instructions.txt", file=sys.stderr)
        sys.exit(1)

_skip_dotenv = _env_flag("REACHY_MINI_SKIP_DOTENV", default=False)

if _skip_dotenv:
    logger.info("Skipping .env loading because REACHY_MINI_SKIP_DOTENV is set")
else:
    # Locate .env file (search upward from current working directory)
    dotenv_path = find_dotenv(usecwd=True)

    if dotenv_path:
        # Load .env and override environment variables
        load_dotenv(dotenv_path=dotenv_path, override=True)
        logger.info(f"Configuration loaded from {dotenv_path}")
    else:
        logger.warning("No .env file found, using environment variables")


class Config:
    """Configuration class for the conversation app."""

    # Required by selected realtime/cascade providers.
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # The key is downloaded in console.py if needed
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")

    # Optional
    BACKEND_PROVIDER = _normalize_backend_provider(os.getenv("BACKEND_PROVIDER"), os.getenv("MODEL_NAME"))
    MODEL_NAME = _resolve_model_name(BACKEND_PROVIDER, os.getenv("MODEL_NAME"))
    HF_HOME = os.getenv("HF_HOME", "./cache")
    LOCAL_VISION_MODEL = os.getenv("LOCAL_VISION_MODEL", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    HF_TOKEN = os.getenv("HF_TOKEN")  # Optional, falls back to hf auth login if not set
    QWEN_REALTIME_URL = os.getenv("QWEN_REALTIME_URL")
    QWEN_REALTIME_VOICE = os.getenv("QWEN_REALTIME_VOICE")
    QWEN_TOOL_MODE = os.getenv("QWEN_TOOL_MODE", "router").strip().lower()
    REACHY_MINI_MEMORY_WRITE_MODE = os.getenv("REACHY_MINI_MEMORY_WRITE_MODE", "extractor_only").strip().lower()

    logger.debug(
        "Backend provider: %s, Model: %s, HF_HOME: %s, Vision Model: %s",
        BACKEND_PROVIDER,
        MODEL_NAME,
        HF_HOME,
        LOCAL_VISION_MODEL,
    )

    _profiles_directory_env = os.getenv("REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY")
    PROFILES_DIRECTORY = (
        Path(_profiles_directory_env) if _profiles_directory_env else Path(__file__).parent / "profiles"
    )
    _tools_directory_env = os.getenv("REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY")
    TOOLS_DIRECTORY = Path(_tools_directory_env) if _tools_directory_env else None
    AUTOLOAD_EXTERNAL_TOOLS = _env_flag("AUTOLOAD_EXTERNAL_TOOLS", default=False)
    REACHY_MINI_CUSTOM_PROFILE = LOCKED_PROFILE or os.getenv("REACHY_MINI_CUSTOM_PROFILE")

    logger.debug(f"Custom Profile: {REACHY_MINI_CUSTOM_PROFILE}")

    def __init__(self) -> None:
        """Initialize the configuration."""
        if self.REACHY_MINI_CUSTOM_PROFILE and self.PROFILES_DIRECTORY != DEFAULT_PROFILES_DIRECTORY:
            selected_profile_path = self.PROFILES_DIRECTORY / self.REACHY_MINI_CUSTOM_PROFILE
            if not selected_profile_path.is_dir():
                available_profiles = sorted(_collect_profile_names(self.PROFILES_DIRECTORY))
                raise RuntimeError(
                    "Config.__init__(): Selected profile "
                    f"'{self.REACHY_MINI_CUSTOM_PROFILE}' was not found in external profiles root "
                    f"{self.PROFILES_DIRECTORY}. "
                    f"Available external profiles: {available_profiles}. "
                    "Either set 'REACHY_MINI_CUSTOM_PROFILE' to one of the available external profiles "
                    "or unset 'REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY' to use built-in profiles."
                )

        if self.PROFILES_DIRECTORY != DEFAULT_PROFILES_DIRECTORY:
            external_profiles = _collect_profile_names(self.PROFILES_DIRECTORY)
            internal_profiles = _collect_profile_names(DEFAULT_PROFILES_DIRECTORY)
            _raise_on_name_collisions(
                label="profile",
                external_root=self.PROFILES_DIRECTORY,
                internal_root=DEFAULT_PROFILES_DIRECTORY,
                external_names=external_profiles,
                internal_names=internal_profiles,
            )

        if self.TOOLS_DIRECTORY is not None:
            builtin_tools_root = Path(__file__).parent / "tools"
            external_tools = _collect_tool_module_names(self.TOOLS_DIRECTORY)
            internal_tools = _collect_tool_module_names(builtin_tools_root)
            _raise_on_name_collisions(
                label="tool",
                external_root=self.TOOLS_DIRECTORY,
                internal_root=builtin_tools_root,
                external_names=external_tools,
                internal_names=internal_tools,
            )

        if self.PROFILES_DIRECTORY != DEFAULT_PROFILES_DIRECTORY:
            logger.warning(
                "Environment variable 'REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY' is set. "
                "Profiles (instructions.txt, ...) will be loaded from %s.",
                self.PROFILES_DIRECTORY,
            )
        else:
            logger.info(
                "'REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY' is not set. "
                "Using built-in profiles from %s.",
                DEFAULT_PROFILES_DIRECTORY,
            )

        if self.TOOLS_DIRECTORY is not None:
            logger.warning(
                "Environment variable 'REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY' is set. "
                "External tools will be loaded from %s.",
                self.TOOLS_DIRECTORY,
            )
        else:
            logger.info(
                "'REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY' is not set. "
                "Using built-in shared tools only."
            )


config = Config()


def refresh_runtime_config_from_env() -> None:
    """Refresh mutable runtime config fields from the current environment."""
    config.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    config.DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
    config.BACKEND_PROVIDER = _normalize_backend_provider(os.getenv("BACKEND_PROVIDER"), os.getenv("MODEL_NAME"))
    config.MODEL_NAME = _resolve_model_name(config.BACKEND_PROVIDER, os.getenv("MODEL_NAME"))
    config.QWEN_REALTIME_URL = os.getenv("QWEN_REALTIME_URL")
    config.QWEN_REALTIME_VOICE = os.getenv("QWEN_REALTIME_VOICE")
    config.QWEN_TOOL_MODE = os.getenv("QWEN_TOOL_MODE", "router").strip().lower()
    config.REACHY_MINI_MEMORY_WRITE_MODE = os.getenv(
        "REACHY_MINI_MEMORY_WRITE_MODE",
        "extractor_only",
    ).strip().lower()
    config.REACHY_MINI_CUSTOM_PROFILE = LOCKED_PROFILE or os.getenv("REACHY_MINI_CUSTOM_PROFILE")


def get_backend_choice(model_name: str | None = None) -> str:
    """Return the configured backend family."""
    if model_name is not None:
        return _normalize_backend_provider(model_name=model_name)
    return _normalize_backend_provider(config.BACKEND_PROVIDER, config.MODEL_NAME)


def get_model_name_for_backend(backend: str) -> str:
    """Return the default model name for a backend selector value."""
    return DEFAULT_MODEL_NAME_BY_BACKEND[_normalize_backend_provider(backend)]


def get_available_voices_for_backend(backend: str | None = None) -> list[str]:
    """Return the curated voice list for a backend selector value."""
    normalized_backend = get_backend_choice() if backend is None else _normalize_backend_provider(backend)
    if normalized_backend == QWEN_OMNI_BACKEND:
        return list(QWEN_AVAILABLE_VOICES)
    return list(AVAILABLE_VOICES)


def get_default_voice_for_backend(backend: str | None = None) -> str:
    """Return the default voice for a backend selector value."""
    normalized_backend = get_backend_choice() if backend is None else _normalize_backend_provider(backend)
    return DEFAULT_VOICE_BY_BACKEND[normalized_backend]


def is_qwen_omni_model() -> bool:
    """Return True if the configured backend is Qwen Omni Realtime."""
    return get_backend_choice() == QWEN_OMNI_BACKEND


def set_custom_profile(profile: str | None) -> None:
    """Update the selected custom profile at runtime and expose it via env.

    This ensures modules that read `config` and code that inspects the
    environment see a consistent value.
    """
    if LOCKED_PROFILE is not None:
        return
    try:
        config.REACHY_MINI_CUSTOM_PROFILE = profile
    except Exception:
        pass
    try:
        import os as _os

        if profile:
            _os.environ["REACHY_MINI_CUSTOM_PROFILE"] = profile
        else:
            # Remove to reflect default
            _os.environ.pop("REACHY_MINI_CUSTOM_PROFILE", None)
    except Exception:
        pass
