"""Cascade-specific configuration, loaded only when cascade mode is active."""

import os
import logging
import platform
import importlib
from typing import Any, Dict
from pathlib import Path

import yaml


logger = logging.getLogger(__name__)

# Suppress verbose debug logging from numba (used by librosa for audio resampling)
logging.getLogger("numba").setLevel(logging.WARNING)

# Metadata keys (not passed to provider constructor)
ASR_METADATA_KEYS = {"module", "class", "streaming", "location", "requires", "hardware", "description", "status", "import_check", "install_extra"}
LLM_METADATA_KEYS = {"module", "class", "location", "requires", "description", "import_check", "install_extra"}
TTS_METADATA_KEYS = {"module", "class", "location", "requires", "hardware", "description", "import_check", "install_extra"}


def _load_cascade_config() -> Dict[str, Any]:
    """Load cascade configuration from YAML file."""
    config_file = Path("cascade.yaml")

    if not config_file.exists():
        abs_path = config_file.resolve()
        raise RuntimeError(
            f"cascade.yaml not found at {abs_path}. "
            "Run the app from the project root, or copy cascade.yaml to your working directory."
        )

    try:
        with open(config_file, encoding="utf-8") as f:
            config: Dict[str, Any] = yaml.safe_load(f)

        # Validate required top-level keys
        required_keys = ["asr", "llm", "tts"]
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise RuntimeError(f"cascade.yaml is missing required keys: {', '.join(missing_keys)}")

        logger.info("Cascade configuration loaded from cascade.yaml")
        return config

    except yaml.YAMLError as e:
        raise RuntimeError(f"Invalid YAML syntax in cascade.yaml:\n{e}\nPlease check the file for syntax errors.")


class CascadeConfig:
    """Configuration class for cascade pipeline mode."""

    def __init__(self) -> None:
        """Initialize cascade configuration by loading cascade.yaml."""
        # API keys from environment
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        self.ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
        self.GRADIUM_API_KEY = os.getenv("GRADIUM_API_KEY")
        self.DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
        self.SPARK_API_KEY = os.getenv("SPARK_API_KEY")  # 科大讯飞星火大模型 (格式: "key:secret" 或单独的 key)
        self.SPARK_API_SECRET = os.getenv("SPARK_API_SECRET")  # 科大讯飞 APISecret (可选，与 SPARK_API_KEY 分开设置)
        self.SPARK_APP_ID = os.getenv("SPARK_APP_ID")    # 科大讯飞应用ID (可选)

        self._cascade = _load_cascade_config()

        # VAD config
        vad_cfg = self._cascade.get("vad", {})
        self.vad_backend: str = os.getenv("CASCADE_VAD_BACKEND") or vad_cfg.get("backend", "onnx")
        self.vad_threshold: float = float(vad_cfg.get("threshold", 0.5))
        self.vad_min_speech_duration_ms: int = int(vad_cfg.get("min_speech_duration_ms", 250))
        self.vad_min_silence_duration_ms: int = int(vad_cfg.get("min_silence_duration_ms", 500))
        self.sentence_pause_threshold_ms: int = int(vad_cfg.get("sentence_pause_threshold_ms", 200))
        self.enable_sentence_warmup: bool = bool(vad_cfg.get("enable_sentence_warmup", True))

        # ASR config
        self.asr_provider = os.getenv("CASCADE_ASR_PROVIDER") or self._cascade["asr"]["provider"]
        self.asr_providers = self._cascade["asr"]["providers"]

        # LLM config
        self.llm_provider = os.getenv("CASCADE_LLM_PROVIDER") or self._cascade["llm"]["provider"]
        self.llm_providers = self._cascade["llm"]["providers"]
        self.llm_temperature: float = self._cascade["llm"].get("temperature", 1.0)

        # TTS config
        self.tts_provider = os.getenv("CASCADE_TTS_PROVIDER") or self._cascade["tts"]["provider"]
        self.tts_providers = self._cascade["tts"]["providers"]
        self.tts_trim_silence = self._cascade["tts"].get("trim_silence", True)

        # Transcript analysis config (optional section)
        ta = self._cascade.get("transcript_analysis", {})
        self.gliner_model: str = ta.get("gliner_model", "urchade/gliner_small-v2.1")

        self._validate()
        self._log_config()

    def get_asr_provider_info(self, name: str | None = None) -> Dict[str, Any]:
        """Get full provider info from cascade.yaml."""
        provider_name = name or self.asr_provider
        if provider_name not in self.asr_providers:
            available = ", ".join(self.asr_providers.keys())
            raise ValueError(f"Unknown ASR provider: {provider_name}. Available: {available}")
        return self.asr_providers[provider_name]  # type: ignore[no-any-return]

    def get_asr_settings(self, name: str | None = None) -> Dict[str, Any]:
        """Get provider settings (excludes metadata like module, class, streaming, etc.)."""
        info = self.get_asr_provider_info(name)
        return {k: v for k, v in info.items() if k not in ASR_METADATA_KEYS}

    def is_asr_streaming(self, name: str | None = None) -> bool:
        """Check if provider supports streaming."""
        return self.get_asr_provider_info(name)["streaming"]  # type: ignore[no-any-return]

    def get_llm_provider_info(self, name: str | None = None) -> Dict[str, Any]:
        """Get full LLM provider info from cascade.yaml."""
        provider_name = name or self.llm_provider
        if provider_name not in self.llm_providers:
            available = ", ".join(self.llm_providers.keys())
            raise ValueError(f"Unknown LLM provider: {provider_name}. Available: {available}")
        return self.llm_providers[provider_name]  # type: ignore[no-any-return]

    def get_llm_settings(self, name: str | None = None) -> Dict[str, Any]:
        """Get LLM provider settings (excludes metadata)."""
        info = self.get_llm_provider_info(name)
        return {k: v for k, v in info.items() if k not in LLM_METADATA_KEYS}

    def get_tts_provider_info(self, name: str | None = None) -> Dict[str, Any]:
        """Get full TTS provider info from cascade.yaml."""
        provider_name = name or self.tts_provider
        if provider_name not in self.tts_providers:
            available = ", ".join(self.tts_providers.keys())
            raise ValueError(f"Unknown TTS provider: {provider_name}. Available: {available}")
        return self.tts_providers[provider_name]  # type: ignore[no-any-return]

    def get_tts_settings(self, name: str | None = None) -> Dict[str, Any]:
        """Get TTS provider settings (excludes metadata)."""
        info = self.get_tts_provider_info(name)
        return {k: v for k, v in info.items() if k not in TTS_METADATA_KEYS}

    def _validate(self) -> None:
        """Validate selected providers at config load time."""
        api_key_env = {
            "OPENAI_API_KEY": self.OPENAI_API_KEY,
            "DEEPGRAM_API_KEY": self.DEEPGRAM_API_KEY,
            "GEMINI_API_KEY": self.GEMINI_API_KEY,
            "ELEVENLABS_API_KEY": self.ELEVENLABS_API_KEY,
            "GRADIUM_API_KEY": self.GRADIUM_API_KEY,
            "DASHSCOPE_API_KEY": self.DASHSCOPE_API_KEY,
            "SPARK_API_KEY": self.SPARK_API_KEY,
        }

        # Validate VAD backend
        if self.vad_backend == "onnx":
            try:
                importlib.import_module("onnxruntime")
            except ImportError:
                raise RuntimeError(
                    "VAD backend 'onnx' requires 'onnxruntime'. "
                    "Install with: uv sync --extra cascade_silero_vad_onnx"
                )
        elif self.vad_backend == "pytorch":
            try:
                importlib.import_module("torch")
            except ImportError:
                raise RuntimeError(
                    "VAD backend 'pytorch' requires 'torch'. "
                    "Install with: uv sync --extra cascade_silero_vad"
                )
        else:
            raise RuntimeError(
                f"Unknown VAD backend '{self.vad_backend}'. Use 'onnx' or 'pytorch'."
            )

        for section, provider_name, providers_dict in [
            ("asr", self.asr_provider, self.asr_providers),
            ("llm", self.llm_provider, self.llm_providers),
            ("tts", self.tts_provider, self.tts_providers),
        ]:
            # Provider exists?
            if provider_name not in providers_dict:
                available = ", ".join(providers_dict.keys())
                raise RuntimeError(
                    f"Unknown {section.upper()} provider '{provider_name}'. Available: {available}"
                )

            info = providers_dict[provider_name]

            # Required API keys set?
            for required in info.get("requires", []):
                if not api_key_env.get(required):
                    raise RuntimeError(
                        f"{required} environment variable is not set. "
                        f"Required by {section.upper()} provider '{provider_name}'. "
                        f"Set it in your .env file or environment."
                    )

            # Required dependency installed?
            import_check = info.get("import_check")
            if import_check:
                try:
                    importlib.import_module(import_check)
                except ImportError:
                    extra = info.get("install_extra", "???")
                    raise RuntimeError(
                        f"{section.upper()} provider '{provider_name}' requires '{import_check}'. "
                        f"Install with: uv sync --extra {extra}"
                    )

            # Hardware compatible?
            hw = info.get("hardware")
            if hw == "apple_silicon":
                if platform.machine() != "arm64" or platform.system() != "Darwin":
                    raise RuntimeError(
                        f"{section.upper()} provider '{provider_name}' requires Apple Silicon (macOS arm64). "
                        f"Detected: {platform.system()} {platform.machine()}. "
                        f"Choose a cloud provider or one compatible with your hardware."
                    )
            elif hw == "cuda":
                try:
                    import torch
                    if not torch.cuda.is_available():
                        raise RuntimeError(
                            f"{section.upper()} provider '{provider_name}' requires an NVIDIA CUDA GPU. "
                            f"No CUDA device detected. Choose a cloud provider instead."
                        )
                    logger.info(f"{section.upper()} provider '{provider_name}': CUDA GPU detected.")
                except ImportError:
                    raise RuntimeError(
                        f"{section.upper()} provider '{provider_name}' requires PyTorch with CUDA support. "
                        f"PyTorch is not installed."
                    )

    def _log_config(self) -> None:
        """Log the loaded configuration."""
        logger.info(f"Cascade: VAD={self.vad_backend}, ASR={self.asr_provider}, LLM={self.llm_provider}, TTS={self.tts_provider}")
        logger.debug(f"ASR provider info: {self.get_asr_provider_info()}")
        logger.debug(f"LLM provider info: {self.get_llm_provider_info()}")
        logger.debug(f"TTS provider info: {self.get_tts_provider_info()}")


_config: CascadeConfig | None = None


def get_config() -> CascadeConfig:
    """Return the cascade config singleton (created on first call)."""
    global _config
    if _config is None:
        _config = CascadeConfig()
    return _config


def set_config(cfg: CascadeConfig | None) -> None:
    """Override the cascade config singleton (for testing)."""
    global _config
    _config = cfg
