"""Silero VAD ONNX Runtime backend."""

import logging
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


logger = logging.getLogger(__name__)

_MODEL_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
_MODEL_DIR = Path("models") / "VAD" / "silero"
SILERO_SAMPLE_RATE = 16000


def _ensure_model() -> Path:
    """Download silero_vad.onnx if not cached locally."""
    model_path = _MODEL_DIR / "silero_vad.onnx"
    if model_path.exists():
        return model_path
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading Silero VAD ONNX model to {model_path}...")
    urllib.request.urlretrieve(_MODEL_URL, str(model_path))
    logger.info("Silero VAD ONNX model downloaded")
    return model_path


class SileroVADOnnx:
    """Silero VAD inference using ONNX Runtime."""

    def __init__(self) -> None:
        try:
            import onnxruntime
        except ImportError as exc:
            raise RuntimeError(
                "Silero VAD ONNX backend requires onnxruntime. Install with: "
                "uv sync --extra cascade_silero_vad_onnx"
            ) from exc

        model_path = _ensure_model()
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = onnxruntime.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)
        logger.info("Silero VAD ONNX backend initialized")

    def get_speech_prob(self, audio: npt.NDArray[Any], sample_rate: int = SILERO_SAMPLE_RATE) -> float:
        if sample_rate != SILERO_SAMPLE_RATE:
            raise ValueError(f"Silero VAD requires {SILERO_SAMPLE_RATE}Hz audio, got {sample_rate}Hz")

        if audio.dtype == np.int16:
            audio_float = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.float32:
            audio_float = audio
        else:
            audio_float = audio.astype(np.float32)

        audio_input = np.concatenate([self._context, audio_float.reshape(1, -1)], axis=1)
        ort_inputs = {
            "input": audio_input,
            "state": self._state,
            "sr": np.array(sample_rate, dtype=np.int64),
        }
        out, state = self._session.run(None, ort_inputs)
        self._state = state
        self._context = audio_input[:, -64:]
        return out.item()  # type: ignore[no-any-return]

    def reset_states(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)
