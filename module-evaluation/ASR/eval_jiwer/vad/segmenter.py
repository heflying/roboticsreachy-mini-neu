"""VAD 音频切句器。

基于现有 Silero VAD ONNX 实现，将长音频切分为多个话语段。
复用了现有 codebase 的 vad_onnx.py 中的 VAD 核心逻辑。
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt
import soundfile as sf

SILERO_SAMPLE_RATE = 16000
VAD_CHUNK_SIZE = 512  # Silero VAD 固定帧长


@dataclass
class SpeechSegment:
    """一个 VAD 切出的话语段。"""

    audio: npt.NDArray[np.float32]  # 音频样本，float32，采样率 16000
    start_time: float  # 在原音频中的起始时间（秒）
    end_time: float  # 在原音频中的结束时间（秒）
    duration: float  # 段时长（秒）


class SileroSegmenter:
    """基于 Silero VAD 的话语切句器。

    参考现有 codebase cascade/vad.py 的 SileroVAD + VADStateMachine 逻辑，
    简化为离线批量切句场景。
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 500,
        padding_ms: int = 300,
    ) -> None:
        self.threshold = threshold
        self.min_speech_ms = min_speech_duration_ms
        self.min_silence_ms = min_silence_duration_ms
        self.padding_ms = padding_ms
        self._model: Optional[object] = None

    def _load_model(self) -> None:
        """延迟加载 Silero VAD ONNX 模型。"""
        if self._model is not None:
            return

        import onnxruntime

        model_path = self._get_model_path()
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = onnxruntime.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
        self._model = "loaded"

    def _get_model_path(self):
        """获取 VAD 模型路径，优先从现有 codebase 复制，否则下载。"""
        from pathlib import Path

        # 尝试从现有项目路径找
        existing_path = Path(
            "E:/programs/Robot/Reachy/"
            "roboticsreachy-mini-chatbox-neusoft/"
            "roboticsreachy-mini-chatbox-neusoft/"
            "models/VAD/silero/silero_vad.onnx"
        )
        if existing_path.exists():
            return existing_path

        # 本地缓存
        local = Path("models") / "VAD" / "silero" / "silero_vad.onnx"
        if local.exists():
            return local

        # 下载
        import urllib.request

        local.parent.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
        urllib.request.urlretrieve(url, str(local))
        return local

    def segment(self, audio_path: str) -> list[SpeechSegment]:
        """对音频文件进行 VAD 切句。

        Args:
            audio_path: 音频文件路径

        Returns:
            切分后的话语段列表，按时间顺序排列
        """
        self._load_model()

        # 加载音频，统一到 16000Hz float32
        audio, sr = sf.read(audio_path, dtype="float32")
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)  # 立体声 → 单声道

        if sr != SILERO_SAMPLE_RATE:
            import scipy.signal

            audio = scipy.signal.resample_poly(
                audio, up=SILERO_SAMPLE_RATE, down=sr
            ).astype(np.float32)

        if len(audio) == 0:
            return []

        # 滑动窗口 VAD
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros((1, 64), dtype=np.float32)

        speech_probs = []
        for i in range(0, len(audio) - VAD_CHUNK_SIZE + 1, VAD_CHUNK_SIZE):
            chunk = audio[i : i + VAD_CHUNK_SIZE]
            prob, state, context = self._run_vad(chunk, state, context)
            speech_probs.append(prob)

        # 基于阈值提取语音段
        segments = self._probs_to_segments(speech_probs, audio)
        return segments

    def _run_vad(
        self,
        audio: npt.NDArray[np.float32],
        state: npt.NDArray[np.float32],
        context: npt.NDArray[np.float32],
    ):
        audio_input = np.concatenate([context, audio.reshape(1, -1)], axis=1)
        ort_inputs = {
            "input": audio_input,
            "state": state,
            "sr": np.array(SILERO_SAMPLE_RATE, dtype=np.int64),
        }
        out, new_state = self._session.run(None, ort_inputs)
        new_context = audio_input[:, -64:]
        return out.item(), new_state, new_context

    def _probs_to_segments(
        self,
        probs: list[float],
        audio: npt.NDArray[np.float32],
    ) -> list[SpeechSegment]:
        """将逐帧语音概率转为话语段列表。"""
        chunk_duration_s = VAD_CHUNK_SIZE / SILERO_SAMPLE_RATE
        min_speech_chunks = int(self.min_speech_ms / 1000 / chunk_duration_s)
        min_silence_chunks = int(self.min_silence_ms / 1000 / chunk_duration_s)

        audio_dur_s = len(audio) / SILERO_SAMPLE_RATE
        padding_s = self.padding_ms / 1000.0

        segments = []
        in_speech = False
        speech_start = 0
        silence_count = 0

        for i, prob in enumerate(probs):
            is_speech = prob >= self.threshold

            if is_speech:
                if not in_speech:
                    # 检测到语音开始
                    speech_start = 0
                    in_speech = True
                    silence_count = 0
                speech_start += 1
            else:
                if in_speech:
                    silence_count += 1
                    # 静音超过阈值 → 切句
                    if (
                        silence_count >= min_silence_chunks
                        and speech_start >= min_speech_chunks
                    ):
                        end_chunk = i - silence_count + 1
                        start_time = max(
                            0.0,
                            (end_chunk - speech_start) * chunk_duration_s - padding_s,
                        )
                        end_time = min(
                            audio_dur_s,
                            end_chunk * chunk_duration_s + padding_s,
                        )
                        start_sample = int(start_time * SILERO_SAMPLE_RATE)
                        end_sample = int(end_time * SILERO_SAMPLE_RATE)

                        segments.append(
                            SpeechSegment(
                                audio=audio[start_sample:end_sample],
                                start_time=start_time,
                                end_time=end_time,
                                duration=end_time - start_time,
                            )
                        )
                        in_speech = False
                        speech_start = 0
                        silence_count = 0

        # 末尾未闭合的语音段
        if in_speech and speech_start >= min_speech_chunks:
            end_chunk = len(probs)
            start_time = max(
                0.0,
                (end_chunk - speech_start) * chunk_duration_s - padding_s,
            )
            end_time = audio_dur_s
            start_sample = int(start_time * SILERO_SAMPLE_RATE)
            end_sample = len(audio)

            segments.append(
                SpeechSegment(
                    audio=audio[start_sample:end_sample],
                    start_time=start_time,
                    end_time=end_time,
                    duration=end_time - start_time,
                )
            )

        # 过滤过短的噪音片段
        segments = [s for s in segments if s.duration >= self.min_speech_ms / 1000]
        return segments
