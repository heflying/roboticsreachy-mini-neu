"""Sherpa-ONNX Paraformer 流式 ASR 实现。

Paraformer 是达摩院提出的非自回归端到端语音识别模型（U2++ 架构）。
sherpa-onnx 通过 OnlineRecognizer.from_paraformer() 支持流式推理。

与 Zipformer (transducer) 的区别：
- 架构：CTC + Attention（非 Transducer）
- 模型文件：encoder.onnx + decoder.onnx（无 joiner）
- 无需 feature_dim 参数
"""

import logging
import os
import time

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from asr import ASRProvider

logger = logging.getLogger(__name__)


class SherpaOnnxParaformerASR(ASRProvider):
    """基于 sherpa-onnx OnlineRecognizer.from_paraformer 的流式 ASR。

    使用达摩院 Paraformer 模型，AISHELL-1 CER ~1.95%，中文识别精度当前最优。
    支持流式推理，兼容评测管线的 process_chunk 接口。
    """

    _MODEL_FILES = (
        "encoder.int8.onnx",
        "decoder.int8.onnx",
        "tokens.txt",
    )

    def __init__(
        self,
        # model_id: str = "csukuangfj/sherpa-onnx-streaming-paraformer-bilingual-zh-en",
        # model_dir: str = "models/ASR/paraformer-bilingual-zh-en",
        model_id: str = "csukuangfj/sherpa-onnx-streaming-paraformer-trilingual-zh-cantonese-en",
        model_dir: str = "models/ASR/paraformer-trilingual-zh-cantonese-en",
        num_threads: int = 1,
        sample_rate: int = 16000,
        decoding_method: str = "greedy_search",
        enable_endpoint: bool = False,  # 评测时关闭内置 endpoint，由 VAD 控制断句
        rule1_min_trailing_silence: float = 2.4,
        rule2_min_trailing_silence: float = 1.2,
        min_chunk_samples: int = 4800,  # 0.3s @ 16kHz
        use_hf_mirror: bool = True,
        debug_log: bool = False,  # 打印每个 final chunk 的详细解码信息
    ) -> None:
        super().__init__()
        self._debug_log = debug_log
        self._model_id = model_id
        self._model_dir = Path(model_dir)
        self._num_threads = num_threads
        self._sample_rate = sample_rate
        self._decoding_method = decoding_method
        self._enable_endpoint = enable_endpoint
        self._rule1_min_trailing_silence = rule1_min_trailing_silence
        self._rule2_min_trailing_silence = rule2_min_trailing_silence
        self._min_chunk_samples = min_chunk_samples
        self._use_hf_mirror = use_hf_mirror

        self._recognizer: Any = None
        self._stream: Any = None
        self._audio_buffer: list[float] = []

        # Eager load
        self._ensure_model()
        logger.info("SherpaOnnxParaformerASR initialized")

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        if self._model_loaded:
            return
        self._download_if_missing()
        self._create_recognizer()
        self._model_loaded = True

    def _download_if_missing(self) -> None:
        if self._all_files_present():
            logger.info(f"Model files found in {self._model_dir}")
            return
        logger.info(f"Downloading model from {self._model_id}...")

        old_endpoint = os.environ.get("HF_ENDPOINT")
        try:
            if self._use_hf_mirror:
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            from huggingface_hub import snapshot_download

            t0 = time.perf_counter()
            self._model_dir.mkdir(parents=True, exist_ok=True)
            snapshot_download(repo_id=self._model_id, local_dir=str(self._model_dir))
            logger.info(f"Model downloaded in {time.perf_counter() - t0:.1f}s")
        finally:
            if old_endpoint is not None:
                os.environ["HF_ENDPOINT"] = old_endpoint
            elif self._use_hf_mirror:
                del os.environ["HF_ENDPOINT"]

    def _all_files_present(self) -> bool:
        if not self._model_dir.exists():
            return False
        for fname in self._MODEL_FILES:
            fpath = self._model_dir / fname
            if not fpath.exists() or fpath.stat().st_size == 0:
                return False
        return True

    def _create_recognizer(self) -> None:
        import sherpa_onnx

        encoder = str(self._model_dir / "encoder.int8.onnx")
        decoder = str(self._model_dir / "decoder.int8.onnx")
        tokens = str(self._model_dir / "tokens.txt")

        t0 = time.perf_counter()
        self._recognizer = sherpa_onnx.OnlineRecognizer.from_paraformer(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            num_threads=self._num_threads,
            sample_rate=self._sample_rate,
            enable_endpoint_detection=self._enable_endpoint,
            rule1_min_trailing_silence=self._rule1_min_trailing_silence,
            rule2_min_trailing_silence=self._rule2_min_trailing_silence,
            decoding_method=self._decoding_method,
            provider="cpu",
        )
        logger.info(
            f"Recognizer created in {time.perf_counter() - t0:.2f}s "
            f"(threads={self._num_threads})"
        )

    # ------------------------------------------------------------------
    # ASRProvider 接口
    # ------------------------------------------------------------------

    def warmup(self) -> None:
        """发送一小段静音以预热模型。"""
        warmup_audio = np.zeros(self._sample_rate, dtype=np.float32)  # 1 second
        self.start_utterance()
        self.process_chunk(warmup_audio, is_final=True)
        logger.info("Warmup complete")

    def start_utterance(self) -> None:
        """开始新话语，创建在线流。"""
        self._stream = self._recognizer.create_stream()
        self._audio_buffer = []

    def process_chunk(
        self, audio: npt.NDArray[np.float32], is_final: bool = False
    ) -> str:
        """喂入音频 chunk。

        收敛内化：只有 is_final=True 时才会完成解码并返回最终文本。
        中间过程的 partial result 不返回（按设计决策 #2）。

        Paraformer 与 Zipformer 使用相同的 OnlineRecognizer 流式接口，
        因此 process_chunk 逻辑完全一致。
        """
        if self._stream is None:
            self.start_utterance()

        # 确保 float32 一维
        audio = np.asarray(audio, dtype=np.float32).ravel()

        if len(audio) > 0:
            self._audio_buffer.extend(audio.tolist())

        if not is_final:
            # 积累够最小帧才喂给 recognizer
            if len(self._audio_buffer) < self._min_chunk_samples:
                return ""
            buf = np.array(self._audio_buffer, dtype=np.float32)
            self._stream.accept_waveform(self._sample_rate, buf)
            self._audio_buffer = []
            # 持续解码当前所有就绪帧，避免积压到 final chunk
            while self._recognizer.is_ready(self._stream):
                self._recognizer.decode_stream(self._stream)
            return ""

        # is_final: 喂入剩余音频，调用 input_finished() 完成解码
        # 1) 喂入缓冲中剩余音频
        if self._audio_buffer:
            buf = np.array(self._audio_buffer, dtype=np.float32)
            self._stream.accept_waveform(self._sample_rate, buf)
            self._audio_buffer = []

        # 2) 喂入一段静音，推动 CTC 提交尾部 pending 的最后一个字
        silence = np.zeros(int(self._sample_rate * 1.), dtype=np.float32)
        self._stream.accept_waveform(self._sample_rate, silence)
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)

        # 3) 通知 recognizer 输入结束，完成解码
        self._stream.input_finished()
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)

        # 4) 获取最终结果，去掉汉字旁边的空格（保留英文单词间的空格）
        import re
        result = self._recognizer.get_result(self._stream).strip()
        # 去掉汉字前后的空格：\s 匹配空格/制表符等，[\u4e00-\u9fff] 匹配汉字
        result = re.sub(r"(?<=[\u4e00-\u9fff])\s+|\s+(?=[\u4e00-\u9fff])", "", result)

        if self._debug_log:
            logger.info(f"[Paraformer final] result='{result}'")

        # 5) 清理流
        self._stream = None

        return result

    @property
    def model_info(self) -> dict[str, Any]:
        return {
            "model_name": self._model_id.rsplit("/", 1)[-1],
            "model_id": self._model_id,
            "model_type": "sherpa_onnx_paraformer",
            "num_threads": self._num_threads,
            "sample_rate": self._sample_rate,
            "decoding_method": self._decoding_method,
            "enable_endpoint": self._enable_endpoint,
            "chunk_buffer_samples": self._min_chunk_samples,
        }
