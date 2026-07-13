"""ASR 基类 — 流式推理统一接口。

设计决策：
- 收敛内化：process_chunk 只在文本稳定后才返回非空字符串
- chunk 缓冲内化：子类自行处理过小的音频帧
"""

import abc
from typing import Optional, Any

import numpy as np
import numpy.typing as npt


class ASRProvider(abc.ABC):
    """流式 ASR 抽象基类。

    每个 ASR 模型实现为一个类，提供统一的三个接口：
    - __init__: 加载模型
    - warmup: 预热，可选
    - process_chunk: 喂音频 chunk，is_final=True 时收敛并返回最终文本
    """

    def __init__(self) -> None:
        self._model_loaded = False

    @abc.abstractmethod
    def warmup(self) -> None:
        """预热模型。第一次推理前调用，避免首帧延迟失真。"""
        ...

    @abc.abstractmethod
    def process_chunk(
        self, audio: npt.NDArray[np.float32], is_final: bool = False
    ) -> str:
        """喂入一个音频 chunk。

        Args:
            audio: float32 数组，形状 (samples,) 或 (1, samples)，采样率 16000
            is_final: 是否为当前话语的最后一个 chunk。
                      设为 True 时强制收敛并返回最终识别文本。

        Returns:
            收敛后的文本（is_final=True 时），否则返回空字符串。
        """
        ...

    def start_utterance(self) -> None:
        """开始新的话语。子类可重写以创建新的流/重置状态。"""
        pass

    @property
    @abc.abstractmethod
    def model_info(self) -> dict[str, Any]:
        """返回模型元信息，用于结果标注。"""
        ...
