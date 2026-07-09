"""实时性指标 — RTF / TTFC / 收敛延迟 / E2E Latency。

所有时间测量基于 time.perf_counter()。
"""

import time
from dataclasses import dataclass, field


@dataclass
class TimingMetrics:
    """单条话语的实时性指标。"""

    audio_duration_s: float = 0.0  # 音频时长（秒）
    processing_time_s: float = 0.0  # 推理总耗时（秒）
    rtf: float = 0.0  # 实时率 = processing / audio
    ttfc_s: float = 0.0  # 首次 chunk 发送 → 首个字符返回
    convergence_latency_s: float = 0.0  # 最后 chunk 发送 → 最终稳定文本
    e2e_latency_s: float = 0.0  # 首次 chunk 发送 → 最终文本

    num_chunks: int = 0  # 总 chunk 数
    first_chunk_time: float = 0.0  # 首个 chunk 发送时间戳
    last_chunk_time: float = 0.0  # 最后 chunk 发送时间戳
    first_text_time: float = 0.0  # 首个非空文本返回时间戳
    final_text_time: float = 0.0  # 最终文本返回时间戳


@dataclass
class AggregateTiming:
    """聚合后的实时性汇总。"""

    total_audio_duration_s: float = 0.0
    total_processing_time_s: float = 0.0
    avg_rtf: float = 0.0  # 总处理时间 / 总音频时长
    avg_ttfc_s: float = 0.0
    avg_convergence_latency_s: float = 0.0
    avg_e2e_latency_s: float = 0.0
    num_utterances: int = 0
    p50_ttfc_s: float = 0.0
    p90_ttfc_s: float = 0.0
    p95_ttfc_s: float = 0.0
    p99_ttfc_s: float = 0.0
    p50_e2e_s: float = 0.0
    p90_e2e_s: float = 0.0


class TimingTracker:
    """实时性打点器。

    在 ASR process_chunk 调用前后打点。
    """

    def __init__(self) -> None:
        self._first_chunk_time: float = 0.0
        self._last_chunk_time: float = 0.0
        self._first_text_time: float = 0.0
        self._final_text_time: float = 0.0
        self._num_chunks: int = 0
        self._processing_start: float = 0.0
        self._processing_end: float = 0.0

    def start_utterance(self) -> None:
        """开始新话语的打点。"""
        self._first_chunk_time = 0.0
        self._last_chunk_time = 0.0
        self._first_text_time = 0.0
        self._final_text_time = 0.0
        self._num_chunks = 0

    def on_chunk_sent(self, timestamp: float, is_first: bool) -> None:
        """记录 chunk 发送时间。"""
        if is_first:
            self._first_chunk_time = timestamp
        self._last_chunk_time = timestamp
        self._num_chunks += 1

    def on_text_received(self, timestamp: float, text: str, is_final: bool) -> None:
        """记录文本返回时间。"""
        if text and self._first_text_time == 0.0:
            self._first_text_time = timestamp
        if is_final:
            self._final_text_time = timestamp

    def set_processing_time(self, start: float, end: float) -> None:
        """设置推理耗时。"""
        self._processing_start = start
        self._processing_end = end

    def finalize(self, audio_duration_s: float) -> TimingMetrics:
        """生成 TimingMetrics。"""
        proc_time = max(0, self._processing_end - self._processing_start)
        rtf = proc_time / audio_duration_s if audio_duration_s > 0 else 0.0
        ttfc = (
            max(0, self._first_text_time - self._first_chunk_time)
            if self._first_chunk_time > 0 and self._first_text_time > 0
            else 0.0
        )
        convergence = (
            max(0, self._final_text_time - self._last_chunk_time)
            if self._last_chunk_time > 0 and self._final_text_time > 0
            else 0.0
        )
        e2e = (
            max(0, self._final_text_time - self._first_chunk_time)
            if self._first_chunk_time > 0 and self._final_text_time > 0
            else 0.0
        )

        return TimingMetrics(
            audio_duration_s=audio_duration_s,
            processing_time_s=proc_time,
            rtf=round(rtf, 6),
            ttfc_s=round(ttfc, 6),
            convergence_latency_s=round(convergence, 6),
            e2e_latency_s=round(e2e, 6),
            num_chunks=self._num_chunks,
            first_chunk_time=self._first_chunk_time,
            last_chunk_time=self._last_chunk_time,
            first_text_time=self._first_text_time,
            final_text_time=self._final_text_time,
        )


def aggregate_timing(timings: list[TimingMetrics]) -> AggregateTiming:
    """聚合实时性指标。"""
    if not timings:
        return AggregateTiming()

    total_audio = sum(t.audio_duration_s for t in timings)
    total_proc = sum(t.processing_time_s for t in timings)
    n = len(timings)

    ttfc_list = sorted([t.ttfc_s for t in timings if t.ttfc_s > 0])
    e2e_list = sorted([t.e2e_latency_s for t in timings if t.e2e_latency_s > 0])

    def percentile(data: list[float], p: float) -> float:
        if not data:
            return 0.0
        idx = int(len(data) * p / 100)
        return data[min(idx, len(data) - 1)]

    return AggregateTiming(
        total_audio_duration_s=round(total_audio, 3),
        total_processing_time_s=round(total_proc, 3),
        avg_rtf=round(total_proc / total_audio, 6) if total_audio > 0 else 0.0,
        avg_ttfc_s=round(sum(t.ttfc_s for t in timings) / n, 6) if n > 0 else 0.0,
        avg_convergence_latency_s=round(
            sum(t.convergence_latency_s for t in timings) / n, 6
        )
        if n > 0
        else 0.0,
        avg_e2e_latency_s=round(
            sum(t.e2e_latency_s for t in timings) / n, 6
        )
        if n > 0
        else 0.0,
        num_utterances=n,
        p50_ttfc_s=round(percentile(ttfc_list, 50), 6),
        p90_ttfc_s=round(percentile(ttfc_list, 90), 6),
        p95_ttfc_s=round(percentile(ttfc_list, 95), 6),
        p99_ttfc_s=round(percentile(ttfc_list, 99), 6),
        p50_e2e_s=round(percentile(e2e_list, 50), 6),
        p90_e2e_s=round(percentile(e2e_list, 90), 6),
    )
