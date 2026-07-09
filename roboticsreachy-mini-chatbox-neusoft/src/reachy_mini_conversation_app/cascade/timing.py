"""Cascade Architecture Performance Metrics System v1.0.

Based on: docs/级联架构性能指标设计.md

Two-Level Metrics:
- L1 (User-Perceived): response_start, TTFB, transcript_show
- L2 (Internal Breakdown): ASR(B1-B5), Gap(G1/G2), LLM(C1-C3), TTS(D1-D5), Playback(E1/E2)
"""

from __future__ import annotations
import time
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


logger = logging.getLogger(__name__)


# ============================================================================
# 阈值定义 (基于设计文档 Section 五)
# ============================================================================

class RatingLevel(Enum):
    """评级等级"""
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    ACCEPTABLE = "ACCEPTABLE"
    NEEDS_IMPROVEMENT = "需改进"
    MONITOR_ONLY = "仅监控"


@dataclass
class ThresholdConfig:
    """阈值配置"""
    excellent: float
    good: float
    acceptable: float
    hotspot_mark: float
    core_hotspot: bool = False


# L1 阈值 (用户感知)
L1_THRESHOLDS = {
    "L1_response_start": ThresholdConfig(150, 300, 500, 500, core_hotspot=False),
    "L2_ttfb": ThresholdConfig(500, 800, 1200, 1200, core_hotspot=True),
    "L3_transcript_show": ThresholdConfig(200, 400, 600, 600, core_hotspot=False),
}

# L2 ASR 阈值
# B1/B2/B3 在用户说话期间并行完成，不计入 TTFB 关键路径
# B5 已删除：B4 已覆盖到 asr_result_delivered，B5 为 B4 内部子阶段
ASR_THRESHOLDS = {
    "B1_asr_connect": ThresholdConfig(200, 500, 1000, 500, core_hotspot=False),
    "B2_asr_init": ThresholdConfig(100, 200, 300, 200, core_hotspot=False),
    "B3_asr_audio_send": ThresholdConfig(100, 200, 300, 200, core_hotspot=False),
    "B4_asr_cloud_process": ThresholdConfig(200, 300, 500, 300, core_hotspot=True),
}

# L2 ASR 阈值 (本地推理模式, 如 Zipformer sherpa-onnx)
ASR_LOCAL_THRESHOLDS = {
    "B4_asr_local_process": ThresholdConfig(30, 50, 100, 100, core_hotspot=True),
}

# L2 Gap 阈值 (级联特有隐藏瓶颈)
GAP_THRESHOLDS = {
    "G1_asr_llm_gap": ThresholdConfig(20, 50, 100, 50, core_hotspot=False),
    "G2_llm_tts_gap": ThresholdConfig(20, 50, 100, 50, core_hotspot=False),
}

# L2 LLM 阈值
LLM_THRESHOLDS = {
    "C1_llm_request_send": ThresholdConfig(50, 100, 200, 100, core_hotspot=False),
    "C2_llm_first_token": ThresholdConfig(300, 500, 800, 500, core_hotspot=True),
    "C3_llm_stream_output": None,
}

# L2 TTS 阈值 (云端 WebSocket 模式, 如 Qwen Realtime)
TTS_THRESHOLDS = {
    "D1_tts_connect": ThresholdConfig(200, 500, 1000, 500, core_hotspot=False),
    "D2_tts_init": ThresholdConfig(100, 200, 300, 200, core_hotspot=False),
    "D3_tts_first_audio": ThresholdConfig(200, 300, 500, 300, core_hotspot=True),
    "D4_tts_stream_output": None,
}

# L2 TTS 阈值 (本地推理模式, 如 Piper)
TTS_LOCAL_THRESHOLDS = {
    "D1_tts_connect": None,  # Always 0ms (local, no connection)
    "D2_tts_thread_start": ThresholdConfig(50, 100, 200, 100, core_hotspot=False),
    "D3_tts_model_inference": ThresholdConfig(300, 500, 800, 500, core_hotspot=True),
    "D4_tts_generation": None,  # Monitor only
}

# L2 Playback 阈值
PLAYBACK_THRESHOLDS = {
    "E1_audio_queue": ThresholdConfig(30, 50, 100, 50, core_hotspot=False),
    "E2_playback_start": ThresholdConfig(50, 100, 200, 100, core_hotspot=False),
}


# ============================================================================
# 指标结果数据类
# ============================================================================

@dataclass
class MetricResult:
    """单个指标结果"""
    name: str
    code: str
    value_ms: float
    rating: RatingLevel
    is_hotspot: bool
    is_core_hotspot: bool
    is_reuse: bool = False
    is_monitor_only: bool = False
    is_parallel: bool = False


# ============================================================================
# 指标计算器
# ============================================================================

class MetricsCalculator:
    """指标计算与评级引擎"""

    @staticmethod
    def rate(value_ms: float, threshold: Optional[ThresholdConfig]) -> Tuple[RatingLevel, bool, bool]:
        """计算评级

        Returns:
            (rating_level, is_hotspot, is_core_hotspot)
        """
        if threshold is None:
            return RatingLevel.MONITOR_ONLY, False, False

        is_hotspot = value_ms > threshold.hotspot_mark
        is_core = threshold.core_hotspot

        if value_ms <= threshold.excellent:
            return RatingLevel.EXCELLENT, is_hotspot, is_core
        elif value_ms <= threshold.good:
            return RatingLevel.GOOD, is_hotspot, is_core
        elif value_ms <= threshold.acceptable:
            return RatingLevel.ACCEPTABLE, is_hotspot, is_core
        else:
            return RatingLevel.NEEDS_IMPROVEMENT, is_hotspot, is_core

    @staticmethod
    def format_rating_icon(rating: RatingLevel) -> str:
        """返回评级图标"""
        icons = {
            RatingLevel.EXCELLENT: "✅",
            RatingLevel.GOOD: "👍",
            RatingLevel.ACCEPTABLE: "⚠️",
            RatingLevel.NEEDS_IMPROVEMENT: "❌",
            RatingLevel.MONITOR_ONLY: "",
        }
        return icons.get(rating, "")


# ============================================================================
# LatencyTracker 重构
# ============================================================================

class LatencyTracker:
    """Cascade Architecture Latency Tracker

    支持:
    - 事件记录 (mark)
    - L1/L2 分层指标计算
    - 阈值评级
    - 热点判定
    - 指标追溯公式验证
    - 标准报告输出
    """

    # 事件名称标准化映射
    EVENT_ALIASES = {
        # VAD / Speech endpoints
        "vad_speech_end": "speech_end",
        "user_stop_click": "speech_end",
        # ASR events (B1-B5)
        "asr_ws_connect_start": "asr_b1_start",
        "asr_ws_connected": "asr_b1_end",
        "asr_ws_reused": "asr_reuse",
        "asr_session_update_sent": "asr_b2_end",
        "asr_audio_send_start": "asr_b3_start",
        "asr_audio_send_complete": "asr_b3_end",
        "asr_commit_sent": "asr_b4_start",
        "asr_final_received": "asr_b4_end",
        "asr_result_delivered": "asr_b5_end",
        # ASR events - 本地 ASR (sherpa-onnx Zipformer)
        "asr_local_ready": "asr_reuse",
        "asr_local_stream_start": "asr_b2_end",
        "asr_local_chunk_decode": "asr_b3_end",
        "asr_local_final_decode": "asr_b4_start",
        # LLM events (C1-C3, G1)
        "llm_start": "llm_g1_end",
        "llm_request_sending": "llm_c1_start",
        "llm_stream_opened": "llm_c1_end",
        "llm_first_token": "llm_c2_end",
        "llm_first_speech_chunk": "response_start",
        "llm_complete": "llm_g2_start",
        # TTS events (D1-D5, G2)
        "tts_start": "tts_g2_end",
        "tts_ws_connect_start": "tts_d1_start",
        "tts_ws_connected": "tts_d1_end",
        "tts_ws_reused": "tts_reuse",
        "tts_ws_preconnect_start": "tts_preconnect_start",
        "tts_ws_preconnected": "tts_preconnect_done",
        "tts_ws_preconnect_reused": "tts_preconnect_reuse",
        "tts_ws_prepared_stale": "tts_preconnect_stale",
        "tts_session_update_sent": "tts_d2_end",
        "tts_commit_sent": "tts_d4_start",
        "tts_first_chunk_ready": "tts_d4_end",
        "tts_finish_event_received": "tts_d5_end",
        # Piper 本地 TTS 事件
        "tts_model_generation_start": "tts_local_d2_end",
        "tts_model_first_chunk": "tts_local_model_first",
        "tts_model_generation_complete": "tts_local_d5_end",
        # TTS preconnect wait events (新增 - WebSocket预连接优化)
        "tts_wait_preconnect_start": "tts_wait_start",
        "tts_wait_preconnect_success": "tts_wait_success",
        "tts_wait_preconnect_timeout": "tts_wait_timeout",
        "tts_wait_preconnect_failed": "tts_wait_failed",
        # Playback events (E1-E2)
        "audio_playback_started": "first_audio",
        "tts_audio_queued": "playback_e1_end",
        "playback_complete": "playback_e2_end",
        # UI events
        "transcript_show": "transcript_show",
    }

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self.start_time: Optional[float] = None
        self.reference_name: str = "pipeline_start"
        self.turn_number: int = 1
        self._preserved_events: List[Dict[str, Any]] = []  # events to carry across reset
        self._cancelled: bool = False

    def reset(self, reference_name: str = "pipeline_start") -> None:
        """Reset tracker for new conversation turn.

        Any events previously stored in _preserved_events are re-injected
        with their original perf_counter timestamps so that durations like
        TTFB remain accurate across the reset boundary.

        Note: _cancelled is NOT cleared here — it is cleared in next_turn(),
        which runs after print_summary(). This ensures the cancelled tag
        survives even if reset() is called before the cancelled turn's report.
        """
        self.events = list(self._preserved_events)
        self._preserved_events = []
        self.start_time = time.perf_counter()
        self.reference_name = reference_name
        logger.info(f"⏱️  LATENCY TRACKING STARTED: {reference_name}")

    def next_turn(self) -> None:
        """Advance to next turn for report numbering and clear per-turn state."""
        self.turn_number += 1
        self._cancelled = False

    def mark_cancelled(self) -> None:
        """Mark the current turn as cancelled (e.g. by barge-in).

        Cancelled turns are reported with diagnostic data but excluded
        from SLO scoring. The flag persists until next_turn() is called
        (after print_summary()), so it survives reset() boundaries.
        """
        self._cancelled = True
        logger.info("⏱️  Turn marked as CANCELLED (barge-in)")

    @property
    def is_cancelled(self) -> bool:
        """Whether the current turn was cancelled."""
        return self._cancelled

    def preserve_event(self, event_name: str) -> None:
        """Preserve an event so it survives the next reset() call.

        Finds the most recent event matching event_name and saves it.
        When reset() is called, preserved events are re-injected into
        the new session with their original perf_counter timestamps.
        """
        canonical = self.EVENT_ALIASES.get(event_name, event_name)
        for event in reversed(self.events):
            if event["canonical"] == canonical or event["name"] == event_name:
                self._preserved_events.append(event)
                logger.debug(f"Preserved event '{event_name}' for next reset (ts={event['timestamp']:.3f})")
                return

    def mark(self, event_name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Mark a timing event."""
        if self.start_time is None:
            self.reset()

        timestamp = time.perf_counter()
        self._add_event(event_name, timestamp, metadata)

    def inject_event(self, event_name: str, timestamp: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Inject an event with an external timestamp (e.g. from before tracker.reset).

        Used when an event like vad_speech_end occurs before the tracker is reset
        for a new turn. The caller saves the perf_counter timestamp before reset,
        then injects it afterwards so it appears in the new session's timeline.
        """
        if self.start_time is None:
            self.reset()
        self._add_event(event_name, timestamp, metadata)

    def _add_event(self, event_name: str, timestamp: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a timing event to the list."""
        start = self.start_time or timestamp
        elapsed_ms = (timestamp - start) * 1000

        canonical_name = self.EVENT_ALIASES.get(event_name, event_name)

        event = {
            "name": event_name,
            "canonical": canonical_name,
            "timestamp": timestamp,
            "elapsed_ms": elapsed_ms,
            "metadata": metadata or {},
        }
        self.events.append(event)

        metadata_str = ""
        if metadata:
            parts = [f"{k}={v}" for k, v in metadata.items()]
            metadata_str = f" ({', '.join(parts)})"
        logger.info(f"⏱️  [{elapsed_ms:7.1f}ms] {event_name}{metadata_str}")

    def get_duration(self, start_event: str, end_event: str, use_first: bool = False) -> Optional[float]:
        """Get duration between two events in milliseconds.

        Args:
            use_first: If True, use first matching event; if False (default),
                use last matching event. Default is False because preserved
                events from previous turns can cause first-match to pick stale
                timestamps. Using last-match naturally selects current-turn events.
        """
        start_canonical = self.EVENT_ALIASES.get(start_event, start_event)
        end_canonical = self.EVENT_ALIASES.get(end_event, end_event)

        start_ts: float | None = None
        end_ts: float | None = None

        for event in self.events:
            canonical = event.get("canonical", event["name"])

            if canonical == start_canonical or event["name"] == start_event:
                if use_first:
                    if start_ts is None:
                        start_ts = event["timestamp"]
                else:
                    start_ts = event["timestamp"]

            if canonical == end_canonical or event["name"] == end_event:
                if use_first:
                    if end_ts is None:
                        end_ts = event["timestamp"]
                else:
                    end_ts = event["timestamp"]

        if start_ts is not None and end_ts is not None:
            return float((end_ts - start_ts) * 1000)
        return None

    def has_event(self, event_name: str) -> bool:
        """Check if an event exists."""
        canonical = self.EVENT_ALIASES.get(event_name, event_name)
        return any(
            e.get("canonical", e["name"]) == canonical or e["name"] == event_name
            for e in self.events
        )

    # ========================================================================
    # L1 指标计算
    # ========================================================================

    def calculate_l1_metrics(self) -> List[MetricResult]:
        """计算 L1 用户感知指标"""
        results = []

        speech_end = self._get_speech_end_event()
        response_start = self._get_response_start_event()

        # L1 响应启动延迟
        if speech_end and response_start:
            duration = self.get_duration(speech_end, response_start, use_first=True)
            if duration is not None:
                rating, is_hotspot, is_core = MetricsCalculator.rate(
                    duration, L1_THRESHOLDS["L1_response_start"]
                )
                results.append(MetricResult(
                    name="响应启动延迟(vad结束说话 到 llm输出首字)",
                    code="L1",
                    value_ms=duration,
                    rating=rating,
                    is_hotspot=is_hotspot,
                    is_core_hotspot=is_core,
                ))

        # L2 首音延迟 (TTFB) - 核心
        first_audio = "audio_playback_started"
        if speech_end and self.has_event(first_audio):
            duration = self.get_duration(speech_end, first_audio, use_first=True)
            if duration is not None:
                rating, is_hotspot, is_core = MetricsCalculator.rate(
                    duration, L1_THRESHOLDS["L2_ttfb"]
                )
                results.append(MetricResult(
                    name="首音延迟 (TTFB vad结束说话 到 开始输出音频，=L1+llm首句耗时+D3)",
                    code="L2",
                    value_ms=duration,
                    rating=rating,
                    is_hotspot=is_hotspot,
                    is_core_hotspot=is_core,
                ))

        # L3 转录延迟
        transcript_show = "transcript_show"
        if speech_end and self.has_event(transcript_show):
            duration = self.get_duration(speech_end, transcript_show, use_first=True)
            if duration is not None:
                rating, is_hotspot, is_core = MetricsCalculator.rate(
                    duration, L1_THRESHOLDS["L3_transcript_show"]
                )
                results.append(MetricResult(
                    name="转录延迟(vad结束说话 到 显示转录字符)",
                    code="L3",
                    value_ms=duration,
                    rating=rating,
                    is_hotspot=is_hotspot,
                    is_core_hotspot=is_core,
                ))

        return results

    # ========================================================================
    # L2 指标计算
    # ========================================================================

    def calculate_l2_asr_metrics(self) -> List[MetricResult]:
        """计算 ASR 阶段 L2 指标"""
        results = []

        # Local ASR: B1/B2/B3 run in parallel with VAD, not on critical path.
        # Show B1/B2 as reuse markers, compute only B4 (final decode → result delivered).
        if self._is_local_asr():
            # B1 本地连接: 0ms reuse
            results.append(MetricResult(
                name="ASR连接建立(本地)",
                code="B1",
                value_ms=0.0,
                rating=RatingLevel.EXCELLENT,
                is_hotspot=False,
                is_core_hotspot=False,
                is_reuse=True,
            ))
            # B2 本地初始化: 0ms reuse
            results.append(MetricResult(
                name="ASR初始化(本地)",
                code="B2",
                value_ms=0.0,
                rating=RatingLevel.EXCELLENT,
                is_hotspot=False,
                is_core_hotspot=False,
                is_reuse=True,
            ))
            # B4 本地处理 (核心)
            if self.has_event("asr_local_final_decode") and self.has_event("asr_result_delivered"):
                duration = self.get_duration("asr_local_final_decode", "asr_result_delivered", use_first=True)
                if duration is not None:
                    rating, is_hotspot, is_core = MetricsCalculator.rate(
                        duration, ASR_LOCAL_THRESHOLDS["B4_asr_local_process"]
                    )
                    results.append(MetricResult(
                        name="ASR本地处理",
                        code="B4",
                        value_ms=duration,
                        rating=rating,
                        is_hotspot=is_hotspot,
                        is_core_hotspot=is_core,
                    ))
            return results

        # Cloud ASR: full B1-B4 calculation
        # B1 ASR 连接
        if self.has_event("asr_ws_reused"):
            results.append(MetricResult(
                name="ASR连接建立",
                code="B1",
                value_ms=0.0,
                rating=RatingLevel.EXCELLENT,
                is_hotspot=False,
                is_core_hotspot=False,
                is_reuse=True,
            ))
        elif self.has_event("asr_local_ready"):
            # 本地 ASR: 无需连接，0ms reuse
            results.append(MetricResult(
                name="ASR连接建立(本地)",
                code="B1",
                value_ms=0.0,
                rating=RatingLevel.EXCELLENT,
                is_hotspot=False,
                is_core_hotspot=False,
                is_reuse=True,
            ))
        elif self.has_event("asr_ws_connect_start") and self.has_event("asr_ws_connected"):
            duration = self.get_duration("asr_ws_connect_start", "asr_ws_connected")
            if duration is not None:
                rating, is_hotspot, is_core = MetricsCalculator.rate(
                    duration, ASR_THRESHOLDS["B1_asr_connect"]
                )
                results.append(MetricResult(
                    name="ASR连接建立",
                    code="B1",
                    value_ms=duration,
                    rating=rating,
                    is_hotspot=is_hotspot,
                    is_core_hotspot=is_core,
                ))

        # B2 ASR 初始化
        if self.has_event("asr_ws_reused"):
            # reuse 时 B2 也显示 0ms
            results.append(MetricResult(
                name="ASR初始化",
                code="B2",
                value_ms=0.0,
                rating=RatingLevel.EXCELLENT,
                is_hotspot=False,
                is_core_hotspot=False,
                is_reuse=True,
            ))
        elif self.has_event("asr_local_stream_start"):
            # 本地 ASR: 流已就绪，0ms reuse
            results.append(MetricResult(
                name="ASR初始化(本地)",
                code="B2",
                value_ms=0.0,
                rating=RatingLevel.EXCELLENT,
                is_hotspot=False,
                is_core_hotspot=False,
                is_reuse=True,
            ))
        elif self.has_event("asr_ws_connected") and self.has_event("asr_session_update_sent"):
            duration = self.get_duration("asr_ws_connected", "asr_session_update_sent")
            if duration is not None:
                rating, is_hotspot, is_core = MetricsCalculator.rate(
                    duration, ASR_THRESHOLDS["B2_asr_init"]
                )
                results.append(MetricResult(
                    name="ASR初始化",
                    code="B2",
                    value_ms=duration,
                    rating=rating,
                    is_hotspot=is_hotspot,
                    is_core_hotspot=is_core,
                ))

        # B3 ASR 音频发送
        if self.has_event("asr_audio_send_start") and self.has_event("asr_audio_send_complete"):
            duration = self.get_duration("asr_audio_send_start", "asr_audio_send_complete")
            if duration is not None:
                rating, is_hotspot, is_core = MetricsCalculator.rate(
                    duration, ASR_THRESHOLDS["B3_asr_audio_send"]
                )
                results.append(MetricResult(
                    name="ASR音频发送",
                    code="B3",
                    value_ms=duration,
                    rating=rating,
                    is_hotspot=is_hotspot,
                    is_core_hotspot=is_core,
                ))

        # B4 ASR 云端处理 (核心)
        if self.has_event("asr_commit_sent") and self.has_event("asr_result_delivered"):
            duration = self.get_duration("asr_commit_sent", "asr_result_delivered", use_first=True)
            if duration is not None:
                rating, is_hotspot, is_core = MetricsCalculator.rate(
                    duration, ASR_THRESHOLDS["B4_asr_cloud_process"]
                )
                results.append(MetricResult(
                    name="ASR云端处理",
                    code="B4",
                    value_ms=duration,
                    rating=rating,
                    is_hotspot=is_hotspot,
                    is_core_hotspot=is_core,
                ))

        return results

    def calculate_l2_gap_metrics(self) -> List[MetricResult]:
        """计算 Gap 衔接指标 (级联特有隐藏瓶颈)"""
        results = []

        # G1 ASR→LLM 衔接
        if self.has_event("asr_result_delivered"):
            asr_end = "asr_result_delivered"
        elif self.has_event("asr_final_received"):
            asr_end = "asr_final_received"
        else:
            # Fallback: ASR events missing (e.g. after barge-in reset cleared them).
            # Use speech_end as proxy — captures lock-wait + ASR finalization.
            speech_end = self._get_speech_end_event()
            asr_end = speech_end

        if asr_end and self.has_event("llm_start"):
            duration = self.get_duration(asr_end, "llm_start", use_first=True)
            if duration is not None and duration > 1.0:
                rating, is_hotspot, is_core = MetricsCalculator.rate(
                    duration, GAP_THRESHOLDS["G1_asr_llm_gap"]
                )
                g1_name = "ASR→LLM衔接(含等待)" if asr_end != "asr_result_delivered" and asr_end != "asr_final_received" else "ASR→LLM衔接"
                results.append(MetricResult(
                    name=g1_name,
                    code="G1",
                    value_ms=duration,
                    rating=rating,
                    is_hotspot=is_hotspot,
                    is_core_hotspot=is_core,
                ))

        # G2 LLM→TTS 衔接 (仅监控)
        # 流式模式下 tts_start 早于 llm_complete，G2 可能为负值
        # 关键路径法：G2 不计入 TTFB 公式，仅用于监控衔接效率
        if self.has_event("llm_complete") and self.has_event("tts_start"):
            duration = self.get_duration("llm_complete", "tts_start")
            if duration is not None:
                results.append(MetricResult(
                    name="LLM→TTS衔接(llm结束输出 到 tts开始处理)",
                    code="G2",
                    value_ms=duration,
                    rating=RatingLevel.MONITOR_ONLY,
                    is_hotspot=False,
                    is_core_hotspot=False,
                    is_monitor_only=True,
                ))

        return results

    def calculate_l2_llm_metrics(self) -> List[MetricResult]:
        """计算 LLM 阶段 L2 指标"""
        results = []

        # C1 LLM 请求发送 (llm_start → llm_stream_opened，包含请求构建+发送)
        if self.has_event("llm_start") and self.has_event("llm_stream_opened"):
            duration = self.get_duration("llm_start", "llm_stream_opened", use_first=True)
            if duration is not None:
                rating, is_hotspot, is_core = MetricsCalculator.rate(
                    duration, LLM_THRESHOLDS["C1_llm_request_send"]
                )
                results.append(MetricResult(
                    name="LLM请求发送(发送请求 到 llm数据流打开)",
                    code="C1",
                    value_ms=duration,
                    rating=rating,
                    is_hotspot=is_hotspot,
                    is_core_hotspot=is_core,
                ))

        # C2 LLM 串行生成 (核心热点)
        # 关键路径法：C2 = llm_stream_opened → tts_start
        # 这是 LLM 对 TTFB 的串行贡献部分（首个句子就绪前的 LLM 时间）
        # 如果没有 tts_start（批处理模式），回退到 llm_complete
        # use_first=True: multi-segment TTS fires tts_start per segment; C2 must
        # stop at the FIRST tts_start on the TTFB critical path.
        if self.has_event("llm_stream_opened"):
            c2_end = "tts_start" if self.has_event("tts_start") else (
                "llm_complete" if self.has_event("llm_complete") else None
            )
            if c2_end:
                use_first_c2 = c2_end == "tts_start"
                duration = self.get_duration("llm_stream_opened", c2_end, use_first=use_first_c2)
                if duration is not None:
                    rating, is_hotspot, is_core = MetricsCalculator.rate(
                        duration, LLM_THRESHOLDS["C2_llm_first_token"]
                    )
                    results.append(MetricResult(
                        name="LLM串行生成(llm数据流打开 到 tts开始处理)",
                        code="C2",
                        value_ms=duration,
                        rating=rating,
                        is_hotspot=is_hotspot,
                        is_core_hotspot=is_core,
                    ))

        # C3 LLM 并行生成 (仅监控)
        # 关键路径法：C3 = tts_start → llm_complete
        # 这是 LLM 在 TTS 已启动后继续生成的时间（与 TTS 并行）
        if self.has_event("tts_start") and self.has_event("llm_complete"):
            duration = self.get_duration("tts_start", "llm_complete", use_first=True)
            if duration is not None:
                results.append(MetricResult(
                    name="LLM并行生成(tts开始处理 到 llm完成)",
                    code="C3",
                    value_ms=duration,
                    rating=RatingLevel.MONITOR_ONLY,
                    is_hotspot=False,
                    is_core_hotspot=False,
                    is_monitor_only=True,
                    is_parallel=True,
                ))

        return results

    def _is_local_tts(self) -> bool:
        """检测是否为本地 TTS 模式 (如 Piper)。

        判定逻辑：有 tts_model_generation_start 但没有 tts_ws_connected。
        """
        return (
            self.has_event("tts_model_generation_start")
            and not self.has_event("tts_ws_connected")
            and not self.has_event("tts_ws_reused")
        )

    def _is_local_asr(self) -> bool:
        """检测是否为本地 ASR 模式 (如 Zipformer sherpa-onnx)。

        Checks the original event name directly to avoid alias collision:
        asr_ws_reused and asr_local_ready both map to canonical "asr_reuse",
        so has_event() would give false positives for cloud ASR reuse.
        """
        return any(e["name"] == "asr_local_ready" for e in self.events)

    def calculate_l2_tts_metrics(self) -> List[MetricResult]:
        """计算 TTS 阶段 L2 指标（自动识别云端/本地 TTS）"""
        if self._is_local_tts():
            return self._calculate_l2_tts_local_metrics()
        return self._calculate_l2_tts_cloud_metrics()

    def _calculate_l2_tts_local_metrics(self) -> List[MetricResult]:
        """计算本地 TTS (Piper) 的 L2 指标。"""
        results = []

        # D1 TTS 连接 (本地模式: 0ms, 无需连接)
        results.append(MetricResult(
            name="TTS连接(本地)",
            code="D1",
            value_ms=0.0,
            rating=RatingLevel.EXCELLENT,
            is_hotspot=False,
            is_core_hotspot=False,
            is_reuse=True,
        ))

        # D2 TTS 线程启动 (tts_start → tts_model_generation_start)
        # use_first=True: streaming multi-segment TTS fires these per segment;
        # D2 measures the FIRST segment's thread start on the TTFB critical path.
        if self.has_event("tts_start") and self.has_event("tts_model_generation_start"):
            duration = self.get_duration("tts_start", "tts_model_generation_start", use_first=True)
            if duration is not None:
                rating, is_hotspot, is_core = MetricsCalculator.rate(
                    duration, TTS_LOCAL_THRESHOLDS["D2_tts_thread_start"]
                )
                results.append(MetricResult(
                    name="TTS线程启动",
                    code="D2",
                    value_ms=duration,
                    rating=rating,
                    is_hotspot=is_hotspot,
                    is_core_hotspot=is_core,
                ))

        # D4 TTS 模型推理首音频 (核心热点)
        # tts_model_generation_start → tts_first_chunk_ready
        # use_first=True: D3 measures the FIRST segment's inference on the TTFB critical path.
        if self.has_event("tts_model_generation_start") and self.has_event("tts_first_chunk_ready"):
            duration = self.get_duration("tts_model_generation_start", "tts_first_chunk_ready", use_first=True)
            if duration is not None:
                rating, is_hotspot, is_core = MetricsCalculator.rate(
                    duration, TTS_LOCAL_THRESHOLDS["D3_tts_model_inference"]
                )
                results.append(MetricResult(
                    name="TTS模型推理(开始生成 到 首音频就绪)",
                    code="D3",
                    value_ms=duration,
                    rating=rating,
                    is_hotspot=is_hotspot,
                    is_core_hotspot=is_core,
                ))

        # D5 TTS 完整生成 (仅监控)
        if self.has_event("tts_model_generation_start") and self.has_event("tts_model_generation_complete"):
            duration = self.get_duration("tts_model_generation_start", "tts_model_generation_complete")
            if duration is not None:
                results.append(MetricResult(
                    name="TTS完整生成(第一句)",
                    code="D4",
                    value_ms=duration,
                    rating=RatingLevel.MONITOR_ONLY,
                    is_hotspot=False,
                    is_core_hotspot=False,
                    is_monitor_only=True,
                ))

        return results

    def _calculate_l2_tts_cloud_metrics(self) -> List[MetricResult]:
        """计算云端 TTS (Qwen Realtime) 的 L2 指标。"""
        results = []

        # D1 TTS 连接建立 (从 tts_start 开始，包含准备等待时间)
        if self.has_event("tts_ws_reused"):
            # reuse 时显示 0ms（连接已就绪）
            results.append(MetricResult(
                name="TTS连接建立",
                code="D1",
                value_ms=0.0,
                rating=RatingLevel.EXCELLENT,
                is_hotspot=False,
                is_core_hotspot=False,
                is_reuse=True,
            ))
        elif self.has_event("tts_start") and self.has_event("tts_ws_connected"):
            # 使用 tts_start → tts_ws_connected（包含准备等待）
            duration = self.get_duration("tts_start", "tts_ws_connected")
            if duration is not None:
                rating, is_hotspot, is_core = MetricsCalculator.rate(
                    duration, TTS_THRESHOLDS["D1_tts_connect"]
                )
                results.append(MetricResult(
                    name="TTS连接建立",
                    code="D1",
                    value_ms=duration,
                    rating=rating,
                    is_hotspot=is_hotspot,
                    is_core_hotspot=is_core,
                ))

        # D2 TTS 初始化
        if self.has_event("tts_ws_connected") and self.has_event("tts_session_update_sent"):
            duration = self.get_duration("tts_ws_connected", "tts_session_update_sent")
            if duration is not None:
                rating, is_hotspot, is_core = MetricsCalculator.rate(
                    duration, TTS_THRESHOLDS["D2_tts_init"]
                )
                results.append(MetricResult(
                    name="TTS初始化",
                    code="D2",
                    value_ms=duration,
                    rating=rating,
                    is_hotspot=is_hotspot,
                    is_core_hotspot=is_core,
                ))

        # D3 TTS 首音频 (核心热点)
        if self.has_event("tts_commit_sent") and self.has_event("tts_first_chunk_ready"):
            duration = self.get_duration("tts_commit_sent", "tts_first_chunk_ready")
            if duration is not None:
                rating, is_hotspot, is_core = MetricsCalculator.rate(
                    duration, TTS_THRESHOLDS["D3_tts_first_audio"]
                )
                results.append(MetricResult(
                    name="TTS首音频",
                    code="D3",
                    value_ms=duration,
                    rating=rating,
                    is_hotspot=is_hotspot,
                    is_core_hotspot=is_core,
                ))

        # D4 TTS 流式输出 (仅监控)
        if self.has_event("tts_first_chunk_ready") and self.has_event("tts_finish_event_received"):
            duration = self.get_duration("tts_first_chunk_ready", "tts_finish_event_received")
            if duration is not None:
                results.append(MetricResult(
                    name="TTS流式输出",
                    code="D4",
                    value_ms=duration,
                    rating=RatingLevel.MONITOR_ONLY,
                    is_hotspot=False,
                    is_core_hotspot=False,
                    is_monitor_only=True,
                ))

        return results

    def calculate_l2_playback_metrics(self) -> List[MetricResult]:
        """计算播放阶段 L2 指标

        自动检测批处理/流式模式：
        - 批处理模式：audio_playback_started 在 tts_audio_queued 之后
          E1 = 音频入队 (tts_first_chunk_ready → tts_audio_queued)
          E2 = 播放启动 (tts_audio_queued → audio_playback_started)
        - 流式模式：audio_playback_started 在 tts_audio_queued 之前
          E1 = 首块播放延迟 (tts_first_chunk_ready → audio_playback_started)
          E2 = 播放排空 (tts_audio_queued → playback_complete, 仅监控)
        """
        results = []

        # Detect streaming mode: playback starts before all audio is queued
        # Use first-match for mode detection and E1 to avoid multi-segment TTS skew.
        playback_ts = self._get_event_timestamp("audio_playback_started", use_first=True)
        audio_queued_ts = self._get_event_timestamp("tts_audio_queued")
        is_streaming = (
            playback_ts is not None
            and (audio_queued_ts is None or playback_ts < audio_queued_ts)
        )

        if is_streaming:
            # ── Streaming mode metrics ──

            # E1 首块播放延迟 (tts_first_chunk_ready → audio_playback_started)
            # use_first=True: both events must be from the FIRST TTS segment
            if self.has_event("tts_first_chunk_ready") and playback_ts:
                first_chunk_ts = self._get_event_timestamp("tts_first_chunk_ready", use_first=True)
                if first_chunk_ts and playback_ts >= first_chunk_ts:
                    duration = (playback_ts - first_chunk_ts) * 1000
                    rating, is_hotspot, is_core = MetricsCalculator.rate(
                        duration, PLAYBACK_THRESHOLDS["E1_audio_queue"]
                    )
                    results.append(MetricResult(
                        name="首块播放延迟",
                        code="E1",
                        value_ms=duration,
                        rating=rating,
                        is_hotspot=is_hotspot,
                        is_core_hotspot=is_core,
                    ))

            # E2 播放排空 (tts_audio_queued → playback_complete, 仅监控)
            if audio_queued_ts and self.has_event("playback_complete"):
                complete_ts = self._get_event_timestamp("playback_complete")
                if complete_ts and complete_ts > audio_queued_ts:
                    drain_ms = (complete_ts - audio_queued_ts) * 1000
                    results.append(MetricResult(
                        name="播放排空",
                        code="E2",
                        value_ms=drain_ms,
                        rating=RatingLevel.MONITOR_ONLY,
                        is_hotspot=False,
                        is_core_hotspot=False,
                        is_monitor_only=True,
                    ))
        else:
            # ── Batch mode metrics ──

            # E1 音频入队 (tts_first_chunk_ready → tts_audio_queued)
            if self.has_event("tts_first_chunk_ready") and audio_queued_ts:
                first_chunk_ts = self._get_event_timestamp("tts_first_chunk_ready")
                if first_chunk_ts and audio_queued_ts > first_chunk_ts:
                    duration = self.get_duration("tts_first_chunk_ready", "tts_audio_queued")
                    if duration is not None:
                        rating, is_hotspot, is_core = MetricsCalculator.rate(
                            duration, PLAYBACK_THRESHOLDS["E1_audio_queue"]
                        )
                        results.append(MetricResult(
                            name="音频入队",
                            code="E1",
                            value_ms=duration,
                            rating=rating,
                            is_hotspot=is_hotspot,
                            is_core_hotspot=is_core,
                        ))

            # E2 播放启动 (tts_audio_queued → audio_playback_started)
            if audio_queued_ts and playback_ts and playback_ts > audio_queued_ts:
                duration = self.get_duration("tts_audio_queued", "audio_playback_started")
                if duration is not None:
                    rating, is_hotspot, is_core = MetricsCalculator.rate(
                        duration, PLAYBACK_THRESHOLDS["E2_playback_start"]
                    )
                    results.append(MetricResult(
                        name="播放启动",
                        code="E2",
                        value_ms=duration,
                        rating=rating,
                        is_hotspot=is_hotspot,
                        is_core_hotspot=is_core,
                    ))

        return results

    def _get_event_timestamp(self, event_name: str, use_first: bool = False) -> Optional[float]:
        """获取事件时间戳。

        Args:
            use_first: True=返回第一次匹配（适用于多段TTS的首段关键路径指标），
                       False=返回最后一次匹配（默认，适用于 preserved 事件场景）。
        """
        canonical = self.EVENT_ALIASES.get(event_name, event_name)
        result: float | None = None
        for e in self.events:
            if e.get("canonical", e["name"]) == canonical or e["name"] == event_name:
                if use_first and result is None:
                    return e["timestamp"]
                result = e["timestamp"]
        return result

    # ========================================================================
    # 辅助方法
    # ========================================================================

    def _get_speech_end_event(self) -> str:
        """获取 speech_end 事件名称"""
        if self.has_event("vad_speech_end"):
            return "vad_speech_end"
        elif self.has_event("user_stop_click"):
            return "user_stop_click"
        return ""

    def _get_response_start_event(self) -> str:
        """获取 response_start 事件名称"""
        if self.has_event("llm_first_speech_chunk"):
            return "llm_first_speech_chunk"
        elif self.has_event("llm_first_token"):
            return "llm_first_token"
        elif self.has_event("llm_complete"):
            return "llm_complete"
        return ""

    # ========================================================================
    # 热点定位
    # ========================================================================

    def identify_hotspots(self) -> List[Tuple[str, float, str]]:
        """识别性能热点

        Returns:
            List of (code, value_ms, description)
        """
        hotspots = []

        all_metrics = []
        all_metrics.extend(self.calculate_l2_asr_metrics())
        all_metrics.extend(self.calculate_l2_gap_metrics())
        all_metrics.extend(self.calculate_l2_llm_metrics())
        all_metrics.extend(self.calculate_l2_tts_metrics())
        all_metrics.extend(self.calculate_l2_playback_metrics())

        core_hotspots = [m for m in all_metrics if m.is_core_hotspot and m.is_hotspot]
        hidden_hotspots = [m for m in all_metrics if m.code.startswith("G") and m.is_hotspot]
        normal_hotspots = [m for m in all_metrics if m.is_hotspot and not m.is_core_hotspot and not m.code.startswith("G")]

        for m in core_hotspots:
            desc = self._get_hotspot_description(m.code)
            hotspots.append((m.code, m.value_ms, desc))

        for m in hidden_hotspots:
            desc = f"{m.name}存在隐藏瓶颈"
            hotspots.append((m.code, m.value_ms, desc))

        for m in normal_hotspots:
            desc = self._get_hotspot_description(m.code)
            hotspots.append((m.code, m.value_ms, desc))

        return hotspots

    def _get_hotspot_description(self, code: str) -> str:
        """获取热点描述"""
        is_local_tts = self._is_local_tts()
        is_local_asr = self._is_local_asr()
        descriptions = {
            "B4": "ASR本地处理延迟" if is_local_asr else "ASR云端处理延迟",
            "C2": "LLM串行推理是核心热点",
            "D3": "TTS本地模型推理延迟" if is_local_tts else "TTS生成延迟",
            "B1": "ASR连接延迟",
            "B2": "ASR初始化延迟",
            "B3": "ASR音频上传延迟",
            "C1": "LLM请求发送延迟",
            "D1": "TTS连接延迟",
            "D2": "TTS线程启动延迟" if is_local_tts else "TTS初始化延迟",
            "E1": "音频入队延迟",
            "E2": "播放启动延迟",
            "G1": "ASR→LLM衔接延迟",
        }
        return descriptions.get(code, f"{code}存在延迟")

    # ========================================================================
    # 指标追溯验证
    # ========================================================================

    def validate_trace_formula(self) -> Dict[str, Any]:
        """验证 L1 指标追溯公式（关键路径法）

        公式: TTFB = B4 + G1 + C1 + C2 + D2 + D4
        关键路径法：
        - B1/B2/B3: 用户说话期间并行完成，不计入 TTFB
        - B5: 已删除（B4 覆盖到 asr_result_delivered）
        - C2: llm_stream_opened → tts_start（串行部分）
        - C3: tts_start → llm_complete（并行，仅监控）
        - G2: 流式模式下为负值，仅监控
        - E1/E2: 流式模式下顺序颠倒，不计入 TTFB

        Returns:
            {actual_ttfb, calculated_sum, components, deviation_ms, deviation_pct, timeline_gaps}
        """
        speech_end = self._get_speech_end_event()
        first_audio = "audio_playback_started"

        if not speech_end or not self.has_event(first_audio):
            return {"valid": False, "reason": "Missing boundary events"}

        # A new utterance can be captured before the cancelled/current turn prints
        # its report. The report belongs to the first input/output chain after
        # reset(), not to later VAD events appended by the next utterance.
        actual_ttfb = self.get_duration(speech_end, first_audio, use_first=True)
        if actual_ttfb is None:
            return {"valid": False, "reason": "Cannot calculate TTFB"}

        components: Dict[str, float] = {}
        timeline_events: List[Dict[str, Any]] = []

        is_local_tts = self._is_local_tts()
        is_local_asr = self._is_local_asr()

        # 关键路径事件链（串行路径）
        # Local ASR: B4 runs in parallel with VAD, skip B4 in event chain.
        # Use extended G1 (speech_end → llm_start) instead of B4 + G1.
        if is_local_asr:
            event_chain = [
                ("speech_end", speech_end),
                ("G1_end", "llm_start"),
                ("C1_start", "llm_start"),
                ("C1_end", "llm_stream_opened"),
                ("LLM_first", "llm_first_token"),  # 中间节点，仅展示
                ("C2_end", "tts_start"),  # C2 关键路径截断到 tts_start
            ]
        else:
            event_chain = [
                ("speech_end", speech_end),
                ("B4_start", "asr_commit_sent"),
                ("B4_end", "asr_result_delivered"),
                ("G1_end", "llm_start"),
                ("C1_start", "llm_start"),
                ("C1_end", "llm_stream_opened"),
                ("LLM_first", "llm_first_token"),
                ("C2_end", "tts_start"),
            ]

        # TTS 阶段：根据云端/本地模式构建不同事件链
        if is_local_tts:
            event_chain.extend([
                ("D2_end", "tts_model_generation_start"),
                ("D3_end", "tts_first_chunk_ready"),
                ("first_audio", "audio_playback_started"),
            ])
        else:
            event_chain.extend([
                ("D1_end", "tts_ws_connected"),
                ("D2_end", "tts_session_update_sent"),
                ("D3_start", "tts_commit_sent"),
                ("D3_end", "tts_first_chunk_ready"),
                ("first_audio", "audio_playback_started"),
            ])

        # 构建时间线
        prev_event = None
        tts_reused = self.has_event("tts_ws_reused") or is_local_tts

        for label, event_name in event_chain:
            exists = self.has_event(event_name)
            timestamp = None
            if exists:
                # TTS and playback events may repeat across multi-segment streaming.
                # Use first-match for critical path events to stay on TTFB timeline.
                use_first_ts = label in {
                    "speech_end",
                    "B4_start",
                    "B4_end",
                    "G1_end",
                    "C1_start",
                    "C1_end",
                    "LLM_first",
                    "C2_end",
                    "D1_end",
                    "D2_end",
                    "D3_start",
                    "D3_end",
                    "first_audio",
                }
                timestamp = self._get_event_timestamp(event_name, use_first=use_first_ts)

            is_reuse_marker = False
            if label.startswith("D1") or label.startswith("D2"):
                if tts_reused and not exists:
                    is_reuse_marker = True
                    exists = True

            timeline_events.append({
                "label": label,
                "event": event_name,
                "exists": exists,
                "timestamp": timestamp,
                "is_reuse": is_reuse_marker,
            })

            if prev_event and exists and timeline_events[-2]["exists"]:
                prev_ts = timeline_events[-2].get("timestamp")
                if prev_ts and timestamp:
                    duration = (timestamp - prev_ts) * 1000
                    timeline_events[-1]["duration_from_prev_ms"] = duration

            prev_event = event_name if exists else prev_event

        # 计算各阶段指标（仅关键路径串行部分）
        if is_local_asr:
            # Local ASR: extended G1 = speech_end → llm_start (includes B4 + lock wait)
            g1_extended = self.get_duration(speech_end, "llm_start", use_first=True)
            if g1_extended:
                components["G1"] = g1_extended
        else:
            # Cloud ASR: B4 + G1 decomposition
            asr_metrics = self.calculate_l2_asr_metrics()
            for m in asr_metrics:
                if m.code == "B4":
                    components[m.code] = m.value_ms

            if self.has_event("asr_result_delivered") and self.has_event("llm_start"):
                g1 = self.get_duration("asr_result_delivered", "llm_start", use_first=True)
                if g1:
                    components["G1"] = g1

        llm_metrics = self.calculate_l2_llm_metrics()
        for m in llm_metrics:
            if m.code in ("C1", "C2"):
                components[m.code] = m.value_ms

        # TTS: D2 + D3（关键路径串行部分）
        tts_metrics = self.calculate_l2_tts_metrics()
        for m in tts_metrics:
            if m.code in ("D2", "D3"):
                components[m.code] = m.value_ms

        calculated_sum = sum(components.values())
        deviation_ms = actual_ttfb - calculated_sum
        deviation_pct = (deviation_ms / actual_ttfb) * 100 if actual_ttfb > 0 else 0

        return {
            "valid": True,
            "actual_ttfb_ms": actual_ttfb,
            "calculated_sum_ms": calculated_sum,
            "components": components,
            "deviation_ms": deviation_ms,
            "deviation_pct": round(deviation_pct, 1),
            "timeline_events": timeline_events,
            "tts_reused": tts_reused,
        }

    def print_timeline_validation(self) -> None:
        """打印时间线连续性验证详情（用于运行时证据）"""
        result = self.validate_trace_formula()
        if not result.get("valid"):
            logger.warning(f"时间线验证失败: {result.get('reason')}")
            return

        logger.info("=" * 80)
        logger.info("【时间线连续性验证】")
        logger.info("=" * 80)

        # 打印时间线事件链
        logger.info("事件链:")
        prev_ts = None
        timeline = result.get("timeline_events", [])

        for entry in timeline:
            label = entry["label"]
            event = entry["event"]
            exists = entry["exists"]
            ts = entry.get("timestamp")
            is_reuse = entry.get("is_reuse", False)

            if is_reuse:
                status = "🔄 reuse"
                ts_str = "N/A (reuse)"
            elif exists:
                status = "✅"
                ts_str = f"@{ts:.3f}" if ts else "N/A"
            else:
                status = "❌ 缺失"
                ts_str = "N/A"

            if prev_ts and ts:
                gap = (ts - prev_ts) * 1000
                gap_str = f"[+{gap:.1f}ms]"
            else:
                gap_str = ""

            logger.info(f"  {label:12s} {event:30s} {status} {ts_str} {gap_str}")
            if ts:
                prev_ts = ts

        # 打印指标组成
        logger.info("")
        logger.info("指标组成:")
        components = result.get("components", {})
        for code, value in sorted(components.items()):
            logger.info(f"  {code}: {value:.1f}ms")

        # 打印偏差分析
        logger.info("")
        logger.info("偏差分析:")
        actual = result.get("actual_ttfb_ms", 0)
        calc = result.get("calculated_sum_ms", 0)
        dev = result.get("deviation_ms", 0)
        pct = result.get("deviation_pct", 0)

        logger.info(f"  实际 TTFB:    {actual:.1f}ms")
        logger.info(f"  计算总和:     {calc:.1f}ms")
        logger.info(f"  偏差:         {dev:.1f}ms ({pct:.1f}%)")

        # 分析偏差来源
        if abs(pct) > 5:
            logger.warning(f"  ⚠️ 偏差超过5%，分析原因:")
            # 检查 TTS 准备时间（tts_start → tts_commit_sent）
            if self.has_event("tts_start") and self.has_event("tts_commit_sent"):
                tts_prep = self.get_duration("tts_start", "tts_commit_sent")
                if tts_prep and tts_prep > 10:
                    logger.warning(f"     - TTS准备时间未计入: {tts_prep:.1f}ms (tts_start→commit)")

            # 检查 ASR commit 前的等待
            if self.has_event("vad_speech_end") and self.has_event("asr_commit_sent"):
                asr_wait = self.get_duration("vad_speech_end", "asr_commit_sent")
                if asr_wait and asr_wait > 10:
                    logger.warning(f"     - ASR提交前等待: {asr_wait:.1f}ms (speech_end→commit)")

            # 检查 barge-in 锁等待 (speech_end → asr_result_delivered 时间过长)
            if self.has_event("vad_speech_end") and self.has_event("asr_result_delivered"):
                lock_wait = self.get_duration("vad_speech_end", "asr_result_delivered")
                if lock_wait and lock_wait > 200:
                    logger.info(f"     - Barge-in 等待: {lock_wait:.1f}ms (speech_end→asr_result_delivered)")
        else:
            logger.info(f"  ✅ 偏差小于5%，时间线连续性良好")

        logger.info("=" * 80)

    # ========================================================================
    # 报告输出
    # ========================================================================

    def print_summary(self) -> None:
        """输出标准性能报告 (符合设计文档格式)"""
        if not self.events:
            logger.warning("No timing events recorded")
            return

        cancelled_tag = "  [BARGE-IN 中断 - 仅供参考]" if self._cancelled else ""

        logger.info("=" * 80)
        logger.info(f"性能报告 - Cascade{cancelled_tag}")
        logger.info(f"轮次: #{self.turn_number}")
        logger.info("=" * 80)

        # L1 指标
        l1_metrics = self.calculate_l1_metrics()
        logger.info("")
        logger.info("【一级指标 - 核心】")
        for m in l1_metrics:
            icon = MetricsCalculator.format_rating_icon(m.rating)
            extra = ""
            if m.code == "L2":
                extra = " ← 核心"
            label = f"{m.name}"
            dots = "." * (35 - len(label))
            rating_str = m.rating.value if m.rating != RatingLevel.MONITOR_ONLY else ""
            logger.info(f"  {m.code} {label}{dots}  {m.value_ms:>7.1f}ms  {icon} {rating_str}{extra}")

        # L2 ASR 阶段
        asr_metrics = self.calculate_l2_asr_metrics()
        logger.info("")
        logger.info("【二级指标 - ASR阶段】")
        for m in asr_metrics:
            self._log_metric(m)

        # L2 Gap 衔接
        gap_metrics = self.calculate_l2_gap_metrics()
        logger.info("")
        logger.info("【二级指标 - Gap衔接】")
        for m in gap_metrics:
            self._log_metric(m, is_gap=True)

        # L2 LLM 阶段
        llm_metrics = self.calculate_l2_llm_metrics()
        logger.info("")
        logger.info("【二级指标 - LLM阶段】")
        for m in llm_metrics:
            self._log_metric(m)

        # L2 TTS 阶段
        tts_metrics = self.calculate_l2_tts_metrics()
        logger.info("")
        logger.info("【二级指标 - TTS阶段】")
        for m in tts_metrics:
            self._log_metric(m)

        # L2 播放阶段
        playback_metrics = self.calculate_l2_playback_metrics()
        logger.info("")
        logger.info("【二级指标 - 播放阶段】")
        for m in playback_metrics:
            self._log_metric(m)

        # 指标追溯
        trace = self.validate_trace_formula()
        if trace.get("valid"):
            logger.info("")
            logger.info("【指标追溯】")
            components_str = " + ".join(
                f"{k}({v:.0f}ms)" for k, v in trace["components"].items()
            )
            logger.info(f"  L2 ≈ {components_str}")
            logger.info(f"     ≈ {trace['calculated_sum_ms']:.0f}ms")
            logger.info(f"  实际 L2 = {trace['actual_ttfb_ms']:.0f}ms")
            logger.info(f"  偏差 = {trace['deviation_ms']:.0f}ms ({trace['deviation_pct']}%)")

        # 热点定位 (cancelled 轮次跳过 SLO 评分)
        if not self._cancelled:
            hotspots = self.identify_hotspots()
            if hotspots:
                logger.info("")
                logger.info("【热点定位】")
                for i, (code, value, desc) in enumerate(hotspots[:5], 1):
                    threshold = self._get_threshold_for_code(code)
                    logger.info(f"  {i}. {code}={value:.0f}ms >{threshold}ms → {desc}")

        # 时间线连续性验证（证据兜底）
        self.print_timeline_validation()

    def _log_metric(self, m: MetricResult, is_gap: bool = False) -> None:
        """输出单个指标"""
        icon = MetricsCalculator.format_rating_icon(m.rating)
        dots = "." * (35 - len(m.name))

        extras: List[str] = []
        if m.is_reuse:
            extras.append("(reuse)")
        if m.is_parallel:
            extras.append("(∥并行)")
        if m.is_monitor_only:
            extras.append("(仅监控)")
        if m.is_core_hotspot and m.is_hotspot:
            extras.append("核心热点")
        elif is_gap and m.is_hotspot:
            extras.append("隐藏热点")
        elif m.is_hotspot:
            extras.append("热点")

        extra_str = " ".join(extras)
        rating_str = m.rating.value if m.rating != RatingLevel.MONITOR_ONLY else ""

        logger.info(f"  {m.code} {m.name}{dots}  {m.value_ms:>7.1f}ms  {icon} {rating_str}  {extra_str}")

    def _get_threshold_for_code(self, code: str) -> int:
        """获取热点阈值"""
        thresholds_map = {
            "B1": 500, "B2": 200, "B3": 200, "B4": 300,
            "G1": 50, "G2": 50,
            "C1": 100, "C2": 500,
            "D1": 500, "D2": 200, "D3": 300,
            "E1": 50, "E2": 100,
        }
        return thresholds_map.get(code, 100)


# Global tracker instance
tracker = LatencyTracker()
