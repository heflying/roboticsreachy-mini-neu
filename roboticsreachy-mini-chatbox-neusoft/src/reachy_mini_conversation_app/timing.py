"""Latency tracking for Realtime voice APIs.

Implements the unified performance metrics system defined in docs/性能指标设计.md v2.0

Level 1 Metrics (一级指标): 3 core metrics, architecture-agnostic, stable
Level 2 Metrics (二级指标): Architecture-specific breakdown for hotspot detection
"""

from __future__ import annotations
import time
import logging
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


# ==============================================================================
# BENCHMARK THRESHOLDS (基准阈值 - 精简版 v2.0)
# ==============================================================================

BENCHMARKS = {
    # Level 1: Core Metrics (一级核心指标 - 跨架构统一)
    "L1_response_start": {"excellent": 150, "good": 300, "acceptable": 500},
    "L2_ttfb": {"excellent": 500, "good": 800, "acceptable": 1200},
    "L3_transcript": {"excellent": 200, "good": 400, "acceptable": 600},
    # Level 2: Realtime Architecture (二级 - Realtime 架构)
    "R1_ws_connect": {"excellent": 200, "good": 500, "acceptable": 1000},
    "P2_parallel_process": {"excellent": 300, "good": 500, "acceptable": 800},
    "Q1_audio_transfer": {"excellent": 100, "good": 200, "acceptable": 300},  # 音频生成+传输需≥100ms
}


def _rating(value: float, benchmark_key: str) -> str:
    """Return rating based on benchmark thresholds."""
    bench = BENCHMARKS.get(benchmark_key, {"excellent": 100, "good": 200, "acceptable": 300})
    if value <= bench["excellent"]:
        return "✅ EXCELLENT"
    elif value <= bench["good"]:
        return "👍 GOOD"
    elif value <= bench["acceptable"]:
        return "⚠️ ACCEPTABLE"
    return "❌ NEEDS IMPROVEMENT"


def _is_hotspot(value: float, benchmark_key: str) -> str:
    """Return hotspot marker if exceeds good threshold."""
    bench = BENCHMARKS.get(benchmark_key, {"excellent": 100, "good": 200, "acceptable": 300})
    if value > bench["good"]:
        return " ⚠️ 热点"
    return ""


# ==============================================================================
# UNIFIED EVENT NAMES (一级统一事件名 - 精简版)
# ==============================================================================

L1_EVENTS = {
    "speech_end": "User stops speaking (VAD speech_stopped)",
    "response_start": "System starts generating response",
    "first_audio": "First audio chunk arrives (TTFB)",
    "transcript_show": "Transcript text displayed",
}

L2_REALTIME_EVENTS = {
    "R1_ws_connect_done": "WebSocket connection established",
    "P2_parallel_process": "Server parallel processing (ASR+LLM+TTS)",
    "Q1_first_audio_received": "First audio packet received locally",
}


class LatencyTracker:
    """Centralized latency tracking following unified metrics system v2.0.

    L1: 3 core metrics (cross-architecture, stable)
    L2: 3 internal metrics for Realtime architecture

    Connection events (R1/R2) are stored separately and persist across turns.
    Reconnection events are tracked for fault monitoring.
    """

    def __init__(self) -> None:
        """Initialize latency tracker."""
        self.events: List[Dict[str, Any]] = []  # Per-turn events
        self.connection_events: List[Dict[str, Any]] = []  # Connection phase events (R1/R2)
        self.reconnect_count: int = 0  # 重连次数计数
        self.reconnect_reasons: List[str] = []  # 重连原因记录
        self.start_time: Optional[float] = None
        self.turn_id: int = 0

    def reset(self, turn_id: int = 0) -> None:
        """Reset tracker for new conversation turn. Connection events are preserved."""
        self.events = []  # Only clear per-turn events
        self.start_time = time.perf_counter()
        self.turn_id = turn_id
        logger.info(f"⏱️  [Turn #{turn_id}] LATENCY TRACKING STARTED")

    def mark_reconnect(self, reason: str = "") -> None:
        """Mark a reconnection event for fault monitoring."""
        self.reconnect_count += 1
        self.reconnect_reasons.append(reason)
        logger.warning(f"🔌 [FAULT] WebSocket 重连 #{self.reconnect_count}: {reason}")

    def mark(self, event_name: str, metadata: Optional[Dict[str, Any]] = None, level: int = 1) -> None:
        """Mark a timing event.

        Args:
            event_name: Event identifier
            metadata: Optional metadata dict
            level: 1 for Level 1 (core), 2 for Level 2 (internal)

        Connection events (R1/R2) are stored in connection_events and persist across turns.
        """
        timestamp = time.perf_counter()

        # Connection events (R1/R2) don't need start_time reference
        is_connection_event = event_name.startswith("R1_") or event_name.startswith("R2_")

        if is_connection_event:
            elapsed_ms = 0.0  # Connection events use absolute timestamps
        else:
            if self.start_time is None:
                self.reset()
            assert self.start_time is not None
            elapsed_ms = (timestamp - self.start_time) * 1000

        event = {
            "name": event_name,
            "timestamp": timestamp,
            "elapsed_ms": elapsed_ms,
            "metadata": metadata or {},
            "level": level,
        }

        # Store connection events separately
        if is_connection_event:
            self.connection_events.append(event)
        else:
            self.events.append(event)

        # Format output
        level_prefix = "L1" if level == 1 else "L2"
        metadata_str = ""
        if metadata:
            parts = [f"{k}={v}" for k, v in metadata.items() if k != "transcript"]
            metadata_str = f" ({', '.join(parts)})" if parts else ""

        logger.info(f"⏱️  [{elapsed_ms:7.1f}ms] {level_prefix}:{event_name}{metadata_str}")

    # ==========================================================================
    # Level 1 Core Markers (一级核心指标 - 便捷方法)
    # ==========================================================================

    def mark_speech_end(self) -> None:
        """L1: User stops speaking (计时起点)."""
        self.mark("speech_end", level=1)

    def mark_response_start(self) -> None:
        """L1: Response generation started."""
        self.mark("response_start", level=1)

    def mark_first_audio(self, audio_bytes: int = 0) -> None:
        """L1: First audio chunk arrives (TTFB)."""
        meta = {"audio_bytes": audio_bytes} if audio_bytes else {}
        self.mark("first_audio", meta, level=1)

    def mark_transcript_show(self, transcript: str = "") -> None:
        """L1: Transcript displayed."""
        meta = {"transcript": transcript, "transcript_len": len(transcript)} if transcript else {}
        self.mark("transcript_show", meta, level=1)

    # ==========================================================================
    # Level 2 Internal Markers (二级内部分解 - Realtime 架构)
    # ==========================================================================

    def mark_R1_ws_connect_start(self) -> None:
        """L2 Stage R: WebSocket connection initiated."""
        self.mark("R1_ws_connect_start", level=2)

    def mark_R1_ws_connect_done(self) -> None:
        """L2 Stage R: WebSocket connected."""
        self.mark("R1_ws_connect_done", level=2)

    def mark_R2_session_config_sent(self) -> None:
        """L2 Stage R: Session config sent."""
        self.mark("R2_session_config_sent", level=2)

    def mark_R2_session_config_done(self) -> None:
        """L2 Stage R: Session config confirmed."""
        self.mark("R2_session_config_done", level=2)

    def mark_P2_response_created(self) -> None:
        """L2 Stage P: Response created event (parallel processing start)."""
        self.mark("P2_response_created", level=2)

    def mark_Q1_first_audio_received(self, audio_bytes: int = 0) -> None:
        """L2 Stage Q: First audio received."""
        meta = {"audio_bytes": audio_bytes} if audio_bytes else {}
        self.mark("Q1_first_audio_received", meta, level=2)

    # ==========================================================================
    # Duration Calculation
    # ==========================================================================

    def get_duration(self, start_event: str, end_event: str) -> Optional[float]:
        """Get duration between two events in milliseconds.

        Searches both connection_events (persistent) and events (per-turn).
        """
        start_ts = None
        end_ts = None

        # Combine both event lists for search
        all_events = self.connection_events + self.events

        for event in all_events:
            if event["name"] == start_event and start_ts is None:
                start_ts = event["timestamp"]
            elif event["name"] == end_event and end_ts is None:
                end_ts = event["timestamp"]

        if start_ts is not None and end_ts is not None:
            return float((end_ts - start_ts) * 1000)
        return None

    # ==========================================================================
    # Level 1 Metrics Computation (一级指标计算)
    # ==========================================================================

    def compute_L1_metrics(self) -> Dict[str, Optional[float]]:
        """Compute Level 1 core metrics."""
        return {
            "L1_response_start": self.get_duration("speech_end", "response_start"),
            "L2_ttfb": self.get_duration("speech_end", "first_audio"),
            "L3_transcript": self.get_duration("speech_end", "transcript_show"),
        }

    # ==========================================================================
    # Level 2 Metrics Computation (二级指标计算 - Realtime)
    # ==========================================================================

    def compute_L2_metrics(self) -> Dict[str, Optional[float]]:
        """Compute Level 2 internal metrics for Realtime."""
        return {
            "R1_ws_connect": self.get_duration("R1_ws_connect_start", "R1_ws_connect_done"),
            "P2_parallel_process": self.get_duration("speech_end", "P2_response_created"),
            "Q1_audio_transfer": self.get_duration("response_start", "first_audio"),
        }

    # ==========================================================================
    # Report Output
    # ==========================================================================

    def print_summary(self) -> None:
        """Print standardized performance report (v2.0 精简版)."""
        if not self.events:
            logger.warning("No timing events recorded")
            return

        # Compute metrics
        l1_metrics = self.compute_L1_metrics()
        l2_metrics = self.compute_L2_metrics()

        # =====================================================================
        # REPORT HEADER
        # =====================================================================
        logger.info("=" * 80)
        logger.info(f"性能报告 - Realtime")
        logger.info(f"轮次: #{self.turn_id}")
        logger.info("=" * 80)

        # =====================================================================
        # LEVEL 1: CORE METRICS (一级指标 - 核心)
        # =====================================================================
        logger.info("")
        logger.info("【一级指标 - 核心】")

        # L1: Response Start Latency
        l1_val = l1_metrics.get("L1_response_start")
        if l1_val is not None:
            rating = _rating(l1_val, "L1_response_start")
            logger.info(f"  L1 响应启动延迟......... {l1_val:>7.1f}ms  {rating}")

        # L2: TTFB (KEY METRIC)
        l2_val = l1_metrics.get("L2_ttfb")
        if l2_val is not None:
            rating = _rating(l2_val, "L2_ttfb")
            logger.info(f"  L2 首音延迟 (TTFB)...... {l2_val:>7.1f}ms  {rating}  ← 核心")

        # L3: Transcript Latency
        l3_val = l1_metrics.get("L3_transcript")
        if l3_val is not None:
            rating = _rating(l3_val, "L3_transcript")
            logger.info(f"  L3 转录延迟............. {l3_val:>7.1f}ms  {rating}")

        # =====================================================================
        # LEVEL 2: INTERNAL BREAKDOWN (二级指标 - 内部)
        # =====================================================================
        logger.info("")
        logger.info("【二级指标 - 内部】")

        # R1: WebSocket Connect
        r1_val = l2_metrics.get("R1_ws_connect")
        if r1_val is not None:
            hotspot = _is_hotspot(r1_val, "R1_ws_connect")
            logger.info(f"  R1 WS连接建立........... {r1_val:>7.1f}ms{hotspot}")

        # P2: Parallel Processing
        p2_val = l2_metrics.get("P2_parallel_process")
        if p2_val is not None:
            hotspot = _is_hotspot(p2_val, "P2_parallel_process")
            logger.info(f"  P2 并行处理延迟......... {p2_val:>7.1f}ms{hotspot}")

        # Q1: Audio Transfer
        q1_val = l2_metrics.get("Q1_audio_transfer")
        if q1_val is not None:
            hotspot = _is_hotspot(q1_val, "Q1_audio_transfer")
            logger.info(f"  Q1 音频包传输........... {q1_val:>7.1f}ms{hotspot}")

        # =====================================================================
        # METRIC TRACE (指标追溯)
        # =====================================================================
        logger.info("")
        logger.info("【指标追溯】")
        if l2_val is not None and p2_val is not None and q1_val is not None:
            logger.info(f"  L2 ≈ P2({p2_val:.0f}ms) + Q1({q1_val:.0f}ms) ≈ {p2_val + q1_val:.0f}ms")

        # =====================================================================
        # CONVERSATION CONTENT (对话内容)
        # =====================================================================
        user_transcript = ""
        ai_transcript = ""
        for event in self.events:
            if event["name"] == "transcript_show":
                user_transcript = event["metadata"].get("transcript", "")
            if event["name"] == "Q4_transcript_done":
                ai_transcript = event["metadata"].get("transcript", "")

        if user_transcript or ai_transcript:
            logger.info("")
            logger.info("【对话内容】")
            if user_transcript:
                display = user_transcript[:50] + "..." if len(user_transcript) > 50 else user_transcript
                logger.info(f"  用户说: \"{display}\"")
            if ai_transcript:
                display = ai_transcript[:60] + "..." if len(ai_transcript) > 60 else ai_transcript
                logger.info(f"  AI回复: \"{display}\"")

        # =====================================================================
        # FAULT WARNINGS (故障警告)
        # =====================================================================
        if self.reconnect_count > 0:
            logger.info("")
            logger.warning(f"【故障警告】 WebSocket 重连 {self.reconnect_count} 次")
            for i, reason in enumerate(self.reconnect_reasons, 1):
                logger.warning(f"  #{i}: {reason}")

        logger.info("")
        logger.info("=" * 80)


# Global tracker instance
tracker = LatencyTracker()