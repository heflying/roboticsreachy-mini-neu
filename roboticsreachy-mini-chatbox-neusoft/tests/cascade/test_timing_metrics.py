"""Unit tests for timing.py refactored metrics system (Critical Path Method).

Tests for:
- Rating calculation (threshold boundaries)
- Event recording and duration calculation
- L1/L2 metrics calculation
- Hotspot detection logic
- Trace formula validation (critical path method)
- Standard report output format
- Local TTS (Piper) metrics
"""

import pytest
import time
import logging

from reachy_mini_conversation_app.cascade.timing import (
    LatencyTracker,
    MetricsCalculator,
    MetricResult,
    RatingLevel,
    ThresholdConfig,
    L1_THRESHOLDS,
    ASR_THRESHOLDS,
    ASR_LOCAL_THRESHOLDS,
    GAP_THRESHOLDS,
    LLM_THRESHOLDS,
    TTS_THRESHOLDS,
    TTS_LOCAL_THRESHOLDS,
)


class TestMetricsCalculator:
    """测试评级计算"""

    def test_excellent_rating(self) -> None:
        threshold = L1_THRESHOLDS["L2_ttfb"]
        rating, is_hotspot, _is_core = MetricsCalculator.rate(400.0, threshold)
        assert rating == RatingLevel.EXCELLENT
        assert not is_hotspot

    def test_good_rating(self) -> None:
        threshold = L1_THRESHOLDS["L2_ttfb"]
        rating, is_hotspot, _is_core = MetricsCalculator.rate(600.0, threshold)
        assert rating == RatingLevel.GOOD
        assert not is_hotspot

    def test_acceptable_rating(self) -> None:
        threshold = L1_THRESHOLDS["L2_ttfb"]
        rating, is_hotspot, _is_core = MetricsCalculator.rate(1000.0, threshold)
        assert rating == RatingLevel.ACCEPTABLE
        assert not is_hotspot

    def test_needs_improvement_rating(self) -> None:
        threshold = L1_THRESHOLDS["L2_ttfb"]
        rating, is_hotspot, _is_core = MetricsCalculator.rate(1500.0, threshold)
        assert rating == RatingLevel.NEEDS_IMPROVEMENT
        assert is_hotspot

    def test_core_hotspot_detection(self) -> None:
        threshold = ASR_THRESHOLDS["B4_asr_cloud_process"]  # core_hotspot=True
        _rating, is_hotspot, is_core = MetricsCalculator.rate(400.0, threshold)
        assert is_hotspot  # 400 > 300
        assert is_core

    def test_hidden_hotspot_detection(self) -> None:
        threshold = GAP_THRESHOLDS["G1_asr_llm_gap"]
        _rating, is_hotspot, is_core = MetricsCalculator.rate(60.0, threshold)
        assert is_hotspot  # 60 > 50
        assert not is_core

    def test_monitor_only_rating(self) -> None:
        rating, is_hotspot, is_core = MetricsCalculator.rate(1000.0, None)
        assert rating == RatingLevel.MONITOR_ONLY
        assert not is_hotspot
        assert not is_core

    def test_rating_icons(self) -> None:
        assert MetricsCalculator.format_rating_icon(RatingLevel.EXCELLENT) == "✅"
        assert MetricsCalculator.format_rating_icon(RatingLevel.GOOD) == "👍"
        assert MetricsCalculator.format_rating_icon(RatingLevel.ACCEPTABLE) == "⚠️"
        assert MetricsCalculator.format_rating_icon(RatingLevel.NEEDS_IMPROVEMENT) == "❌"
        assert MetricsCalculator.format_rating_icon(RatingLevel.MONITOR_ONLY) == ""


class TestLatencyTracker:
    """测试 LatencyTracker"""

    def test_event_recording(self) -> None:
        tracker = LatencyTracker()
        tracker.reset("test_turn")
        tracker.mark("vad_speech_end")
        tracker.mark("asr_ws_connect_start")
        tracker.mark("asr_ws_connected")

        assert tracker.has_event("vad_speech_end")
        assert tracker.has_event("asr_ws_connect_start")
        assert len(tracker.events) == 3

    def test_duration_calculation(self) -> None:
        tracker = LatencyTracker()
        tracker.reset("test_turn")
        tracker.mark("start_event")

        time.sleep(0.1)  # 100ms delay

        tracker.mark("end_event")

        duration = tracker.get_duration("start_event", "end_event")
        assert duration is not None
        assert duration >= 100.0

    def test_event_aliases(self) -> None:
        tracker = LatencyTracker()
        tracker.reset("test_turn")

        tracker.mark("vad_speech_end")
        assert tracker.has_event("speech_end")

        tracker.mark("asr_ws_reused")
        assert tracker.has_event("asr_reuse")

    def test_reuse_detection(self) -> None:
        tracker = LatencyTracker()
        tracker.reset("test_turn")

        tracker.mark("asr_ws_reused")
        asr_metrics = tracker.calculate_l2_asr_metrics()

        b1_metric = next((m for m in asr_metrics if m.code == "B1"), None)
        assert b1_metric is not None
        assert b1_metric.is_reuse
        assert b1_metric.value_ms == 0.0

    def test_l1_ttfb_calculation(self) -> None:
        tracker = LatencyTracker()
        tracker.reset("test_turn")

        tracker.mark("vad_speech_end")
        tracker.mark("audio_playback_started")

        l1_metrics = tracker.calculate_l1_metrics()
        ttfb_metric = next((m for m in l1_metrics if m.code == "L2"), None)
        assert ttfb_metric is not None
        assert ttfb_metric.name == "首音延迟 (TTFB)"

    def test_l1_transcript_delay_calculation(self) -> None:
        tracker = LatencyTracker()
        tracker.reset("test_turn")

        tracker.mark("vad_speech_end")
        tracker.mark("transcript_show", {"transcript_len": 20})

        l1_metrics = tracker.calculate_l1_metrics()
        transcript_metric = next((m for m in l1_metrics if m.code == "L3"), None)
        assert transcript_metric is not None
        assert transcript_metric.name == "转录延迟"


class TestL2MetricsCalculation:
    """测试 L2 指标计算"""

    def test_asr_metrics_full_flow(self) -> None:
        """ASR 完整流程：B1, B2, B3, B4（B5 已删除）"""
        tracker = LatencyTracker()
        tracker.reset("test_turn")

        tracker.mark("asr_ws_connect_start")
        tracker.mark("asr_ws_connected")
        tracker.mark("asr_session_update_sent")
        tracker.mark("asr_audio_send_start")
        tracker.mark("asr_audio_send_complete")
        tracker.mark("asr_commit_sent")
        tracker.mark("asr_final_received", {"text_len": 30})
        tracker.mark("asr_result_delivered", {"transcript_len": 30})

        asr_metrics = tracker.calculate_l2_asr_metrics()

        codes = [m.code for m in asr_metrics]
        assert "B1" in codes
        assert "B2" in codes
        assert "B3" in codes
        assert "B4" in codes
        assert "B5" not in codes  # B5 已删除

    def test_asr_metrics_reuse_flow(self) -> None:
        tracker = LatencyTracker()
        tracker.reset("test_turn")

        tracker.mark("asr_ws_reused")
        tracker.mark("asr_commit_sent")
        tracker.mark("asr_final_received")

        asr_metrics = tracker.calculate_l2_asr_metrics()

        b1_metric = next((m for m in asr_metrics if m.code == "B1"), None)
        assert b1_metric is not None
        assert b1_metric.is_reuse
        assert b1_metric.value_ms == 0.0

    def test_gap_metrics_g1_calculation(self) -> None:
        """G1 正常计算"""
        tracker = LatencyTracker()
        tracker.reset("test_turn")

        tracker.mark("asr_result_delivered")
        time.sleep(0.01)
        tracker.mark("llm_start")

        gap_metrics = tracker.calculate_l2_gap_metrics()
        codes = [m.code for m in gap_metrics]
        assert "G1" in codes

    def test_gap_metrics_g2_monitor_only(self) -> None:
        """G2 为仅监控（流式模式下可能为负值）"""
        tracker = LatencyTracker()
        tracker.reset("test_turn")

        tracker.mark("llm_complete")
        tracker.mark("tts_start")

        gap_metrics = tracker.calculate_l2_gap_metrics()
        g2_metric = next((m for m in gap_metrics if m.code == "G2"), None)
        assert g2_metric is not None
        assert g2_metric.is_monitor_only
        assert g2_metric.rating == RatingLevel.MONITOR_ONLY

    def test_gap_metrics_g2_negative_in_streaming(self) -> None:
        """流式模式下 G2 为负值（tts_start 在 llm_complete 之前）"""
        tracker = LatencyTracker()
        tracker.reset("test_turn")

        tracker.mark("tts_start")
        tracker.mark("llm_complete")

        gap_metrics = tracker.calculate_l2_gap_metrics()
        g2_metric = next((m for m in gap_metrics if m.code == "G2"), None)
        assert g2_metric is not None
        assert g2_metric.value_ms < 0  # 负值正常
        assert g2_metric.is_monitor_only  # 仅监控，不计入公式

    def test_llm_metrics_c2_critical_path(self) -> None:
        """C2 关键路径：llm_stream_opened → tts_start"""
        tracker = LatencyTracker()
        tracker.reset("test_turn")

        tracker.mark("llm_request_sending")
        tracker.mark("llm_stream_opened")
        tracker.mark("tts_start")
        tracker.mark("llm_complete")

        llm_metrics = tracker.calculate_l2_llm_metrics()

        c2_metric = next((m for m in llm_metrics if m.code == "C2"), None)
        assert c2_metric is not None
        # C2 应该是 llm_stream_opened → tts_start
        assert c2_metric.value_ms >= 0

    def test_llm_metrics_c3_parallel(self) -> None:
        """C3 并行监控：tts_start → llm_complete"""
        tracker = LatencyTracker()
        tracker.reset("test_turn")

        tracker.mark("llm_request_sending")
        tracker.mark("llm_stream_opened")
        tracker.mark("tts_start")
        tracker.mark("llm_complete")

        llm_metrics = tracker.calculate_l2_llm_metrics()

        c3_metric = next((m for m in llm_metrics if m.code == "C3"), None)
        assert c3_metric is not None
        assert c3_metric.is_monitor_only
        assert c3_metric.is_parallel

    def test_llm_metrics_c2_fallback_no_tts_start(self) -> None:
        """批处理模式下无 tts_start，C2 回退到 llm_complete"""
        tracker = LatencyTracker()
        tracker.reset("test_turn")

        tracker.mark("llm_request_sending")
        tracker.mark("llm_stream_opened")
        tracker.mark("llm_complete")

        llm_metrics = tracker.calculate_l2_llm_metrics()

        c2_metric = next((m for m in llm_metrics if m.code == "C2"), None)
        assert c2_metric is not None
        # 无 tts_start 时回退到 llm_complete
        assert c2_metric.value_ms >= 0

    def test_tts_metrics_cloud_calculation(self) -> None:
        tracker = LatencyTracker()
        tracker.reset("test_turn")

        tracker.mark("tts_start")
        tracker.mark("tts_ws_connect_start")
        tracker.mark("tts_ws_connected")
        tracker.mark("tts_session_update_sent")
        tracker.mark("tts_commit_sent")
        tracker.mark("tts_first_chunk_ready")
        tracker.mark("tts_finish_event_received")

        tts_metrics = tracker.calculate_l2_tts_metrics()

        codes = [m.code for m in tts_metrics]
        assert "D1" in codes
        assert "D2" in codes
        assert "D3" in codes
        assert "D4" in codes

    def test_tts_metrics_local_piper(self) -> None:
        """本地 TTS (Piper) 指标：D1=0, D2, D3, D4"""
        tracker = LatencyTracker()
        tracker.reset("test_turn")

        # 本地 TTS 事件
        tracker.mark("tts_start")
        tracker.mark("tts_model_generation_start")
        tracker.mark("tts_first_chunk_ready")
        tracker.mark("tts_model_generation_complete")

        assert tracker._is_local_tts()

        tts_metrics = tracker.calculate_l2_tts_metrics()

        codes = [m.code for m in tts_metrics]
        assert "D1" in codes
        assert "D2" in codes
        assert "D3" in codes
        assert "D4" in codes

        # D1 应为 0ms (本地)
        d1 = next((m for m in tts_metrics if m.code == "D1"), None)
        assert d1 is not None
        assert d1.value_ms == 0.0
        assert d1.is_reuse


class TestHotspotIdentification:
    """测试热点识别"""

    def test_core_hotspot_b4_detection(self) -> None:
        tracker = LatencyTracker()
        tracker.reset("test_turn")

        tracker.mark("asr_commit_sent")
        time.sleep(0.35)
        tracker.mark("asr_result_delivered")

        hotspots = tracker.identify_hotspots()

        b4_hotspot = next((h for h in hotspots if h[0] == "B4"), None)
        assert b4_hotspot is not None
        assert b4_hotspot[2] == "ASR云端处理延迟"

    def test_hidden_hotspot_g1_detection(self) -> None:
        tracker = LatencyTracker()
        tracker.reset("test_turn")

        tracker.mark("asr_final_received")
        time.sleep(0.06)
        tracker.mark("llm_start")

        hotspots = tracker.identify_hotspots()

        g1_hotspot = next((h for h in hotspots if h[0] == "G1"), None)
        assert g1_hotspot is not None


class TestTraceFormulaValidation:
    """测试指标追溯公式验证（关键路径法）"""

    def test_trace_formula_critical_path_cloud(self) -> None:
        """云端 TTS 关键路径公式：TTFB = B4 + G1 + C1 + C2 + D2 + D3"""
        tracker = LatencyTracker()
        tracker.reset("test_turn")

        tracker.mark("vad_speech_end")
        tracker.mark("asr_commit_sent")
        tracker.mark("asr_result_delivered")
        tracker.mark("llm_start")
        tracker.mark("llm_request_sending")
        tracker.mark("llm_stream_opened")
        tracker.mark("llm_first_token")
        tracker.mark("tts_start")  # C2 截断点
        tracker.mark("llm_complete")
        tracker.mark("tts_ws_connected")
        tracker.mark("tts_session_update_sent")
        tracker.mark("tts_commit_sent")
        tracker.mark("tts_first_chunk_ready")
        tracker.mark("audio_playback_started")

        trace = tracker.validate_trace_formula()

        assert trace.get("valid")
        components = trace.get("components", {})

        # 关键路径组件
        assert "B4" in components
        assert "G1" in components
        assert "C1" in components
        assert "C2" in components
        assert "D2" in components
        assert "D3" in components

        # 不在关键路径中
        assert "B5" not in components  # 已删除
        assert "G2" not in components  # 仅监控
        assert "C3" not in components  # 并行监控
        assert "D1" not in components  # 不在关键路径公式中

    def test_trace_formula_critical_path_local_tts(self) -> None:
        """本地 TTS 关键路径公式"""
        tracker = LatencyTracker()
        tracker.reset("test_turn")

        tracker.mark("vad_speech_end")
        tracker.mark("asr_commit_sent")
        tracker.mark("asr_result_delivered")
        tracker.mark("llm_start")
        tracker.mark("llm_stream_opened")
        tracker.mark("tts_start")
        tracker.mark("tts_model_generation_start")
        tracker.mark("tts_first_chunk_ready")
        tracker.mark("audio_playback_started")

        trace = tracker.validate_trace_formula()

        assert trace.get("valid")
        components = trace.get("components", {})
        assert "B4" in components
        assert "C2" in components
        assert "D2" in components
        assert "D3" in components

    def test_trace_formula_with_reuse(self) -> None:
        """reuse 模式下的追溯公式"""
        tracker = LatencyTracker()
        tracker.reset("test_turn")

        tracker.mark("vad_speech_end")
        tracker.mark("asr_ws_reused")
        tracker.mark("asr_commit_sent")
        tracker.mark("asr_result_delivered")
        tracker.mark("llm_start")
        tracker.mark("llm_stream_opened")
        tracker.mark("tts_start")
        tracker.mark("tts_model_generation_start")
        tracker.mark("tts_first_chunk_ready")
        tracker.mark("audio_playback_started")

        trace = tracker.validate_trace_formula()
        assert trace.get("valid")
        assert trace.get("tts_reused")

    def test_trace_formula_deviation_reasonable(self) -> None:
        """偏差应在合理范围内（使用实际延迟）"""
        tracker = LatencyTracker()
        tracker.reset("test_turn")

        tracker.mark("vad_speech_end")
        time.sleep(0.05)  # speech_end → asr_commit
        tracker.mark("asr_commit_sent")
        time.sleep(0.05)  # B4
        tracker.mark("asr_result_delivered")
        time.sleep(0.01)  # G1
        tracker.mark("llm_start")
        time.sleep(0.01)  # C1
        tracker.mark("llm_stream_opened")
        time.sleep(0.05)  # C2
        tracker.mark("tts_start")
        time.sleep(0.01)  # D2
        tracker.mark("tts_model_generation_start")
        time.sleep(0.05)  # D3
        tracker.mark("tts_first_chunk_ready")
        tracker.mark("audio_playback_started")

        trace = tracker.validate_trace_formula()

        assert trace.get("valid")
        # 本地 TTS: B4 + G1 + C1 + C2 + D2 + D3 ≈ actual_ttfb
        if trace.get("actual_ttfb_ms", 0) > 10:
            assert abs(trace.get("deviation_pct", 100)) < 25


class TestReportOutput:
    """测试报告输出"""

    def test_print_summary_format(self, caplog: pytest.LogCaptureFixture) -> None:
        tracker = LatencyTracker()
        tracker.reset("test_turn")

        tracker.mark("vad_speech_end")
        tracker.mark("asr_ws_reused")
        tracker.mark("audio_playback_started")

        with caplog.at_level(logging.INFO):
            tracker.print_summary()

        log_messages = [r.message for r in caplog.records]
        assert any("一级指标" in msg for msg in log_messages)
        assert any("=" * 80 in msg for msg in log_messages)

    def test_reuse_marker_in_report(self) -> None:
        tracker = LatencyTracker()
        tracker.reset("test_turn")

        tracker.mark("asr_ws_reused")
        asr_metrics = tracker.calculate_l2_asr_metrics()
        b1_metric = next((m for m in asr_metrics if m.code == "B1"), None)

        assert b1_metric is not None
        assert b1_metric.is_reuse
        assert b1_metric.rating == RatingLevel.EXCELLENT

    def test_b5_not_in_report(self) -> None:
        """B5 不应出现在任何指标计算中"""
        tracker = LatencyTracker()
        tracker.reset("test_turn")

        tracker.mark("asr_commit_sent")
        tracker.mark("asr_final_received")
        tracker.mark("asr_result_delivered")

        asr_metrics = tracker.calculate_l2_asr_metrics()
        codes = [m.code for m in asr_metrics]
        assert "B5" not in codes


class TestLocalASRMetrics:
    """Tests for local ASR (sherpa-onnx Zipformer) timing metrics."""

    def test_b1_local_asr_shows_reuse(self):
        """Local ASR B1 should show 0ms reuse when asr_local_ready event exists."""
        tracker = LatencyTracker()
        tracker.reset("test")
        t0 = tracker.start_time
        tracker.events = [
            {"name": "vad_speech_end", "canonical": "speech_end", "timestamp": t0 + 0.1, "elapsed_ms": 100, "metadata": {}},
            {"name": "asr_local_ready", "canonical": "asr_reuse", "timestamp": t0 + 0.1, "elapsed_ms": 100, "metadata": {}},
            {"name": "asr_local_stream_start", "canonical": "asr_b2_end", "timestamp": t0 + 0.1001, "elapsed_ms": 100.1, "metadata": {}},
        ]
        results = tracker.calculate_l2_asr_metrics()
        b1 = [m for m in results if m.code == "B1"]
        assert len(b1) == 1
        assert b1[0].is_reuse is True
        assert b1[0].value_ms == 0.0

    def test_b4_local_asr_uses_local_path(self):
        """Local ASR B4 should use asr_local_final_decode -> asr_result_delivered."""
        tracker = LatencyTracker()
        tracker.reset("test")
        t0 = tracker.start_time
        tracker.events = [
            {"name": "asr_local_final_decode", "canonical": "asr_b4_start", "timestamp": t0 + 0.1, "elapsed_ms": 100, "metadata": {}},
            {"name": "asr_result_delivered", "canonical": "asr_b5_end", "timestamp": t0 + 0.135, "elapsed_ms": 135, "metadata": {"transcript_len": 4}},
        ]
        results = tracker.calculate_l2_asr_metrics()
        b4 = [m for m in results if m.code == "B4"]
        assert len(b4) == 1
        assert b4[0].value_ms == pytest.approx(35.0, abs=1.0)

    def test_b4_cloud_asr_unchanged(self):
        """Cloud ASR B4 path should remain unchanged when no local events."""
        tracker = LatencyTracker()
        tracker.reset("test")
        t0 = tracker.start_time
        tracker.events = [
            {"name": "asr_commit_sent", "canonical": "asr_b4_start", "timestamp": t0 + 0.1, "elapsed_ms": 100, "metadata": {}},
            {"name": "asr_result_delivered", "canonical": "asr_b5_end", "timestamp": t0 + 0.4, "elapsed_ms": 400, "metadata": {}},
        ]
        results = tracker.calculate_l2_asr_metrics()
        b4 = [m for m in results if m.code == "B4"]
        assert len(b4) == 1
        assert b4[0].value_ms == pytest.approx(300.0, abs=1.0)


class TestBargeInTimingRegression:
    """Regression tests for overlapped barge-in timing reports."""

    def test_l1_uses_current_turn_input_boundary_when_next_vad_events_arrive(self):
        """A later utterance's vad_speech_end must not make the current report negative."""
        tracker = LatencyTracker()
        tracker.reset("test")
        t0 = tracker.start_time

        for name, offset in [
            ("vad_speech_end", 0.0),
            ("asr_local_ready", 0.001),
            ("asr_local_final_decode", 0.003),
            ("asr_result_delivered", 0.005),
            ("llm_start", 0.006),
            ("llm_stream_opened", 0.380),
            ("llm_first_token", 0.384),
            ("llm_first_speech_chunk", 0.385),
            ("tts_start", 0.974),
            ("tts_model_generation_start", 0.975),
            ("tts_first_chunk_ready", 1.635),
            ("audio_playback_started", 1.636),
            # Next utterance starts before this cancelled turn prints its report.
            ("vad_speech_start", 1.900),
            ("asr_local_ready", 1.902),
            ("vad_speech_end", 2.211),
        ]:
            tracker.inject_event(name, t0 + offset)

        l1_metrics = {m.code: m for m in tracker.calculate_l1_metrics()}

        assert l1_metrics["L1"].value_ms == pytest.approx(385.0, abs=1.0)
        assert l1_metrics["L2"].value_ms == pytest.approx(1636.0, abs=1.0)
        assert l1_metrics["L1"].value_ms > 0
        assert l1_metrics["L2"].value_ms > 0

    def test_trace_formula_uses_current_turn_input_boundary_when_next_vad_events_arrive(self):
        """Trace validation must stay on the current turn's first input/output chain."""
        tracker = LatencyTracker()
        tracker.reset("test")
        t0 = tracker.start_time

        for name, offset in [
            ("vad_speech_end", 0.0),
            ("asr_local_ready", 0.001),
            ("asr_local_final_decode", 0.003),
            ("asr_result_delivered", 0.005),
            ("llm_start", 0.006),
            ("llm_stream_opened", 0.380),
            ("llm_first_token", 0.384),
            ("tts_start", 0.974),
            ("tts_model_generation_start", 0.975),
            ("tts_first_chunk_ready", 1.635),
            ("audio_playback_started", 1.636),
            ("vad_speech_start", 1.900),
            ("asr_local_ready", 1.902),
            ("vad_speech_end", 2.211),
        ]:
            tracker.inject_event(name, t0 + offset)

        trace = tracker.validate_trace_formula()

        assert trace["valid"]
        assert trace["actual_ttfb_ms"] == pytest.approx(1636.0, abs=1.0)
        assert trace["components"]["G1"] == pytest.approx(6.0, abs=1.0)
        assert abs(trace["deviation_ms"]) < 5.0
