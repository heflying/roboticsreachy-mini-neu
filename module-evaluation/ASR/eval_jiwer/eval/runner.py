"""评测运行器。

核心评测逻辑：VAD 切句 → ASR 推理 → 指标计算。
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from asr import ASRProvider
from asr.sherpa_onnx_zipformer import SherpaOnnxZipformerASR
from asr.sherpa_onnx_paraformer import SherpaOnnxParaformerASR
from vad.segmenter import SileroSegmenter, SpeechSegment
from jsonl_loader import JsonlLoader
from eval.metrics import (
    ErrorMetrics,
    AggregateMetrics,
    compute_cer,
    aggregate_metrics,
)
from eval.timing import (
    TimingMetrics,
    AggregateTiming,
    TimingTracker,
    aggregate_timing,
)
from utils import get_hardware_info, now_iso

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """评测运行器。

    将数据集声明的 Utterance 逐一送入评测管线：
    1. VAD 切句（对于非预切分数据集如 SeniorTalk）
    2. 逐句 ASR 推理 + 实时性打点
    3. 拼接全对话文本 → 与标注文本计算 CER
    4. 汇总结果写入 JSON
    """

    def __init__(
        self,
        asr: ASRProvider,
        segmenter: Optional[SileroSegmenter] = None,
        chunk_size_ms: int = 200,
        strip_punctuation: bool = True,
    ) -> None:
        self._asr = asr
        self._segmenter = segmenter or SileroSegmenter()
        self._chunk_samples = int(chunk_size_ms / 1000 * 16000)
        self._strip_punctuation = strip_punctuation

    def evaluate_dataset(
        self,
        dataset: JsonlLoader,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """评测整个数据集。

        Returns:
            包含汇总指标和逐条详情的字典
        """
        utterances = dataset.load()
        logger.info(
            f"Evaluating {dataset.name()}: {len(utterances)} samples"
        )

        all_error_metrics: list[ErrorMetrics] = []
        all_timing_metrics: list[TimingMetrics] = []
        per_utterance_details: list[dict] = []

        n_total = len(utterances)
        t_start = time.perf_counter()

        for i, utt in enumerate(utterances):
            idx = i + 1
            pct = idx / n_total * 100

            if dry_run:
                continue

            try:
                result = self._evaluate_one(utt)
                per_utterance_details.append(result)

                # 收集错误指标（拼接后全对话级别）
                if "error" not in result:
                    em = result.get("error_metrics", {})
                    if em:
                        all_error_metrics.append(
                            ErrorMetrics(
                                cer=em.get("cer", 0),
                                substitutions=em.get("substitutions", 0),
                                deletions=em.get("deletions", 0),
                                insertions=em.get("insertions", 0),
                                hits=em.get("hits", 0),
                                reference_length=em.get("reference_length", 0),
                            )
                        )

                    # 收集每句的实时性指标
                    per_seg_timings = result.get("per_segment_timing", [])
                    for t in per_seg_timings:
                        all_timing_metrics.append(
                            TimingMetrics(
                                audio_duration_s=t.get("audio_duration_s", 0),
                                processing_time_s=t.get("processing_time_s", 0),
                                rtf=t.get("rtf", 0),
                                ttfc_s=t.get("ttfc_s", 0),
                                convergence_latency_s=t.get("convergence_latency_s", 0),
                                e2e_latency_s=t.get("e2e_latency_s", 0),
                                num_chunks=t.get("num_chunks", 0),
                            )
                        )
            except Exception as e:
                logger.error(f"Failed to evaluate {utt.audio_path}: {e}")
                per_utterance_details.append(
                    {"audio_path": utt.audio_path, "error": str(e)}
                )

            # 进度输出
            elapsed = time.perf_counter() - t_start
            eta = (elapsed / idx) * (n_total - idx) if idx > 0 else 0
            print(
                f"\r[{idx}/{n_total}] {pct:5.1f}% | "
                f"elapsed {elapsed:.0f}s | eta {eta:.0f}s",
                end="",
                flush=True,
            )

        print()  # 换行
        logger.info(
            f"Completed {dataset.name()}: {n_total} samples "
            f"in {time.perf_counter() - t_start:.1f}s"
        )

        # 聚合
        agg_err = aggregate_metrics(all_error_metrics)
        agg_timing = aggregate_timing(all_timing_metrics)

        return {
            "dataset_name": dataset.name(),
            "num_utterances": len(utterances),
            "num_evaluated": len([d for d in per_utterance_details if "error" not in d]),
            "aggregate_cer": _dataclass_to_dict(agg_err),
            "aggregate_timing": _dataclass_to_dict(agg_timing),
            "per_utterance": per_utterance_details,
        }

    def _evaluate_one(self, utt) -> dict[str, Any]:
        """评测单条话语（可能是一个短句，也可能是完整对话）。"""
        result = {
            "audio_path": utt.audio_path,
            "reference_text": utt.reference_text,
            "dataset_name": utt.dataset_name,
            "metadata": utt.metadata,
        }

        # VAD 切句
        segments = self._segmenter.segment(utt.audio_path)
        if not segments:
            result["error"] = "No speech detected"
            return result

        result["num_segments"] = len(segments)

        # 逐句 ASR 推理
        asr_texts: list[str] = []
        per_seg_timing: list[dict] = []

        for seg in segments:
            seg_text, seg_timing = self._transcribe_segment(seg)
            asr_texts.append(seg_text)
            per_seg_timing.append(_dataclass_to_dict(seg_timing))

        # 拼接全对话文本
        full_asr_text = "".join(asr_texts)

        # 与完整标注文本计算 CER
        error_metrics = compute_cer(
            utt.reference_text, full_asr_text,
            strip_punctuation=self._strip_punctuation,
        )

        result["asr_text"] = full_asr_text
        result["error_metrics"] = _dataclass_to_dict(error_metrics)
        result["per_segment_asr"] = asr_texts
        result["per_segment_timing"] = per_seg_timing

        return result

    def _transcribe_segment(
        self, segment: SpeechSegment
    ) -> tuple[str, TimingMetrics]:
        """对单个 VAD 片段进行 ASR 推理并打点。"""
        tracker = TimingTracker()
        audio = segment.audio
        total_samples = len(audio)

        # 分 chunk 喂入
        self._asr.start_utterance()
        t_proc_start = time.perf_counter()

        final_text = ""
        for offset in range(0, total_samples, self._chunk_samples):
            end = min(offset + self._chunk_samples, total_samples)
            chunk = audio[offset:end]
            is_final = end >= total_samples

            t_send = time.perf_counter()
            tracker.on_chunk_sent(t_send, is_first=(offset == 0))

            text = self._asr.process_chunk(chunk, is_final=is_final)

            t_recv = time.perf_counter()
            if text:
                tracker.on_text_received(t_recv, text, is_final=is_final)
                final_text = text

        t_proc_end = time.perf_counter()
        tracker.set_processing_time(t_proc_start, t_proc_end)

        timing = tracker.finalize(segment.duration)
        return final_text, timing


def create_asr(args) -> ASRProvider:
    """根据命令行参数创建 ASR 实例。"""
    if args.model_type == "sherpa_onnx_zipformer":
        return SherpaOnnxZipformerASR(
            model_id=args.sherpa_model_id,
            model_dir=args.sherpa_model_dir,
            num_threads=args.sherpa_num_threads,
        )
    elif args.model_type == "sherpa_onnx_paraformer":
        return SherpaOnnxParaformerASR(
            model_id=args.sherpa_model_id,
            model_dir=args.sherpa_model_dir,
            num_threads=args.sherpa_num_threads,
        )
    raise ValueError(f"Unknown model type: {args.model_type}")


def build_result(
    model_name: str,
    args,
    dataset_results: list[dict[str, Any]],
    hardware_info: dict,
) -> dict[str, Any]:
    """构建最终输出 JSON。"""
    return {
        "meta": {
            "tool": "eval_jiwer",
            "version": "0.1.0",
            "timestamp": now_iso(),
            "model_name": model_name,
            "model_type": args.model_type,
            "datasets": [r["dataset_name"] for r in dataset_results],
        },
        "hardware": hardware_info,
        "config": {
            "max_hours": args.max_hours,
            "max_utterances": args.max_utterances,
            "vad_threshold": args.vad_threshold,
            "vad_min_speech_ms": args.vad_min_speech_ms,
            "vad_min_silence_ms": args.vad_min_silence_ms,
        },
        "results": dataset_results,
    }


def _dataclass_to_dict(obj) -> dict:
    """将 dataclass 转为字典（递归）。"""
    if hasattr(obj, "__dataclass_fields__"):
        return {
            f: _dataclass_to_dict(getattr(obj, f))
            for f in obj.__dataclass_fields__
        }
    return obj
