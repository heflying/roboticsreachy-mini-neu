"""命令行参数与评测配置。"""

import argparse
from pathlib import Path
from typing import Optional


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ASR 评测管线 — VAD 切句 → ASR 推理 → jiwer 评分",
    )

    parser.add_argument(
        "--model-type",
        default="sherpa_onnx_zipformer",
        choices=["sherpa_onnx_zipformer", "sherpa_onnx_paraformer"],
        help="ASR 后端类型: sherpa_onnx_zipformer | sherpa_onnx_paraformer",
    )

    # Sherpa-ONNX 配置
    parser.add_argument(
        "--sherpa-model-id",
        default="csukuangfj/sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30",
        help="Sherpa-ONNX HuggingFace model repo ID",
    )
    parser.add_argument(
        "--sherpa-model-dir",
        default="models/ASR/zipformer-zh",
        help="Sherpa-ONNX 本地模型目录",
    )
    parser.add_argument(
        "--sherpa-num-threads", type=int, default=1, help="Sherpa-ONNX 线程数"
    )

    # 数据集 (JSONL Manifest)
    # nargs="+" 支持: --manifest a.jsonl b.jsonl
    parser.add_argument(
        "--manifest",
        nargs="+",
        required=True,
        help="JSONL manifest 文件列表，如: manifests/aishell1_test.jsonl manifests/seniortalk_test.jsonl",
    )

    # 数据量限制
    parser.add_argument("--max-hours", type=float, help="每数据集最大测试时长（小时）")
    parser.add_argument(
        "--max-utterances", type=int, help="每数据集最大测试句数"
    )

    # VAD 参数
    parser.add_argument(
        "--vad-threshold", type=float, default=0.5, help="VAD 语音概率阈值"
    )
    parser.add_argument(
        "--vad-min-speech-ms", type=int, default=250, help="VAD 最小语音长度(ms)"
    )
    parser.add_argument(
        "--vad-min-silence-ms",
        type=int,
        default=500,
        help="VAD 最小静音长度(ms)，用于断句",
    )

    # 输出
    parser.add_argument(
        "--output-dir", default="results", help="结果 JSON 输出目录"
    )

    # CER 计算
    parser.add_argument(
        "--keep-punctuation",
        action="store_true",
        help="CER 计算时保留标点符号（默认去除标点）",
    )

    # 其他
    parser.add_argument("--verbose", "-v", action="store_true", help="详细日志")
    parser.add_argument("--dry-run", action="store_true", help="仅统计，不实际推理")

    args = parser.parse_args(argv)
    return args


def get_manifest_list(manifest_args: list[str]) -> list[str]:
    """返回去重后的 manifest 文件路径列表。"""
    seen: set[str] = set()
    result: list[str] = []
    for p in manifest_args:
        p = p.strip()
        if p and p not in seen:
            seen.add(p)
            result.append(p)
    return result
