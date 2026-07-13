"""简单 ASR 测试脚本。

用法:
    python asr_test.py <audio_file> [--model zipformer|paraformer]

示例:
    python asr_test.py test.wav
    python asr_test.py test.wav --model paraformer
    python asr_test.py test.wav --model zipformer --num-threads 4
"""

import argparse
import logging
import time
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from asr.sherpa_onnx_zipformer import SherpaOnnxZipformerASR
from asr.sherpa_onnx_paraformer import SherpaOnnxParaformerASR

logger = logging.getLogger(__name__)


def load_audio(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    """加载音频文件，转为 16kHz 单声道 float32。使用 librosa 重采样，与 roboticsreachy 工程保持一致。"""
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # 立体声 → 单声道
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32)
        logger.info(f"Resampled {sr} → {target_sr} Hz (librosa)")
    return audio


def transcribe_file(asr, audio_path: str, chunk_ms: int = 200) -> tuple[str, float]:
    """对单个音频文件做流式 ASR 推理。

    Returns:
        (识别文本, 处理耗时秒数)
    """
    audio = load_audio(audio_path)
    total_samples = len(audio)
    chunk_samples = int(chunk_ms / 1000 * 16000)

    logger.info(
        f"Audio: {audio_path} ({total_samples / 16000:.2f}s, "
        f"{total_samples} samples, {chunk_samples} samples/chunk)"
    )

    asr.start_utterance()
    t0 = time.perf_counter()

    for offset in range(0, total_samples, chunk_samples):
        end = min(offset + chunk_samples, total_samples)
        chunk = audio[offset:end]
        is_final = end >= total_samples
        text = asr.process_chunk(chunk, is_final=is_final)

    elapsed = time.perf_counter() - t0
    return text, elapsed


def main():
    parser = argparse.ArgumentParser(description="ASR 单文件测试")
    parser.add_argument(
        "--audio-file", default="data/test.wav",
        help="输入音频文件路径，默认 test.wav"
    )
    parser.add_argument(
        "--model", default="zipformer", choices=["zipformer", "paraformer"],
        help="ASR 模型: zipformer | paraformer"
    )
    parser.add_argument(
        "--num-threads", type=int, default=1, help="推理线程数"
    )
    parser.add_argument(
        "--chunk-ms", type=int, default=200,
        help="流式 chunk 间隔 (ms)，默认 200"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: 文件不存在: {audio_path}")
        return 1

    # 创建 ASR
    logger.info(f"初始化 {args.model} 模型...")
    if args.model == "zipformer":
        asr = SherpaOnnxZipformerASR(num_threads=args.num_threads)
    else:
        asr = SherpaOnnxParaformerASR(num_threads=args.num_threads)

    logger.info(f"模型就绪: {asr.model_info['model_name']}")
    asr.warmup()

    # 推理
    text, elapsed = transcribe_file(asr, str(audio_path), chunk_ms=args.chunk_ms)

    rtf = elapsed / (len(load_audio(str(audio_path))) / 16000) if text else 0

    logger.info("=" * 50)
    print(f"\n识别结果: {text}")
    print(f"耗时: {elapsed:.2f}s, RTF: {rtf:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
