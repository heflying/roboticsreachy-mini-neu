"""Omni (端到端语音) 批量测试脚本。

将音频文件送入 Omni 实时语音 API（Qwen Omni / OpenAI Realtime），
收集转录文本、回复文本、合成音频，并保存到输出目录。

输入目录的结构会镜像到输出目录。

Usage:
    cd project_root

    # 测试 input/ 目录下所有 .wav 文件
    python cascade_test/omni/run_omni.py

    # 指定输入目录和输出目录
    python cascade_test/omni/run_omni.py --input_dir cascade_test/omni/input --output_dir cascade_test/omni/output

输出 (每个 .wav 文件):
    {name}_transcript.txt   — 用户语音转录
    {name}_response.txt     — 助手回复文本
    {name}_audio.wav         — 助手合成音频 (24kHz PCM16)
    summary.json             — 汇总结果
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import shutil
import struct
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import websockets
from dotenv import load_dotenv
from scipy.signal import resample

load_dotenv(override=True)

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.prompts import get_session_instructions, get_session_voice

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Qwen Omni 协议常量
# ---------------------------------------------------------------------------
QWEN_INPUT_SAMPLE_RATE: int = 16000
QWEN_OUTPUT_SAMPLE_RATE: int = 24000
DEFAULT_QWEN_WS_URL: str = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"

QWEN_CHINESE_INSTRUCTION_SUFFIX: str = (
    "\n\n请始终使用中文和用户对话，除非用户明确要求使用其他语言。"
    "当用户要求你跳舞、转头、看向某个方向或执行机器人动作时，先简短确认并保持中文表达。"
)

# OpenAI Realtime 协议常量
OPENAI_INPUT_SAMPLE_RATE: int = 24000
OPENAI_OUTPUT_SAMPLE_RATE: int = 24000
DEFAULT_OPENAI_WS_URL: str = "wss://api.openai.com/v1/realtime"


# ---------------------------------------------------------------------------
# 音频工具
# ---------------------------------------------------------------------------


def read_wav_as_mono_int16(path: Path, target_rate: int) -> tuple[np.ndarray, int]:
    """读取 WAV 文件，重采样到 target_rate，返回单声道 int16 numpy 数组和原始采样率。"""
    with wave.open(str(path), "rb") as wf:
        orig_rate = wf.getframerate()
        nchannels = wf.getnchannels()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    dtype = np.int16
    samples = np.frombuffer(raw, dtype=dtype).reshape(-1, nchannels)
    if nchannels > 1:
        samples = samples.mean(axis=1).astype(dtype)

    if orig_rate != target_rate:
        new_len = int(len(samples) * target_rate / orig_rate)
        samples = resample(samples, new_len).astype(np.int16)

    return samples, orig_rate


def save_pcm_as_wav(path: Path, pcm_bytes: bytes, sample_rate: int) -> None:
    """将 PCM 16-bit int16 字节保存为 WAV 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


# ---------------------------------------------------------------------------
# 结果数据结构
# ---------------------------------------------------------------------------


@dataclass
class OmniTurnResult:
    """单条音频的 Omni 测试结果。"""

    file_name: str = ""
    rel_path: str = ""
    transcript: str = ""
    response: str = ""
    audio_data: bytes = field(default_factory=bytes)
    audio_duration_ms: float = 0.0
    error: str = ""


# ---------------------------------------------------------------------------
# Qwen Omni WebSocket 后端
# ---------------------------------------------------------------------------


def _build_qwen_ws_url() -> str:
    override = (config.QWEN_REALTIME_URL or "").strip()
    if override:
        return override
    query = urlencode({"model": config.MODEL_NAME})
    return f"{DEFAULT_QWEN_WS_URL}?{query}"


class QwenOmniClient:
    """精简版 Qwen Omni WebSocket 客户端，专门用于音频文件批量测试。"""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.ws: Any = None
        self._stop = asyncio.Event()

    async def __aenter__(self) -> "QwenOmniClient":
        url = _build_qwen_ws_url()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-DashScope-Beta": "realtime-v1",
        }
        self.ws = await websockets.connect(url, additional_headers=headers)  # type: ignore[assignment]
        # 发送 session.update
        instructions = get_session_instructions()
        if "始终使用中文" not in instructions:
            instructions = f"{instructions}{QWEN_CHINESE_INSTRUCTION_SUFFIX}"
        session_update = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": instructions,
                "voice": get_session_voice(),
                "input_audio_format": "pcm",
                "output_audio_format": "pcm",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.05,
                    "prefix_padding_ms": 500,
                    "silence_duration_ms": 2000,
                    "create_response": False,  # 手动控制 commit/response，避免 VAD 在 chunk 发送间隙自动截断
                    "interrupt_response": True,
                },
                "input_audio_transcription": {"model": "gummy-realtime-v1"},
            },
        }
        await self._send(session_update)
        return self

    async def __aexit__(self, *args: Any) -> None:
        self._stop.set()
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:
                pass
            self.ws = None

    async def _send(self, payload: dict[str, Any]) -> None:
        payload.setdefault("event_id", f"evt_{os.urandom(4).hex()}")
        await self.ws.send(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    async def _recv(self) -> dict[str, Any]:
        raw = await self.ws.recv()
        return json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode("utf-8"))

    async def send_audio_and_collect(self, audio_path: Path) -> OmniTurnResult:
        """发送一个音频文件并收集所有响应。"""
        result = OmniTurnResult(file_name=audio_path.name)

        # 读取并重采样音频
        try:
            samples, _orig_rate = read_wav_as_mono_int16(audio_path, QWEN_INPUT_SAMPLE_RATE)
        except Exception as e:
            result.error = f"Failed to read audio: {e}"
            return result

        # 在音频前后追加静音段：前方 500ms 避免 VAD 漏检开头，末尾 3s 确保不截断
        samples = samples.ravel()  # 确保 1D
        silence_front = np.zeros(int(QWEN_INPUT_SAMPLE_RATE * 0.5), dtype=np.int16)
        silence_tail = np.zeros(int(QWEN_INPUT_SAMPLE_RATE * 3), dtype=np.int16)
        samples = np.concatenate([silence_front, samples, silence_tail])

        # 分片发送音频 (每片约 200ms, 即 3200 samples @16kHz)
        chunk_samples = int(QWEN_INPUT_SAMPLE_RATE * 0.2)  # 200ms
        audio_bytes_all = samples.tobytes()
        offset = 0
        t_send_start = time.perf_counter()
        while offset < len(audio_bytes_all):
            chunk = audio_bytes_all[offset : offset + chunk_samples * 2]  # int16 = 2 bytes/sample
            offset += len(chunk)
            audio_b64 = base64.b64encode(chunk).decode("utf-8")
            await self._send({"type": "input_audio_buffer.append", "audio": audio_b64})

        # 提交音频并请求生成回复
        await self._send({"type": "input_audio_buffer.commit"})
        await self._send({"type": "response.create"})

        # 收集事件
        audio_chunks: list[bytes] = []
        first_audio_time: float | None = None

        try:
            while not self._stop.is_set():
                msg = await asyncio.wait_for(self._recv(), timeout=30.0)
                etype = msg.get("type", "")

                # 用户转录完成
                if etype in (
                    "conversation.item.input_audio_transcription.completed",
                    "input_audio_transcription.completed",
                ):
                    result.transcript = str(
                        msg.get("transcript") or msg.get("text") or ""
                    ).strip()

                # 助手回复文本完成
                elif etype in (
                    "response.audio_transcript.done",
                    "response.output_audio_transcript.done",
                ):
                    result.response = str(
                        msg.get("transcript") or msg.get("text") or ""
                    ).strip()

                # 音频增量
                elif etype in ("response.audio.delta", "response.output_audio.delta"):
                    delta = msg.get("delta") or msg.get("audio")
                    if isinstance(delta, str):
                        audio_bytes = base64.b64decode(delta)
                        audio_chunks.append(audio_bytes)
                        if first_audio_time is None:
                            first_audio_time = time.perf_counter()

                # 回复完成
                elif etype == "response.done":
                    break

                # 错误处理
                elif etype == "error":
                    error_info = msg.get("error", {})
                    result.error = (
                        error_info.get("message", str(error_info))
                        if isinstance(error_info, dict)
                        else str(error_info)
                    )
                    break

        except asyncio.TimeoutError:
            result.error = "Timeout waiting for response"
        except Exception as e:
            if not result.error:
                result.error = str(e)

        # 合并音频并计算时长
        if audio_chunks:
            combined = b"".join(audio_chunks)
            result.audio_data = combined
            result.audio_duration_ms = len(combined) / 2 / QWEN_OUTPUT_SAMPLE_RATE * 1000

        return result


# ---------------------------------------------------------------------------
# 批量运行
# ---------------------------------------------------------------------------


async def run_omni(
    input_dir: Path,
    output_dir: Path,
) -> list[OmniTurnResult]:
    """批量处理 input_dir 下所有 .wav 文件。"""
    # 获取 API key
    api_key = (config.DASHSCOPE_API_KEY or "").strip()
    if not api_key:
        print("ERROR: DASHSCOPE_API_KEY not found in .env")
        return []

    model = config.MODEL_NAME or "unknown"
    print(f"Omni Model: {model}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")

    # 收集音频文件 (递归)
    audio_exts = (".wav",)
    audio_files = sorted(
        p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in audio_exts
    )

    if not audio_files:
        print(f"No .wav files found in: {input_dir}")
        return []

    print(f"Files to process: {len(audio_files)}\n")

    # 清空输出目录重新开始
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    MAX_RETRIES = 10
    RETRY_DELAY = 2.0  # 秒

    results: list[OmniTurnResult] = []
    total_start = time.perf_counter()

    for i, audio_path in enumerate(audio_files):
        rel = audio_path.relative_to(input_dir)
        print(f"[{i+1}/{len(audio_files)}] {rel}  ", end="", flush=True)

        result: OmniTurnResult | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with QwenOmniClient(api_key) as client:
                    result = await client.send_audio_and_collect(audio_path)
            except Exception as e:
                result = OmniTurnResult(file_name=audio_path.name, error=str(e))

            # safety: should never be None, but guard
            if result is None:
                result = OmniTurnResult(file_name=audio_path.name, error="Unknown: result is None")

            if not result.error:
                break  # 成功，跳出重试

            # 仅服务端错误重试，客户端错误（如读文件失败）直接退出
            error_msg = result.error.lower()
            if any(kw in error_msg for kw in ("internal server error", "internal error", "service error", "timeout")):
                if attempt < MAX_RETRIES:
                    print(f"\n    Retry {attempt}/{MAX_RETRIES} (server error) ...", end="", flush=True)
                    await asyncio.sleep(RETRY_DELAY)
                    continue
            break  # 不可重试的错误，或重试次数用尽

        assert result is not None
        result.rel_path = str(rel)

        # 写输出文件 — 子目录镜像输入结构
        if not result.error:
            out_subdir = output_dir / rel.parent
            out_subdir.mkdir(parents=True, exist_ok=True)
            stem = audio_path.stem

            (out_subdir / f"{stem}_transcript.txt").write_text(
                result.transcript, encoding="utf-8"
            )
            (out_subdir / f"{stem}_response.txt").write_text(
                result.response, encoding="utf-8"
            )
            if result.audio_data:
                save_pcm_as_wav(
                    out_subdir / f"{stem}_audio.wav",
                    result.audio_data,
                    QWEN_OUTPUT_SAMPLE_RATE,
                )

        # 打印结果
        if result.error:
            print(f"FAIL: {result.error}")
        else:
            audio_kb = len(result.audio_data) / 1024
            print(
                f"transcript={len(result.transcript)}c | "
                f"response={len(result.response)}c | "
                f"audio={result.audio_duration_ms:.0f}ms ({audio_kb:.1f}KB)"
            )

        results.append(result)

    total_elapsed = time.perf_counter() - total_start
    ok = sum(1 for r in results if not r.error)
    fail = sum(1 for r in results if r.error)
    print(f"\nDone: {ok} OK, {fail} failed | Total: {total_elapsed:.1f}s | Output: {output_dir}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Omni 端到端语音批量测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python cascade_test/omni/run_omni.py
  python cascade_test/omni/run_omni.py --input_dir cascade_test/omni/input --output_dir cascade_test/omni/output
        """,
    )
    parser.add_argument(
        "--input_dir",
        default=None,
        help="输入目录（递归扫描 .wav），默认: cascade_test/omni/input/",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="输出目录，默认: cascade_test/omni/output/",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (default: WARNING)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    script_dir = Path(__file__).resolve().parent
    input_dir = Path(args.input_dir) if args.input_dir else script_dir / "input"
    output_dir = Path(args.output_dir) if args.output_dir else script_dir / "output"

    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.exit(1)

    results = asyncio.run(run_omni(input_dir=input_dir.resolve(), output_dir=output_dir.resolve()))

    if not results:
        return

    # 保存 summary.json
    summary_path = output_dir / "summary.json"
    summary = {
        "backend": config.BACKEND_PROVIDER,
        "model": config.MODEL_NAME,
        "input": str(input_dir.resolve()),
        "total_files": len(results),
        "ok": sum(1 for r in results if not r.error),
        "failed": sum(1 for r in results if r.error),
        "results": [
            {
                "file": r.file_name,
                "rel_path": r.rel_path,
                "transcript": r.transcript[:500] if r.transcript else "",
                "response": r.response[:500] if r.response else "",
                "audio_duration_ms": round(r.audio_duration_ms, 1),
                "audio_kb": round(len(r.audio_data) / 1024, 1),
                "error": r.error,
            }
            for r in results
        ],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
