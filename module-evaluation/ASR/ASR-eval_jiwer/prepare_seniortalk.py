"""SeniorTalk 数据集 → JSONL Manifest 转换脚本。

用法:
    cd eval_jiwer
    python prepare_seniortalk.py \
        --list ../dataset/SeniorTalk/test_data.list \
        --wav-root ../dataset/SeniorTalk/test \
        --output manifests/seniortalk_test.jsonl

输入: test_data.list (JSONL 格式，key/wav/txt)
输出: JSONL manifest，每行 {"audio_path": "...", "text": "...", "duration": 12.34}

去重: 按 key 去重，保留首次出现的记录。
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import soundfile as sf

logger = logging.getLogger(__name__)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="SeniorTalk → JSONL manifest 转换"
    )
    parser.add_argument(
        "--list",
        required=True,
        help="路径: test_data.list (JSONL格式，含 key/wav/txt)",
    )
    parser.add_argument(
        "--wav-root",
        required=True,
        help="WAV 文件根目录（用于计算 duration 和路径映射）",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="输出 JSONL manifest 路径",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="详细日志"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    list_path = Path(args.list).resolve()
    wav_root = Path(args.wav_root).resolve()
    output_path = Path(args.output).resolve()

    if not list_path.exists():
        logger.error(f"List file not found: {list_path}")
        return 1
    if not wav_root.is_dir():
        logger.error(f"WAV root not found: {wav_root}")
        return 1

    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 读入原始 JSONL
    records: list[dict] = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"[{list_path}:{line_no}] Invalid JSON: {e}")

    logger.info(f"Read {len(records)} records from {list_path}")

    # 去重: 按 key
    seen_keys: set[str] = set()
    deduped: list[dict] = []
    for rec in records:
        key = rec.get("key", "")
        if key and key not in seen_keys:
            seen_keys.add(key)
            deduped.append(rec)

    if len(deduped) < len(records):
        logger.info(f"Dedup: {len(records)} → {len(deduped)} ({len(records) - len(deduped)} removed)")

    # 转换 + 计算 duration
    skipped = 0
    output_dir = output_path.parent.resolve()

    with open(output_path, "w", encoding="utf-8") as out:
        for rec in deduped:
            key = rec.get("key", "")
            text = rec.get("txt", "")
            raw_wav = rec.get("wav", "")

            if not key or not text or not raw_wav:
                logger.debug(f"Skipping record (missing fields): key={key}")
                skipped += 1
                continue

            # 路径映射: 原始绝对路径 → 相对 wav_root
            # 原始路径如 /home/chenyang/.../test/S0046/Elderly...W0001.wav
            # 目标相对路径: S0046/Elderly...W0001.wav
            raw_path = Path(raw_wav)
            # 尝试从原始路径中提取 speaker/wav 部分（不管前缀是什么）
            parts = raw_path.parts
            # 找 "test" 之后的部分
            rel_parts: list[str] = []
            found_test = False
            for p in parts:
                if found_test:
                    rel_parts.append(p)
                if p == "test":
                    found_test = True
            if not rel_parts:
                # 回退：相对于 wav_root 查找
                wav_name = raw_path.name
                speaker = raw_path.parent.name
                rel_parts = [speaker, wav_name]

            rel_wav_path = Path(*rel_parts)

            # 计算相对 manifest 的路径
            # manifest 在 eval_jiwer/manifests/seniortalk_test.jsonl
            # WAV 在 ../../dataset/SeniorTalk/test/SXXXX/xxx.wav
            # 先构造 WAV 相对 output_dir 的路径
            full_wav = wav_root / rel_wav_path
            # Python 3.11 没有 PurePath.relative_to(walk_up=)，用 os.path.relpath
            rel_str = os.path.relpath(str(full_wav.resolve()), str(output_dir))
            rel_to_manifest = rel_str.replace("\\", "/")

            # 计算 duration
            duration = 0.0
            if full_wav.exists():
                try:
                    info = sf.info(str(full_wav))
                    duration = round(info.duration, 4)
                except Exception as e:
                    logger.warning(f"Failed to read duration for {full_wav}: {e}")
            else:
                logger.warning(f"WAV not found: {full_wav}")

            # 写入 JSONL
            manifest_line = {
                "audio_path": rel_to_manifest,
                "text": text,
                "duration": duration,
            }
            out.write(json.dumps(manifest_line, ensure_ascii=False) + "\n")

    logger.info(f"Output: {output_path}")
    logger.info(f"  Written: {len(deduped) - skipped} records")
    logger.info(f"  Skipped: {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
