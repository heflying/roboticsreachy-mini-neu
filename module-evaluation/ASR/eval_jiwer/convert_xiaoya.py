"""
临时脚本：将 dataset/ASR_xiao-ya_gen 转换为 eval_jiwer/manifests 格式的 JSONL。

用法：
    cd eval_jiwer
    python convert_xiaoya.py

输出：manifests/xiaoya_gen.jsonl
"""

import json
import sys
from pathlib import Path

import soundfile as sf

# 项目根目录（脚本放在 eval_jiwer/ 下，上级即为根目录）
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "ASR_xiao-ya_gen"
OUTPUT_PATH = SCRIPT_DIR / "manifests" / "xiaoya_gen.jsonl"


def main():
    if not DATASET_DIR.exists():
        print(f"[ERROR] 数据集目录不存在: {DATASET_DIR}")
        sys.exit(1)

    # 读取标注文件
    txt_path = DATASET_DIR / "1.txt"
    if not txt_path.exists():
        print(f"[ERROR] 标注文件不存在: {txt_path}")
        sys.exit(1)

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    print(f"共读取 {len(lines)} 行标注")

    results = []
    skipped = 0

    for idx, text in enumerate(lines, start=1):
        # 跳过空行
        if not text:
            print(f"  跳过第 {idx} 行（空行）")
            skipped += 1
            continue

        # 构造音频文件名：000001.wav 格式
        audio_name = f"{idx:06d}.wav"
        audio_path = DATASET_DIR / audio_name

        if not audio_path.exists():
            print(f"  [WARN] 音频文件不存在: {audio_path}")
            skipped += 1
            continue

        # 获取音频时长
        try:
            info = sf.info(str(audio_path))
            duration = round(info.duration, 3)
        except Exception as e:
            print(f"  [WARN] 无法读取 {audio_name} 时长: {e}")
            duration = 0.0

        # manifest 中的路径是相对于 manifests/ 目录的
        rel_path = Path("..") / ".." / "dataset" / "ASR_xiao-ya_gen" / audio_name

        results.append({
            "audio_path": str(rel_path).replace("\\", "/"),
            "text": text,
            "duration": duration,
        })

    # 写入 JSONL
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n完成！共写入 {len(results)} 条，跳过 {skipped} 条")
    print(f"输出文件: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
