"""
batch_synthesize.py - 批量 TTS 合成脚本

读取多个 txt 文件，将每一行文本转换为语音并保存为 WAV 文件。

用法:
    cd eval_custom && uv run python batch_synthesize.py
"""

import os
import sys
import shutil
from pathlib import Path

# ============================================
# 配置区 —— 按需修改
# ============================================

# txt 文件列表（相对于本脚本所在目录 eval_custom/）
TXT_FILES = [
    "text_files/1.txt",
]

# 起始编号（6 位补零，如 START=1 → 000001.wav）
START = 1

# 输出目录名（相对于脚本运行时的 cwd，即 eval_custom/）
OUTPUT_DIR = "output"

# .env 与 models.toml 路径（相对于 cwd）
ENV_PATH = ".env"
TOML_PATH = "models.toml"

# ============================================

SCRIPT_DIR = Path(__file__).resolve().parent


def prepare_output_dir(output_path: Path):
    """创建或清空 output 目录"""
    if output_path.exists():
        print(f"[清理] 删除已有目录: {output_path}")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"[就绪] 输出目录: {output_path}")


def main():
    # 切到脚本所在目录（eval_custom/）
    os.chdir(SCRIPT_DIR)
    print(f"[工作目录] {Path.cwd()}")

    # 1. 准备 output 目录
    output_path = Path(OUTPUT_DIR)
    prepare_output_dir(output_path)

    # 2. 收集所有文本行
    all_lines = []
    for rel in TXT_FILES:
        txt_path = Path(rel)
        if not txt_path.exists():
            print(f"[错误] 文件不存在: {txt_path}")
            sys.exit(1)
        print(f"[读取] {txt_path}")
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    all_lines.append(stripped)

    total = len(all_lines)
    if total == 0:
        print("[错误] 所有文件内容为空，无有效文本行")
        sys.exit(1)

    print(f"[统计] 共 {len(TXT_FILES)} 个文件，{total} 行有效文本")

    # 3. 初始化 TTS
    print("[加载] 正在初始化 TTS 模型...")
    try:
        from streaming_tts import create_tts_from_env
    except ImportError:
        print("[错误] 无法导入 streaming_tts，请确保 streaming_tts.py 在同目录")
        sys.exit(1)

    try:
        tts = create_tts_from_env(env_path=ENV_PATH, toml_path=TOML_PATH)
    except Exception as e:
        print(f"[错误] TTS 初始化失败: {e}")
        sys.exit(1)

    # 4. 逐行合成
    success = 0
    for idx, text in enumerate(all_lines):
        n = START + idx
        wav_name = f"{n:06d}.wav"
        wav_path = output_path / wav_name

        print(f"[{idx + 1}/{total}] → {wav_name}  \"{text}\"")
        try:
            sr, audio, elapsed = tts.generate(text)
            if len(audio) == 0:
                print(f"  [错误] 合成结果为空！")
                sys.exit(1)
            tts._save_wav(audio, sr, str(wav_path))
            dur = len(audio) / sr if sr else 0
            print(f"  [OK] 耗时={elapsed:.2f}s 音频={dur:.2f}s {sr}Hz")
            success += 1
        except Exception as e:
            print(f"  [失败] {e}")
            sys.exit(1)

    print(f"[完成] 成功 {success}/{total}，输出目录: {output_path.resolve()}")


if __name__ == "__main__":
    main()
