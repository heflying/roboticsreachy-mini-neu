"""Download Zipformer Chinese streaming ASR model for offline use.

Downloads sherpa-onnx-streaming-zipformer-zh-int8 from HuggingFace
to models/zipformer-zh/ (~160MB).

Usage:
    python scripts/download_zipformer_zh.py                  # 默认源
    set HF_ENDPOINT=https://hf-mirror.com && python ...      # 镜像源
"""

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "csukuangfj/sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30"
MODEL_DIR = Path("models/zipformer-zh")

# 中国大陆镜像：set HF_ENDPOINT=https://hf-mirror.com
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "")


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if HF_ENDPOINT:
        print(f"Using mirror: {HF_ENDPOINT}")

    print(f"Downloading from {REPO_ID} -> {MODEL_DIR}/\n")

    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            snapshot_download(
                repo_id=REPO_ID,
                local_dir=str(MODEL_DIR),
                resume_download=True,
                max_workers=1,
            )
            break
        except Exception as e:
            print(f"\nAttempt {attempt}/{max_retries} failed: {e}", file=sys.stderr)
            if attempt < max_retries:
                print("Retrying (partial progress preserved)...\n", file=sys.stderr)
            else:
                if not HF_ENDPOINT:
                    print(
                        "Tip: try China mirror:\n"
                        "  PowerShell: $env:HF_ENDPOINT='https://hf-mirror.com'; python scripts/download_zipformer_zh.py\n"
                        "  CMD:        set HF_ENDPOINT=https://hf-mirror.com && python scripts/download_zipformer_zh.py",
                        file=sys.stderr,
                    )
                sys.exit(1)

    print(f"\nFiles in {MODEL_DIR}/:")
    for p in sorted(MODEL_DIR.rglob("*")):
        if p.is_file():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  {p.relative_to(MODEL_DIR)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
