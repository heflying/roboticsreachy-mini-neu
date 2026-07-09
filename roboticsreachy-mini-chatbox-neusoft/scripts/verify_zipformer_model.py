"""Verify Zipformer model files exist and are non-empty."""
from pathlib import Path

MODEL_DIR = Path("models/zipformer-zh")
FILES = ["encoder.int8.onnx", "decoder.onnx", "joiner.int8.onnx", "tokens.txt"]

all_ok = True
for f in FILES:
    p = MODEL_DIR / f
    if p.exists() and p.stat().st_size > 0:
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  {f}: OK ({size_mb:.1f} MB)")
    else:
        print(f"  {f}: MISSING")
        all_ok = False

if all_ok:
    print("\nAll model files present. Ready to use.")
else:
    print("\nSome files missing. Download with:")
    print("  $env:HF_ENDPOINT='https://hf-mirror.com'; python scripts/download_zipformer_zh.py")
