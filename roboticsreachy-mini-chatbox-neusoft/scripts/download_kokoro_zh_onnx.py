"""Download Kokoro-82M-v1.1-zh ONNX model files to models/kokoro-zh-onnx/ for offline use.

Downloads both normal (model.onnx) and quantized (model_quantized.onnx) variants
from the onnx-community/Kokoro-82M-v1.1-zh-ONNX HuggingFace repo.
"""

from pathlib import Path

from huggingface_hub import hf_hub_download
import numpy as np

REPO_ID = "onnx-community/Kokoro-82M-v1.1-zh-ONNX"
MODEL_DIR = Path("models/kokoro-zh-onnx")

# ONNX model files
MODEL_FILES = [
    "onnx/model.onnx",
    "onnx/model_quantized.onnx",
]

# Config/tokenizer files (root level, NOT inside onnx/)
CONFIG_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
]

DEFAULT_VOICES = ["zf_001"]


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    onnx_dir = MODEL_DIR / "onnx"
    onnx_dir.mkdir(exist_ok=True)
    voices_dir = MODEL_DIR / "voices"
    voices_dir.mkdir(exist_ok=True)

    all_files = MODEL_FILES + CONFIG_FILES

    print(f"Downloading from {REPO_ID} -> {MODEL_DIR}/\n")

    for f in all_files:
        dst = MODEL_DIR / f
        if dst.exists():
            size_mb = dst.stat().st_size / (1024 * 1024)
            print(f"  [skip] {dst} ({size_mb:.1f} MB)")
            continue
        print(f"  [download] {f} -> {dst}")
        hf_hub_download(repo_id=REPO_ID, filename=f, local_dir=str(MODEL_DIR))
        print(f"  [done]")

    print()
    for voice in DEFAULT_VOICES:
        voice_file = voices_dir / f"{voice}.bin"
        if voice_file.exists():
            print(f"  [skip] {voice_file} already exists")
            continue
        filename = f"voices/{voice}.bin"
        print(f"  [download] {filename} -> {voice_file}")
        hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=str(MODEL_DIR))
        print(f"  [done]")

    # Combine individual voice .bin files into voices.npz for kokoro-onnx
    print("\nCombining voice files into voices.npz...")
    voices_dict = {}
    for voice_file in sorted(voices_dir.glob("*.bin")):
        voice_name = voice_file.stem
        try:
            data = np.fromfile(str(voice_file), dtype=np.float32).reshape(-1, 256)
            voices_dict[voice_name] = data
            print(f"  Loaded: {voice_name}")
        except Exception as e:
            print(f"  [error] Failed to load {voice_name}: {e}")

    if voices_dict:
        npz_path = MODEL_DIR / "voices.npz"
        np.savez(str(npz_path), **voices_dict)
        npz_size = npz_path.stat().st_size / (1024 * 1024)
        print(f"  Saved {len(voices_dict)} voices -> voices.npz ({npz_size:.1f} MB)")
    else:
        print("  [warning] No voice files found to combine!")

    print(f"\nAll files saved to {MODEL_DIR}/")
    print("Contents:")
    for p in sorted(MODEL_DIR.rglob("*")):
        if p.is_file():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  {p.relative_to(MODEL_DIR)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
