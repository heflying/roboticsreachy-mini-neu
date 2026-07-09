"""Download Kokoro-82M-v1.1-zh model files to models/kokoro-zh/ for offline use."""

from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "hexgrad/Kokoro-82M-v1.1-zh"
MODEL_DIR = Path("models/kokoro-zh")

FILES = [
    "config.json",
    "kokoro-v1_1-zh.pth",
]

DEFAULT_VOICES = ["zf_001"]


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    voices_dir = MODEL_DIR / "voices"
    voices_dir.mkdir(exist_ok=True)

    for f in FILES:
        dst = MODEL_DIR / f
        if dst.exists():
            print(f"  [skip] {dst} already exists")
            continue
        print(f"  [download] {f} -> {dst}")
        path = hf_hub_download(repo_id=REPO_ID, filename=f, local_dir=str(MODEL_DIR))
        print(f"  [done] {path}")

    for voice in DEFAULT_VOICES:
        voice_file = voices_dir / f"{voice}.pt"
        if voice_file.exists():
            print(f"  [skip] {voice_file} already exists")
            continue
        filename = f"voices/{voice}.pt"
        print(f"  [download] {filename} -> {voice_file}")
        hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=str(MODEL_DIR))
        print(f"  [done]")

    print(f"\nAll files saved to {MODEL_DIR}/")
    print("Contents:")
    for p in sorted(MODEL_DIR.rglob("*")):
        if p.is_file():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  {p.relative_to(MODEL_DIR)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
