"""临时脚本：修复已有 JSON 结果中的 accuracy 字段。

将 accuracy 从 1 - CER 修正为 total_hits / total_reference_length。

用法:
    cd eval_jiwer
    python fix_accuracy.py
    python fix_accuracy.py --glob "results/*.json"   # 自定义匹配
"""

import argparse
import json
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = THIS_DIR / "results"


def fix_one(path: Path) -> bool:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    changed = False
    for r in data.get("results", []):
        cer = r.get("aggregate_cer", {})
        total_hits = cer.get("total_hits", 0)
        total_ref = cer.get("total_reference_length", 0)
        if total_ref > 0:
            new_acc = round(total_hits / total_ref, 6)
        else:
            new_acc = 0.0

        old_acc = cer.get("accuracy")
        if old_acc != new_acc:
            cer["accuracy"] = new_acc
            changed = True
            print(f"  {path.name}: accuracy {old_acc} -> {new_acc}")

    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="修复 JSON 结果中的 accuracy 字段")
    parser.add_argument("--glob", default="*.json", help="匹配模式 (默认: results/*.json)")
    args = parser.parse_args()

    if "/" in args.glob or "\\" in args.glob:
        pattern = Path(args.glob)
        files = sorted(pattern.parent.glob(pattern.name))
    else:
        files = sorted((RESULTS_DIR).glob(args.glob))

    if not files:
        print(f"No files matching: {args.glob}")
        return

    print(f"Found {len(files)} file(s):")
    for fp in files:
        print(f"  {fp}")

    n_fixed = 0
    for fp in files:
        if fix_one(fp):
            n_fixed += 1

    print(f"\nDone. Fixed {n_fixed} file(s).")


if __name__ == "__main__":
    main()
