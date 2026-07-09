#!/usr/bin/env python3
"""
解析已有评测结果，打印正确率汇总和答错题目详情
用法: python parse_results.py [结果目录路径]
      如不指定路径，自动使用 outputs/ 下最新的目录
"""

import json
import sys
from pathlib import Path


def find_latest_output() -> Path:
    """找到 outputs/ 下最新的结果目录"""
    outputs_root = Path(__file__).parent / "outputs"
    if not outputs_root.exists():
        print("错误: outputs/ 目录不存在")
        sys.exit(1)
    dirs = sorted(outputs_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not dirs:
        print("错误: outputs/ 目录下没有结果目录")
        sys.exit(1)
    return dirs[0]


def find_model_dir(reviews_dir: Path) -> str:
    """自动检测 reviews/ 下的模型目录名"""
    if not reviews_dir.exists():
        return None
    subdirs = [p for p in reviews_dir.iterdir() if p.is_dir()]
    if subdirs:
        return subdirs[0].name
    return None


def parse_results(output_dir: Path) -> dict:
    """解析评测结果，返回 {subset: {accuracy, correct, total, errors}}"""
    results = {}
    reviews_dir = output_dir / "reviews"

    # 自动检测模型目录
    model_dir_name = find_model_dir(reviews_dir)
    if not model_dir_name:
        print(f"警告: 结果目录不存在 {reviews_dir}")
        return results

    model_dir = reviews_dir / model_dir_name
    print(f"使用模型目录: {model_dir_name}")

    for jsonl_file in sorted(model_dir.glob("general_mcq_*.jsonl")):
        subset = jsonl_file.stem[len("general_mcq_"):]
        scores = []
        errors = []

        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                acc = item.get("sample_score", {}).get("score", {}).get("value", {}).get("acc", 0)
                scores.append(acc)

                if acc == 0:
                    target = item.get("target", "")
                    pred = item.get("sample_score", {}).get("score", {}).get("extracted_prediction", "")
                    sample_id = item.get("sample_metadata", {}).get("id", item.get("index", ""))
                    question_raw = item.get("messages", [{}])[0].get("content", "")
                    errors.append({
                        "id": sample_id,
                        "question": question_raw,
                        "model_answer": pred,
                        "correct_answer": target
                    })

        total = len(scores)
        if total == 0:
            continue
        correct = int(sum(scores))
        accuracy = sum(scores) / total
        results[subset] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "errors": errors
        }

    return results


def print_summary(results: dict):
    """打印每个子维度的正确率汇总表，并输出答错题目详情"""
    print("\n" + "=" * 70)
    print("评测结果汇总")
    print("=" * 70)

    # 按大类分组
    grouped = {}
    all_errors = []
    for subset, data in results.items():
        parts = subset.split("_", 1)
        if len(parts) == 2:
            category = parts[0]
            name = parts[1]
        else:
            category = "其他"
            name = subset

        if category not in grouped:
            grouped[category] = []
        grouped[category].append((name, data))

        for err in data.get("errors", []):
            all_errors.append((name, err))

    # 打印分组正确率
    for category in sorted(grouped.keys()):
        print(f"\n[{category}]")
        for name, data in grouped[category]:
            acc = data["accuracy"]
            cor = data["correct"]
            tot = data["total"]
            mark = " [错误]" if acc < 1.0 else ""
            print(f"  {name:<30} {cor}/{tot} = {acc:.1%}{mark}")

    # 打印总计
    total_correct = sum(d["correct"] for d in results.values())
    total_questions = sum(d["total"] for d in results.values())
    overall_acc = total_correct / total_questions if total_questions > 0 else 0

    print("\n" + "-" * 70)
    print(f"总计: {total_correct}/{total_questions} = {overall_acc:.1%}")
    print("=" * 70)

    # 打印答错题目详情
    if all_errors:
        print("\n答错题目详情:")
        print("-" * 70)
        for i, (name, err) in enumerate(all_errors, 1):
            print(f"\n[{i}] 维度: {name}")
            print(f"    题目ID: {err['id']}")
            # 提取问题文本（去掉 prompt 模板前缀）
            q = err["question"]
            if "问题：" in q:
                q = q.split("问题：", 1)[1]
            if "选项：" in q:
                q = q.split("选项：")[0].strip()
            # 去掉换行，保留全部文字
            q_short = q.replace("\n", " ")
            print(f"    题目: {q_short}")
            print(f"    模型回答: {err['model_answer']}")
            print(f"    正确答案: {err['correct_answer']}")
        print("\n" + "=" * 70)

    print()


def save_errors(results: dict, output_dir: Path):
    """将答错题目详情保存到文件"""
    all_errors = []
    for subset, data in results.items():
        for err in data.get("errors", []):
            all_errors.append({
                "dimension": subset,
                "id": err["id"],
                "question": err["question"],
                "model_answer": err["model_answer"],
                "correct_answer": err["correct_answer"]
            })

    if not all_errors:
        print("没有答错题目，无需保存错误详情")
        return

    # 保存为 JSON
    json_path = output_dir / "error_details.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_errors, f, ensure_ascii=False, indent=2)
    print(f"\n答错题目已保存到: {json_path}")

    # 保存为可读文本
    txt_path = output_dir / "error_details.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("答错题目详情\n")
        f.write("=" * 70 + "\n\n")
        for i, err in enumerate(all_errors, 1):
            f.write(f"[{i}] 维度: {err['dimension']}\n")
            f.write(f"    题目ID: {err['id']}\n")
            q = err["question"].replace("\n", " ")
            f.write(f"    题目: {q}\n")
            f.write(f"    模型回答: {err['model_answer']}\n")
            f.write(f"    正确答案: {err['correct_answer']}\n\n")
    print(f"答错题目已保存到: {txt_path}")


def main():
    # 获取结果目录
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        output_dir = find_latest_output()
        print(f"使用结果目录: {output_dir}")

    if not output_dir.exists():
        print(f"错误: 目录不存在 {output_dir}")
        sys.exit(1)

    # 解析结果
    print("解析结果...")
    results = parse_results(output_dir)

    if not results:
        print("未解析到任何结果")
        sys.exit(1)

    # 打印汇总
    print_summary(results)

    # 保存答错题目详情
    save_errors(results, output_dir)


if __name__ == "__main__":
    main()
