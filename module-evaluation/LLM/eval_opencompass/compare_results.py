#!/usr/bin/env python3
"""
对比 outputs/ 下所有 cmp_ 开头的目录中的评测结果
生成 HTML 对比报告（Chart.js 柱状图）
用法: python compare_results.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


def find_cmp_dirs(outputs_root: Path) -> list:
    if not outputs_root.exists():
        print("错误: 目录不存在", outputs_root)
        sys.exit(1)
    cmp_dirs = sorted([
        p for p in outputs_root.iterdir()
        if p.is_dir() and p.name.startswith("cmp_")
    ], key=lambda p: p.name)
    return cmp_dirs


def find_model_dir(reviews_dir: Path) -> str:
    if not reviews_dir.exists():
        return None
    subdirs = [p for p in reviews_dir.iterdir() if p.is_dir()]
    if subdirs:
        return subdirs[0].name
    return None


def parse_cmp_dir(cmp_dir: Path) -> dict:
    results = {}
    reviews_dir = cmp_dir / "reviews"
    model_dir_name = find_model_dir(reviews_dir)
    if not model_dir_name:
        print("  警告:", cmp_dir.name, "中没有找到 reviews 子目录")
        return results

    model_dir = reviews_dir / model_dir_name
    for jsonl_file in sorted(model_dir.glob("general_mcq_*.jsonl")):
        subset = jsonl_file.stem[len("general_mcq_"):]
        scores = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                acc = item.get("sample_score", {}).get("score", {}).get("value", {}).get("acc", 0)
                scores.append(acc)
        total = len(scores)
        if total == 0:
            continue
        correct = int(sum(scores))
        accuracy = sum(scores) / total
        results[subset] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }
    return results


def build_comparison(cmp_dirs: list) -> dict:
    cmp_names = [d.name for d in cmp_dirs]
    all_cmp_results = {}

    print("解析各 cmp_ 目录...")
    for cmp_dir in cmp_dirs:
        print("  解析:", cmp_dir.name)
        results = parse_cmp_dir(cmp_dir)
        all_cmp_results[cmp_dir.name] = results
        print("    找到", len(results), "个子维度")

    all_subsets = set()
    for results in all_cmp_results.values():
        all_subsets.update(results.keys())

    # 大类编号 -> 完整大类名（编号+名称）的映射
    # 如 "03" -> "03_智力"
    category_full_name = {}

    grouped = defaultdict(lambda: {"subsets": [], "subset_data": defaultdict(dict)})
    for subset in sorted(all_subsets):
        parts = subset.split("_", 2)
        if len(parts) >= 2:
            category_id = parts[0]  # 如 "03"
            category_name = parts[1] if len(parts) > 1 else ""  # 如 "智力"
            # 拼接完整大类名：编号_名称
            full_name = category_id + "_" + category_name
            category_full_name[category_id] = full_name
            name = parts[2] if len(parts) > 2 else ""
        else:
            category_id = "其他"
            category_full_name[category_id] = "其他"
            name = subset
        grouped[category_id]["subsets"].append((subset, name))

    for category_id, info in grouped.items():
        for subset, name in info["subsets"]:
            for cmp_name in cmp_names:
                acc = all_cmp_results.get(cmp_name, {}).get(subset, {}).get("accuracy", None)
                info["subset_data"][cmp_name][subset] = acc

    overall = {}
    for cmp_name in cmp_names:
        results = all_cmp_results.get(cmp_name, {})
        if not results:
            overall[cmp_name] = None
            continue
        total_correct = sum(d["correct"] for d in results.values())
        total_questions = sum(d["total"] for d in results.values())
        overall[cmp_name] = total_correct / total_questions if total_questions > 0 else None

    return {
        "cmp_names": cmp_names,
        "overall": overall,
        "by_category": {k: v for k, v in sorted(grouped.items())},
        "category_full_name": category_full_name  # 新增：大类编号 -> 完整名称
    }


def make_dataset(label, data, colors, border_colors):
    n = len(data)
    return {
        "label": label,
        "data": data,
        "backgroundColor": colors[:n],
        "borderColor": border_colors[:n],
        "borderWidth": 1
    }


def chart_js(chart_id, title, labels, dataset, title_size=15):
    """返回 new Chart(...) 的 JS 代码字符串"""
    cfg = {
        "type": "bar",
        "data": {
            "labels": labels,
            "datasets": [dataset]
        },
        "options": {
            "responsive": True,
            "plugins": {
                "title": {
                    "display": True,
                    "text": title,
                    "font": {"size": title_size}
                }
            },
            "scales": {
                "y": {
                    "beginAtZero": True,
                    "max": 100,
                    "title": {"display": True, "text": "正确率 (%)"}
                }
            }
        }
    }
    cfg_js = json.dumps(cfg, ensure_ascii=False)
    return "new Chart(document.getElementById('" + chart_id + "'), " + cfg_js + ");"


def generate_html(comp: dict, output_path: Path):
    cmp_names = comp["cmp_names"]
    overall = comp["overall"]
    by_category = comp["by_category"]
    category_full_name = comp.get("category_full_name", {})  # 大类编号 -> 完整名称

    colors = [
        "rgba(54, 162, 235, 0.7)",
        "rgba(255, 99, 132, 0.7)",
        "rgba(75, 192, 192, 0.7)",
        "rgba(255, 206, 86, 0.7)",
        "rgba(153, 102, 255, 0.7)",
        "rgba(255, 159, 64, 0.7)",
    ]
    border_colors = [c.replace("0.7", "1") for c in colors]
    cmp_count = len(cmp_names)

    # 整体排名表格
    sorted_overall = sorted(
        [(n, overall[n]) for n in cmp_names if overall[n] is not None],
        key=lambda x: x[1], reverse=True
    )
    table_rows = ""
    for rank, (name, acc) in enumerate(sorted_overall, 1):
        if rank == 1:
            cls = "rank-1"
        elif rank == 2:
            cls = "rank-2"
        elif rank == 3:
            cls = "rank-3"
        else:
            cls = ""
        table_rows += ('            <tr><td class="' + cls + '">' + str(rank) +
                       '</td><td>' + name + '</td><td>' +
                       '{:.1f}%'.format(acc*100) + '</td></tr>\n')
    for name in cmp_names:
        if overall[name] is None:
            table_rows += ('            <tr><td>-</td><td>' + name +
                           '</td><td>N/A</td></tr>\n')

    # 所有 Chart.js 代码
    charts_parts = []

    # 整体柱状图
    ds = make_dataset("整体正确率 (%)",
                       [overall[n]*100 if overall[n] is not None else None for n in cmp_names],
                       colors, border_colors)
    charts_parts.append(chart_js("chart_overall", "整体正确率对比", cmp_names, ds, title_size=16))

    # 各大类图表
    sections_html = ""
    for category, info in by_category.items():
        # 获取完整大类名（编号_名称）
        cat_display = category_full_name.get(category, category)

        # 大类整体正确率
        category_acc = {}
        for cmp_name in cmp_names:
            accs = [info["subset_data"][cmp_name].get(subset)
                     for subset, _ in info["subsets"]
                     if info["subset_data"][cmp_name].get(subset) is not None]
            category_acc[cmp_name] = sum(accs) / len(accs) * 100 if accs else None

        ds = make_dataset(cat_display + " 整体正确率 (%)",
                          [category_acc[n] for n in cmp_names],
                          colors, border_colors)
        charts_parts.append(chart_js("chart_" + category, cat_display + " - 整体正确率对比",
                                      cmp_names, ds, title_size=15))

        sections_html += '    <div class="category-section">\n'
        sections_html += '        <h2>' + cat_display + '</h2>\n'
        sections_html += '        <div class="chart-container">\n'
        sections_html += '            <canvas id="chart_' + category + '"></canvas>\n'
        sections_html += '        </div>\n'

        for subset, name in info["subsets"]:
            if len(name) > 30:
                display_name = name[:30] + "..."
            else:
                display_name = name
            subset_data = [
                info["subset_data"][n].get(subset) * 100
                if info["subset_data"][n].get(subset) is not None else None
                for n in cmp_names
            ]
            ds = make_dataset("正确率 (%)", subset_data, colors, border_colors)
            charts_parts.append(chart_js("chart_" + subset,
                                          cat_display + " - " + display_name,
                                          cmp_names, ds, title_size=14))

            sections_html += '        <div class="chart-container sub-chart">\n'
            sections_html += '            <canvas id="chart_' + subset + '"></canvas>\n'
            sections_html += '        </div>\n'

        sections_html += '    </div>\n'

    charts_js = "\n    ".join(charts_parts)

    # HTML 模板（不使用 f-string，避免花括号转义问题）
    css = """\
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .overall-section {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .category-section {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .category-section h2 {
            color: #555;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 8px;
        }
        .chart-container {
            position: relative;
            height: 400px;
            margin: 20px 0;
        }
        .sub-chart {
            height: 350px;
            margin-left: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 10px;
            text-align: center;
            border-bottom: 1px solid #ddd;
        }
        th {
            background: #4CAF50;
            color: white;
        }
        tr:hover {
            background: #f5f5f5;
        }
        .rank-1 { color: #FFD700; font-weight: bold; }
        .rank-2 { color: #C0C0C0; font-weight: bold; }
        .rank-3 { color: #CD7F32; font-weight: bold; }"""

    html = ('<!DOCTYPE html>\n'
             '<html lang="zh-CN">\n'
             '<head>\n'
             '    <meta charset="UTF-8">\n'
             '    <title>LLM 评测对比报告</title>\n'
             '    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>\n'
             '    <style>\n' + css + '\n'
             '    </style>\n'
             '</head>\n'
             '<body>\n'
             '    <h1>LLM 评测对比报告</h1>\n'
             '\n'
             '    <div class="overall-section">\n'
             '        <h2>整体正确率排名</h2>\n'
             '        <table>\n'
             '            <tr><th>排名</th><th>模型/跑次</th><th>整体正确率</th></tr>\n' +
             table_rows +
             '        </table>\n'
             '        <div class="chart-container">\n'
             '            <canvas id="chart_overall"></canvas>\n'
             '        </div>\n'
             '    </div>\n'
             '\n' +
             sections_html +
             '\n'
             '    <script>\n'
             '    ' + charts_js + '\n'
             '    </script>\n'
             '</body>\n'
             '</html>')

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print("\n对比报告已生成:", output_path)


def main():
    script_dir = Path(__file__).parent
    outputs_dir = script_dir / "outputs"

    print("=" * 60)
    print("LLM 评测结果对比")
    print("=" * 60)

    cmp_dirs = find_cmp_dirs(outputs_dir)
    if not cmp_dirs:
        print("错误:", outputs_dir, "下没有找到 cmp_ 开头的目录")
        print("请将需要对比的评测结果目录重命名为 cmp_ 开头")
        sys.exit(1)

    print("找到", len(cmp_dirs), "个对比目录:")
    for d in cmp_dirs:
        print("  -", d.name)

    comp = build_comparison(cmp_dirs)
    output_path = outputs_dir / "comparison_report.html"
    generate_html(comp, output_path)
    print("\n请打开查看对比报告:", output_path)


if __name__ == "__main__":
    main()
