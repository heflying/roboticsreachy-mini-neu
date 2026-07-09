"""
compare_results.py - 模型速度对比脚本

读取 result/ 下各模型子目录中的 stats.json，生成 HTML 对比报告。

用法：
  python compare_results.py
  python compare_results.py --result-dir ../result
  python compare_results.py --output comparison.html
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path


def load_all_stats(result_dir: str) -> list:
    """加载 result_dir 下所有子目录中的 stats.json"""
    models = []
    base = Path(result_dir)
    if not base.exists():
        print(f"错误: 结果目录不存在: {result_dir}")
        sys.exit(1)

    for subdir in sorted(base.iterdir()):
        if not subdir.is_dir():
            continue
        stats_path = subdir / "stats.json"
        if not stats_path.exists():
            print(f"跳过 {subdir.name}: 没有 stats.json（请先运行 test_streaming_tts.py）")
            continue
        try:
            with open(stats_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            models.append(data)
            print(f"已加载: {subdir.name}  ({data['summary']['success_count']} 条)")
        except Exception as e:
            print(f"警告: 加载 {stats_path} 失败: {e}")

    if not models:
        print("错误: 未找到任何 stats.json 文件")
        sys.exit(1)

    return models


def build_summary_table(models: list) -> str:
    """构建汇总对比表 HTML"""
    rows = ""
    for m in models:
        s = m["summary"]
        name = m["model"]
        ttft_avg = s.get('average_ttft_s', 0)
        rows += f"""
        <tr>
            <td class="model-name">{name}</td>
            <td>{s['success_count']}/{m['total_sentences']}</td>
            <td>{s['total_synthesis_time_s']:.2f}s</td>
            <td>{s['total_audio_duration_s']:.2f}s</td>
            <td>{s['average_synthesis_time_s']:.4f}s</td>
            <td class="ttft">{ttft_avg:.4f}s</td>
            <td class="rtf">RTF {s['average_rtf']:.1f}x</td>
            <td>{s.get('total_wall_time_s', 0):.1f}s</td>
        </tr>"""

    return f"""
    <h2>速度汇总对比</h2>
    <table>
        <thead>
            <tr>
                <th>模型</th>
                <th>成功率</th>
                <th>总合成耗时</th>
                <th>总音频时长</th>
                <th>平均单句耗时</th>
                <th>平均首音延迟 (TTFT)</th>
                <th>平均实时率</th>
                <th>壁钟时间</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>"""


def build_per_sentence_table(models: list) -> str:
    """构建逐句对比表（横向对比各模型在同一句子上的速度）"""
    # 收集所有句子文本（用第一个模型为准）
    base_records = models[0].get("records", [])
    sentence_map = {}  # index -> text
    for r in base_records:
        if r.get("success"):
            sentence_map[r["index"]] = r["text"]

    # 构建 model_name -> {index -> synthesis_time_s}
    model_data = {}
    for m in models:
        name = m["model"]
        model_data[name] = {}
        for r in m.get("records", []):
            if r.get("success"):
                model_data[name][r["index"]] = r["synthesis_time_s"]

    # 找出每句的最快模型（用于高亮）
    model_names = [m["model"] for m in models]

    header = "<tr><th>#</th><th>句子</th>" + "".join(f"<th>{n}</th>" for n in model_names) + "</tr>"

    rows = ""
    for idx in sorted(sentence_map.keys()):
        text = sentence_map[idx]
        display_text = text[:25] + "…" if len(text) > 25 else text
        times = []
        for name in model_names:
            t = model_data[name].get(idx)
            if t is not None:
                times.append(t)
            else:
                times.append(float("inf"))

        min_t = min(times)
        cells = ""
        for name, t in zip(model_names, times):
            if t == float("inf"):
                cells += '<td class="fail">—</td>'
            elif t == min_t and len([x for x in times if x == min_t]) == 1:
                cells += f'<td class="best">{t:.3f}s</td>'
            else:
                cells += f"<td>{t:.3f}s</td>"

        rows += f"<tr><td>{idx}</td><td class=\"text\">{display_text}</td>{cells}</tr>"

    return f"""
    <h2>逐句合成耗时对比（秒）</h2>
    <p class="hint">绿色 = 该句耗时最短的模型</p>
    <div class="scroll-wrap">
    <table>
        <thead>{header}</thead>
        <tbody>{rows}</tbody>
    </table>
    </div>"""


def build_per_sentence_ttft_table(models: list) -> str:
    """构建逐句 TTFT 对比表"""
    base_records = models[0].get("records", [])
    sentence_map = {}
    for r in base_records:
        if r.get("success"):
            sentence_map[r["index"]] = r["text"]

    model_data = {}
    for m in models:
        name = m["model"]
        model_data[name] = {}
        for r in m.get("records", []):
            if r.get("success"):
                model_data[name][r["index"]] = r.get("ttft_s", 0)

    model_names = [m["model"] for m in models]
    header = "<tr><th>#</th><th>句子</th>" + "".join(f"<th>{n}</th>" for n in model_names) + "</tr>"

    rows = ""
    for idx in sorted(sentence_map.keys()):
        text = sentence_map[idx]
        display_text = text[:25] + "…" if len(text) > 25 else text
        times = [model_data[name].get(idx, float("inf")) for name in model_names]
        min_t = min(times)
        cells = ""
        for name, t in zip(model_names, times):
            if t == float("inf"):
                cells += '<td class="fail">—</td>'
            elif t == min_t and len([x for x in times if x == min_t]) == 1:
                cells += f'<td class="best">{t:.4f}s</td>'
            else:
                cells += f"<td>{t:.4f}s</td>"

        rows += f"<tr><td>{idx}</td><td class=\"text\">{display_text}</td>{cells}</tr>"

    return f"""
    <h2>逐句首音延迟 (TTFT) 对比（秒）</h2>
    <p class="hint">绿色 = 该句 TTFT 最短的模型</p>
    <div class="scroll-wrap">
    <table>
        <thead>{header}</thead>
        <tbody>{rows}</tbody>
    </table>
    </div>"""


def build_rtf_bar(models: list) -> str:
    """构建 RTF 柱状对比（CSS 条形图）"""
    max_rtf = max(m["summary"]["average_rtf"] for m in models)

    bars = ""
    for m in models:
        s = m["summary"]
        rtf = s["average_rtf"]
        pct = (rtf / max_rtf * 90) if max_rtf > 0 else 0
        bars += f"""
        <div class="bar-row">
            <span class="bar-label">{m['model']}</span>
            <span class="bar-track">
                <span class="bar-fill" style="width:{pct}%"></span>
            </span>
            <span class="bar-value">{rtf:.1f}x</span>
        </div>"""

    return f"""
    <h2>平均实时率 (RTF) 对比</h2>
    <p class="hint">RTF = 音频时长 / 合成耗时，越高越快</p>
    <div class="bar-chart">{bars}</div>"""


def build_ttft_bar(models: list) -> str:
    """构建 TTFT 柱状对比（首音延迟，越低越好）"""
    max_ttft = max(m["summary"].get("average_ttft_s", 0) for m in models)
    if max_ttft <= 0:
        return ""

    bars = ""
    for m in models:
        s = m["summary"]
        ttft = s.get("average_ttft_s", 0)
        pct = (ttft / max_ttft * 90) if max_ttft > 0 else 0
        bars += f"""
        <div class="bar-row">
            <span class="bar-label">{m['model']}</span>
            <span class="bar-track">
                <span class="bar-fill-ttft" style="width:{pct}%"></span>
            </span>
            <span class="bar-value">{ttft:.4f}s</span>
        </div>"""

    return f"""
    <h2>平均首音延迟 (TTFT) 对比</h2>
    <p class="hint">首音延迟 = 从调用到第一个音频 chunk 就绪的耗时，越低越好</p>
    <div class="bar-chart">{bars}</div>"""


def build_ranking(models: list) -> str:
    """速度排名"""
    sorted_models = sorted(models, key=lambda m: m["summary"]["average_rtf"], reverse=True)

    rows = ""
    for rank, m in enumerate(sorted_models, 1):
        s = m["summary"]
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"{rank}.")
        ttft = s.get("average_ttft_s", 0)
        rows += f"""
        <tr>
            <td class="rank">{medal}</td>
            <td>{m['model']}</td>
            <td>{s['average_rtf']:.1f}x</td>
            <td>{s['average_synthesis_time_s']:.4f}s/句</td>
            <td>{ttft:.4f}s</td>
            <td>{s['total_synthesis_time_s']:.2f}s</td>
        </tr>"""

    return f"""
    <h2>合成速度排名</h2>
    <table>
        <thead>
            <tr><th>排名</th><th>模型</th><th>平均 RTF</th><th>平均单句耗时</th><th>平均 TTFT</th><th>总合成耗时</th></tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>"""


def generate_html(models: list, output_path: str):
    """生成完整 HTML 报告"""
    title = "TTS 模型速度对比报告"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: -apple-system, "Microsoft YaHei", "PingFang SC", sans-serif;
        background: #f5f6fa; color: #2d3436; line-height: 1.6;
        padding: 20px;
    }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    h1 {{
        font-size: 24px; margin-bottom: 6px; color: #1a1a2e;
    }}
    h2 {{
        font-size: 18px; margin: 32px 0 12px; padding-bottom: 6px;
        border-bottom: 2px solid #6c5ce7; color: #1a1a2e;
    }}
    .meta {{ color: #888; font-size: 13px; margin-bottom: 24px; }}
    .hint {{ color: #888; font-size: 12px; margin-bottom: 8px; }}

    table {{
        width: 100%; border-collapse: collapse; background: #fff;
        border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.06);
        font-size: 13px;
    }}
    th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee; }}
    th {{ background: #6c5ce7; color: #fff; font-weight: 600; white-space: nowrap; }}
    tr:hover td {{ background: #f8f7ff; }}
    .model-name {{ font-weight: 600; color: #6c5ce7; }}
    .rtf {{ font-weight: 700; color: #00b894; }}
    .ttft {{ font-weight: 700; color: #e17055; }}
    .best {{ background: #d4edda !important; font-weight: 600; color: #155724; }}
    .fail {{ color: #ccc; }}
    .text {{ white-space: nowrap; }}
    .rank {{ font-size: 18px; text-align: center; }}

    .scroll-wrap {{ overflow-x: auto; }}

    /* 条形图 */
    .bar-chart {{ background: #fff; border-radius: 8px; padding: 16px 20px;
                  box-shadow: 0 1px 4px rgba(0,0,0,.06); }}
    .bar-row {{ display: flex; align-items: center; margin: 10px 0; }}
    .bar-label {{ width: 220px; font-size: 13px; font-weight: 600; flex-shrink: 0; }}
    .bar-track {{ flex: 1; height: 24px; background: #eee; border-radius: 12px;
                  overflow: hidden; margin: 0 12px; }}
    .bar-fill {{
        height: 100%; background: linear-gradient(90deg, #6c5ce7, #a29bfe);
        border-radius: 12px; transition: width 0.3s;
    }}
    .bar-fill-ttft {{
        height: 100%; background: linear-gradient(90deg, #e17055, #fab1a0);
        border-radius: 12px; transition: width 0.3s;
    }}
    .bar-value {{ width: 60px; font-size: 13px; font-weight: 700; color: #6c5ce7; }}

    footer {{ margin-top: 40px; padding-top: 16px; border-top: 1px solid #ddd;
             color: #aaa; font-size: 12px; text-align: center; }}
</style>
</head>
<body>
<div class="container">
<h1>{title}</h1>
<p class="meta">生成时间: {now} | 共 {len(models)} 个模型</p>

{build_summary_table(models)}
{build_rtf_bar(models)}
{build_ttft_bar(models)}
{build_ranking(models)}
{build_per_sentence_table(models)}
{build_per_sentence_ttft_table(models)}

<footer>由 compare_results.py 自动生成</footer>
</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n报告已生成: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="TTS 模型速度对比工具")
    parser.add_argument(
        "--result-dir", type=str, default="result",
        help="结果目录（相对于工作目录，默认 result）"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="输出 HTML 文件路径（默认 result/comparison.html）"
    )
    args = parser.parse_args()

    # 路径处理
    result_dir = args.result_dir
    if not os.path.isabs(result_dir):
        result_dir = os.path.join(os.getcwd(), result_dir)

    if args.output:
        output_path = args.output
        if not os.path.isabs(output_path):
            output_path = os.path.join(os.getcwd(), output_path)
    else:
        output_path = os.path.join(result_dir, "comparison.html")

    print(f"结果目录: {result_dir}")
    print(f"输出文件: {output_path}")

    # 加载数据
    models = load_all_stats(result_dir)

    # 生成报告
    generate_html(models, output_path)


if __name__ == "__main__":
    main()
