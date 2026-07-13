"""对比多个评测结果，生成 HTML 报告。

扫描 results/ 目录下 cmp_ 开头的 JSON 文件，
汇总 CER、RTF、TTFC 等指标，输出对比表格 HTML。

用法:
    cd eval_jiwer
    python compare.py                     # 自动扫描 results/cmp_*.json
    python compare.py --glob "results/*.json"  # 自定义 glob
    python compare.py --output compare.html    # 指定输出路径
"""

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = THIS_DIR / "results"
DEFAULT_GLOB = "cmp_*.json"   # 在 results/ 下扫描
DEFAULT_OUTPUT = "results/compare.html"


# ────────────────────────────────────────────────────────────
# 数据加载
# ────────────────────────────────────────────────────────────

def discover_files(glob_pattern: str) -> list[Path]:
    """在 results/ 目录下按 glob 模式查找 JSON 文件。"""
    results_dir = Path(glob_pattern).parent if "/" in glob_pattern or "\\" in glob_pattern else RESULTS_DIR
    pattern = Path(glob_pattern).name
    files = sorted(results_dir.glob(pattern))
    if not files:
        print(f"[WARN] No files matching '{glob_pattern}' in {results_dir}")
    return files


def load_result(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ────────────────────────────────────────────────────────────
# HTML 生成
# ────────────────────────────────────────────────────────────

CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background: #f5f7fa; color: #333; padding: 20px;
}
h1 { text-align: center; margin-bottom: 8px; font-size: 24px; }
.subtitle { text-align: center; color: #888; margin-bottom: 24px; font-size: 13px; }

.file-tag {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 12px; font-weight: 600; margin: 0 4px;
}

/* 摘要卡片 */
.summary-grid {
    display: grid; gap: 20px; margin-bottom: 28px;
}
.summary-grid.cols-2 { grid-template-columns: 1fr 1fr; }
.summary-grid.cols-3 { grid-template-columns: 1fr 1fr 1fr; }
.summary-grid.cols-4 { grid-template-columns: 1fr 1fr 1fr 1fr; }

.model-card {
    background: #fff; border-radius: 10px; padding: 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
    border-top: 4px solid #4a90d9;
}
.model-card:nth-child(2) { border-top-color: #e6833a; }
.model-card:nth-child(3) { border-top-color: #50b86c; }
.model-card:nth-child(4) { border-top-color: #9b59b6; }

.model-card h3 { font-size: 15px; margin-bottom: 6px; }
.model-card .meta-line { font-size: 12px; color: #999; margin-bottom: 12px; }

.mini-stat { display: flex; justify-content: space-between; padding: 4px 0; font-size: 14px; border-bottom: 1px solid #eee; }
.mini-stat:last-child { border-bottom: none; }
.mini-stat .label { color: #666; }
.mini-stat .value { font-weight: 600; }

/* 表格 */
h2 { margin: 28px 0 12px; font-size: 18px; }
table {
    width: 100%; border-collapse: collapse; background: #fff;
    border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.06);
    margin-bottom: 20px;
}
thead { background: #f0f2f5; }
th { padding: 10px 14px; text-align: left; font-size: 13px; color: #555; font-weight: 600; }
td { padding: 8px 14px; font-size: 13px; border-bottom: 1px solid #f0f2f5; }
tr:last-child td { border-bottom: none; }

.best { background: #d4edda; font-weight: 700; }
.cer-bad { color: #c0392b; font-weight: 600; }

.footer { text-align: center; color: #bbb; font-size: 12px; margin-top: 40px; }
"""

COLORS = ["#4a90d9", "#e6833a", "#50b86c", "#9b59b6", "#e74c3c", "#1abc9c"]


def file_color(idx: int) -> str:
    return COLORS[idx % len(COLORS)]


def _merge_cer(all_results: list[dict]) -> dict:
    """合并多个数据集的 CER 指标，全部按 reference_length 加权平均。"""
    total_hits = sum(r.get("aggregate_cer", {}).get("total_hits", 0) for r in all_results)
    total_ref = sum(r.get("aggregate_cer", {}).get("total_reference_length", 0) for r in all_results)
    total_sub = sum(r.get("aggregate_cer", {}).get("total_substitutions", 0) for r in all_results)
    total_del = sum(r.get("aggregate_cer", {}).get("total_deletions", 0) for r in all_results)
    total_ins = sum(r.get("aggregate_cer", {}).get("total_insertions", 0) for r in all_results)
    total_utt = sum(r.get("aggregate_cer", {}).get("total_utterances", 0) for r in all_results)

    # SER: 按句子数加权平均（而非简单平均）
    weighted_ser = 0.0
    for r in all_results:
        cer = r.get("aggregate_cer", {})
        n = cer.get("total_utterances", 0)
        ser = cer.get("ser", 0)
        if n > 0 and ser is not None:
            weighted_ser += ser * n
    avg_ser = weighted_ser / total_utt if total_utt > 0 else None

    merged = {}
    if total_ref > 0:
        merged["cer"] = (total_sub + total_del + total_ins) / total_ref
        merged["accuracy"] = total_hits / total_ref
    merged["ser"] = avg_ser
    merged["sub_rate"] = total_sub / total_ref if total_ref > 0 else None
    merged["del_rate"] = total_del / total_ref if total_ref > 0 else None
    merged["ins_rate"] = total_ins / total_ref if total_ref > 0 else None
    merged["total_hits"] = total_hits
    merged["total_reference_length"] = total_ref
    return merged


def _merge_timing(all_results: list[dict]) -> dict:
    """合并多个数据集的实时性指标。"""
    merged = {}
    total_audio = sum(r.get("aggregate_timing", {}).get("total_audio_duration_s", 0) for r in all_results)
    total_proc = sum(r.get("aggregate_timing", {}).get("total_processing_time_s", 0) for r in all_results)

    # RTF: 按音频时长加权
    if total_audio > 0:
        merged["avg_rtf"] = total_proc / total_audio
    else:
        rtf_list = [r.get("aggregate_timing", {}).get("avg_rtf", 0) for r in all_results if r.get("aggregate_timing", {}).get("avg_rtf")]
        merged["avg_rtf"] = sum(rtf_list) / len(rtf_list) if rtf_list else None

    # TTFC 百分位数：各数据集的加权平均（近似）
    pctiles = ["avg_ttfc_s", "p50_ttfc_s", "p90_ttfc_s", "p95_ttfc_s", "p99_ttfc_s"]
    for key in pctiles:
        vals = [r.get("aggregate_timing", {}).get(key) for r in all_results if r.get("aggregate_timing", {}).get(key) is not None]
        merged[key] = sum(vals) / len(vals) if vals else None

    merged["avg_e2e_latency_s"] = None  # 需要原始数据才能精确计算，暂缺
    merged["total_audio_duration_s"] = total_audio
    merged["total_processing_time_s"] = total_proc
    return merged


def build_html(files: list[Path], data_list: list[dict]) -> str:
    n = len(files)
    col_class = f"cols-{min(n, 4)}"

    # ── 摘要卡片（使用合并后的跨数据集指标）──
    cards = ""
    for i, (fp, d) in enumerate(zip(files, data_list)):
        meta = d.get("meta", {})
        hw = d.get("hardware", {})
        results = d.get("results", [])
        ds_names = ", ".join(r["dataset_name"] for r in results)

        # 合并所有数据集的聚合指标
        agg_cer = _merge_cer(results) if results else {}
        agg_timing = _merge_timing(results) if results else {}

        cards += f"""
    <div class="model-card">
      <h3><span class="file-tag" style="background:{file_color(i)};color:#fff;">{i+1}</span>{meta.get("model_name", fp.stem)}</h3>
      <div class="meta-line">{meta.get("model_type","")} · {ds_names} · {meta.get("timestamp","")[:10]}</div>
      <div class="mini-stat"><span class="label">CER</span><span class="value">{_fmt_pct(agg_cer.get('cer'))}</span></div>
      <div class="mini-stat"><span class="label">Sub/Del/Ins</span><span class="value">{_fmt_pct(agg_cer.get('sub_rate'))} / {_fmt_pct(agg_cer.get('del_rate'))} / {_fmt_pct(agg_cer.get('ins_rate'))}</span></div>
      <div class="mini-stat"><span class="label">SER</span><span class="value">{_fmt_pct(agg_cer.get('ser'))}</span></div>
      <div class="mini-stat"><span class="label">Avg RTF</span><span class="value">{_fmt(agg_timing.get('avg_rtf'), 4)}</span></div>
      <div class="mini-stat"><span class="label">Avg TTFC</span><span class="value">{_fmt(agg_timing.get('avg_ttfc_s'), 3)}s</span></div>
      <div class="mini-stat"><span class="label">P50 TTFC</span><span class="value">{_fmt(agg_timing.get('p50_ttfc_s'), 3)}s</span></div>
      <div class="mini-stat"><span class="label">P95 TTFC</span><span class="value">{_fmt(agg_timing.get('p95_ttfc_s'), 3)}s</span></div>
      <div class="mini-stat"><span class="label">Evaluated</span><span class="value">{results[0].get('num_evaluated','-')} / {results[0].get('num_utterances','-')}</span></div>
    </div>"""

    # ── 硬件对比表 ──
    hw_table = _build_hardware_table(files, data_list)

    # ── CER 对比表（多数据集） ──
    cer_table = _build_cer_table(files, data_list)

    # ── 实时性对比表 ──
    timing_table = _build_timing_table(files, data_list)

    # ── 配置对比表 ──
    config_table = _build_config_table(files, data_list)

    gen_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    metrics_legend = """
<h2>指标说明</h2>
<div style="background:#fff;border-radius:8px;padding:20px;box-shadow:0 1px 4px rgba(0,0,0,.06);margin-bottom:20px;">
  <h3 style="font-size:15px;margin-bottom:12px;color:#333;">准确率 / 错误率指标</h3>
  <table style="width:100%;border-collapse:collapse;font-size:13px;margin-bottom:20px;">
    <thead><tr style="background:#f0f2f5;"><th style="padding:8px 12px;text-align:left;">指标</th><th style="padding:8px 12px;text-align:left;">含义</th><th style="padding:8px 12px;text-align:left;">说明</th></tr></thead>
    <tbody>
      <tr style="border-bottom:1px solid #f0f2f5;"><td style="padding:6px 12px;font-weight:600;">CER</td><td style="padding:6px 12px;">字符错误率 (Character Error Rate)</td><td style="padding:6px 12px;">(Sub+Del+Ins) / 参考文本总字数，越低越好。0% 为完美识别。</td></tr>
      <tr style="border-bottom:1px solid #f0f2f5;"><td style="padding:6px 12px;font-weight:600;">Accuracy</td><td style="padding:6px 12px;">字符正确率</td><td style="padding:6px 12px;">1 − CER，越高越好。</td></tr>
      <tr style="border-bottom:1px solid #f0f2f5;"><td style="padding:6px 12px;font-weight:600;">SER</td><td style="padding:6px 12px;">句错率 (Sentence Error Rate)</td><td style="padding:6px 12px;">CER &gt; 1% 的句子占总句子数的比例，越低越好。</td></tr>
      <tr style="border-bottom:1px solid #f0f2f5;"><td style="padding:6px 12px;font-weight:600;">Sub 率</td><td style="padding:6px 12px;">替换错误率</td><td style="padding:6px 12px;">被替换字符数 / 参考文本总字数。ASR 将正确字符识别为其他字符的比例。</td></tr>
      <tr style="border-bottom:1px solid #f0f2f5;"><td style="padding:6px 12px;font-weight:600;">Del 率</td><td style="padding:6px 12px;">删除错误率</td><td style="padding:6px 12px;">被删除字符数 / 参考文本总字数。ASR 漏识别字符的比例。</td></tr>
      <tr style="border-bottom:1px solid #f0f2f5;"><td style="padding:6px 12px;font-weight:600;">Ins 率</td><td style="padding:6px 12px;">插入错误率</td><td style="padding:6px 12px;">插入多余字符数 / 参考文本总字数。ASR 幻觉产生多余字符的比例。</td></tr>
      <tr><td style="padding:6px 12px;font-weight:600;">Hits / RefLen</td><td style="padding:6px 12px;">正确字符数 / 参考总长度</td><td style="padding:6px 12px;">用于交叉验证 CER 计算的正确性。</td></tr>
    </tbody>
  </table>

  <h3 style="font-size:15px;margin-bottom:12px;color:#333;">实时性指标</h3>
  <table style="width:100%;border-collapse:collapse;font-size:13px;">
    <thead><tr style="background:#f0f2f5;"><th style="padding:8px 12px;text-align:left;">指标</th><th style="padding:8px 12px;text-align:left;">含义</th><th style="padding:8px 12px;text-align:left;">说明</th></tr></thead>
    <tbody>
      <tr style="border-bottom:1px solid #f0f2f5;"><td style="padding:6px 12px;font-weight:600;">RTF</td><td style="padding:6px 12px;">实时率 (Real-Time Factor)</td><td style="padding:6px 12px;">处理时长 / 音频时长，越低越好。RTF &lt; 1 表示推理速度快于实时。</td></tr>
      <tr style="border-bottom:1px solid #f0f2f5;"><td style="padding:6px 12px;font-weight:600;">TTFC</td><td style="padding:6px 12px;">首字延迟 (Time To First Chunk)</td><td style="padding:6px 12px;">从输入第一个音频 chunk 到收到首个 ASR 文本的时间，越低越好。影响对话流畅度。</td></tr>
      <tr style="border-bottom:1px solid #f0f2f5;"><td style="padding:6px 12px;font-weight:600;">P50 / P90 / P95 / P99 TTFC</td><td style="padding:6px 12px;">TTFC 百分位数</td><td style="padding:6px 12px;">TTFC 的分布情况。P95 表示 95% 的句子首字延迟低于此值，用于评估长尾延迟。</td></tr>
      <tr><td style="padding:6px 12px;font-weight:600;">E2E Latency</td><td style="padding:6px 12px;">端到端延迟</td><td style="padding:6px 12px;">从输入首个音频 chunk 到收到最终完整文本的时间，越低越好。</td></tr>
    </tbody>
  </table>
</div>"""

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ASR 评测对比报告</title>
<style>{CSS}</style>
</head>
<body>

<h1>ASR 评测对比报告</h1>
<p class="subtitle">共 {n} 个模型 · 生成时间 {gen_time}</p>

<h2>指标总览</h2>
<div class="summary-grid {col_class}">
{cards}
</div>

{hw_table}

{cer_table}

{timing_table}

{config_table}

{metrics_legend}

<div class="footer">Generated by compare.py</div>

</body>
</html>"""


# ── 子表生成 ──

def _build_hardware_table(files: list[Path], data_list: list[dict]) -> str:
    rows = ""
    for i, (fp, d) in enumerate(zip(files, data_list)):
        meta = d.get("meta", {})
        hw = d.get("hardware", {})
        rows += f"""
    <tr>
      <td style="color:{file_color(i)};font-weight:600;">{meta.get('model_name', fp.stem)}</td>
      <td>{hw.get('cpu_model', '-')}</td>
      <td>{hw.get('cpu_cores_physical', '-')}P/{hw.get('cpu_cores_logical', '-')}L</td>
      <td>{_fmt(hw.get('memory_total_gb'), 1)} GB</td>
      <td>{hw.get('python_version', '-')}</td>
      <td>{hw.get('os', '-')}</td>
    </tr>"""

    return f"""
<h2>硬件环境</h2>
<table>
<thead><tr><th>模型</th><th>CPU</th><th>Cores (P/L)</th><th>内存</th><th>Python</th><th>OS</th></tr></thead>
<tbody>{rows}</tbody>
</table>"""


def _build_cer_table(files: list[Path], data_list: list[dict]) -> str:
    # 收集所有数据集名称
    all_datasets: list[str] = []
    for d in data_list:
        for r in d.get("results", []):
            name = r["dataset_name"]
            if name not in all_datasets:
                all_datasets.append(name)

    rows = ""
    for ds in all_datasets:
        rows += f'<tr><td colspan="10" style="background:#fafafa;font-weight:600;">{ds}</td></tr>'
        for i, (fp, d) in enumerate(zip(files, data_list)):
            # 找到对应数据集
            ds_result = next((r for r in d.get("results", []) if r["dataset_name"] == ds), None)
            if not ds_result:
                continue
            cer = ds_result.get("aggregate_cer", {})
            name = d["meta"]["model_name"]
            rows += f"""
    <tr>
      <td style="color:{file_color(i)};font-weight:600;">{name}</td>
      <td>{_fmt_pct(cer.get('cer'))}</td>
      <td>{_fmt_pct(cer.get('accuracy'))}</td>
      <td>{_fmt_pct(cer.get('ser'))}</td>
      <td>{_fmt_pct(cer.get('sub_rate'))}</td>
      <td>{_fmt_pct(cer.get('del_rate'))}</td>
      <td>{_fmt_pct(cer.get('ins_rate'))}</td>
      <td>{cer.get('total_hits', '-')}</td>
      <td>{cer.get('total_reference_length', '-')}</td>
      <td>{ds_result.get('num_evaluated', '-')}</td>
    </tr>"""

    return f"""
<h2>CER 详细对比</h2>
<table>
<thead><tr><th>模型</th><th>CER</th><th>Accuracy</th><th>SER</th><th>Sub</th><th>Del</th><th>Ins</th><th>Hits</th><th>RefLen</th><th>句数</th></tr></thead>
<tbody>{rows}</tbody>
</table>"""


def _build_timing_table(files: list[Path], data_list: list[dict]) -> str:
    """生成实时性对比表，每个模型显示合并后的跨数据集指标。"""
    rows = ""
    for i, (fp, d) in enumerate(zip(files, data_list)):
        meta = d["meta"]
        results = d.get("results", [])
        if not results:
            continue
        # 使用合并后的指标
        timing = _merge_timing(results)
        name = meta["model_name"]
        rows += f"""
    <tr>
      <td style="color:{file_color(i)};font-weight:600;">{name}</td>
      <td>{_fmt(timing.get('avg_rtf'), 4)}</td>
      <td>{_fmt(timing.get('avg_ttfc_s'), 3)}s</td>
      <td>{_fmt(timing.get('p50_ttfc_s'), 3)}s</td>
      <td>{_fmt(timing.get('p90_ttfc_s'), 3)}s</td>
      <td>{_fmt(timing.get('p95_ttfc_s'), 3)}s</td>
      <td>{_fmt(timing.get('p99_ttfc_s'), 3)}s</td>
      <td>{_fmt(timing.get('avg_e2e_latency_s'), 3)}s</td>
      <td>{_fmt(timing.get('total_audio_duration_s'), 1)}s</td>
      <td>{_fmt(timing.get('total_processing_time_s'), 1)}s</td>
    </tr>"""

    return f"""
<h2>实时性对比</h2>
<table>
<thead><tr><th>模型</th><th>Avg RTF</th><th>Avg TTFC</th><th>P50 TTFC</th><th>P90 TTFC</th><th>P95 TTFC</th><th>P99 TTFC</th><th>Avg E2E</th><th>总音频时长</th><th>总处理时长</th></tr></thead>
<tbody>{rows}</tbody>
</table>"""


def _build_config_table(files: list[Path], data_list: list[dict]) -> str:
    rows = ""
    for i, (fp, d) in enumerate(zip(files, data_list)):
        meta = d["meta"]
        cfg = d.get("config", {})
        name = meta["model_name"]
        rows += f"""
    <tr>
      <td style="color:{file_color(i)};font-weight:600;">{name}</td>
      <td>{meta.get('model_type', '-')}</td>
      <td>{cfg.get('vad_threshold', '-')}</td>
      <td>{cfg.get('vad_min_speech_ms', '-')}ms</td>
      <td>{cfg.get('vad_min_silence_ms', '-')}ms</td>
      <td>{cfg.get('max_hours') or '不限'}</td>
      <td>{cfg.get('max_utterances') or '不限'}</td>
    </tr>"""

    return f"""
<h2>评测配置</h2>
<table>
<thead><tr><th>模型</th><th>类型</th><th>VAD阈值</th><th>最小语音(ms)</th><th>最小静音(ms)</th><th>Max Hours</th><th>Max UTT</th></tr></thead>
<tbody>{rows}</tbody>
</table>"""


# ── 格式化 ──

def _fmt_pct(v) -> str:
    if v is None:
        return "-"
    return f"{v * 100:.2f}%"


def _fmt(v, precision=3) -> str:
    if v is None:
        return "-"
    return f"{v:.{precision}f}"


# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="对比 ASR 评测结果，生成 HTML 报告")
    parser.add_argument(
        "--glob", default=DEFAULT_GLOB,
        help=f"在 results/ 下匹配 JSON 文件的 glob 模式 (默认: {DEFAULT_GLOB})",
    )
    parser.add_argument(
        "--output", "-o", default=DEFAULT_OUTPUT,
        help=f"输出 HTML 路径 (默认: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--open", action="store_true",
        help="生成后自动在浏览器中打开",
    )
    args = parser.parse_args()

    # 解析 glob
    files = discover_files(args.glob)
    if not files:
        print("No files to compare. Exiting.")
        return 1

    print(f"Found {len(files)} file(s):")
    for fp in files:
        print(f"  {fp}")

    # 加载
    data_list = [load_result(fp) for fp in files]

    # 生成 HTML
    output_path = THIS_DIR / args.output
    html = build_html(files, data_list)
    output_path.write_text(html, encoding="utf-8")
    print(f"\nReport saved to: {output_path}")

    # 自动打开
    if args.open:
        import webbrowser
        webbrowser.open(str(output_path))

    return 0


if __name__ == "__main__":
    sys.exit(main())
