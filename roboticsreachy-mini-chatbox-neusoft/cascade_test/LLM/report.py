"""Report generation for LLM evaluation results.

Generates structured reports per evaluation doc section 5:
1. Per-category raw results
2. Per-metric aggregated results
3. Final weighted score
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import asdict

from .framework import (
    CaseResult,
    CategoryReport,
    CATEGORIES,
    rate_ttft,
    rate_tokens_per_sec,
    SCORE_WEIGHTS,
)
from .scoring import compute_final_score


def generate_report(
    category_reports: Dict[str, CategoryReport],
    provider_name: str,
    model_name: str = "",
    output_dir: Optional[Path] = None,
) -> str:
    """Generate a full evaluation report.

    Args:
        category_reports: Mapping of category name -> CategoryReport.
        provider_name: Name of the LLM provider tested.
        model_name: Specific model name.
        output_dir: Directory to write report files. If None, returns string only.

    Returns:
        Markdown-formatted report string.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    final = compute_final_score(category_reports)

    lines: List[str] = []
    lines.append(f"# LLM 评测报告")
    lines.append("")
    lines.append(f"- **评测时间**: {timestamp}")
    lines.append(f"- **LLM 提供者**: {provider_name}")
    lines.append(f"- **模型名称**: {model_name or 'N/A'}")
    lines.append("")

    # --- Section 1: Per-category results ---
    lines.append("## 1. 各样本组原始结果")
    lines.append("")

    for cat in CATEGORIES:
        report = category_reports.get(cat)
        if not report:
            lines.append(f"### {cat}")
        lines.append(f"### {cat}")
        lines.append("")
        lines.append(f"| 指标 | 值 |")
        lines.append(f"|---|---|")
        lines.append(f"| 总用例数 | {report.total_cases if report else 0} |")
        lines.append(f"| 通过用例数 | {report.passed_cases if report else 0} |")
        lines.append(f"| 通过率 | {_pct(report.passed_cases, report.total_cases) if report else 'N/A'} |")

        if report and report.avg_ttft_ms is not None:
            lines.append(f"| 平均 TTFT | {report.avg_ttft_ms}ms ({rate_ttft(report.avg_ttft_ms)}) |")
        if report and report.avg_tokens_per_sec is not None:
            lines.append(f"| 平均 Tokens/sec | {report.avg_tokens_per_sec} ({rate_tokens_per_sec(report.avg_tokens_per_sec)}) |")
        if report and report.hallucination_rate is not None:
            lines.append(f"| 幻觉率 | {report.hallucination_rate:.1%} |")
        if report and report.tool_call_accuracy is not None:
            lines.append(f"| 工具调用正确率 | {report.tool_call_accuracy:.1%} |")
        if report and report.multi_turn_recall_rate is not None:
            lines.append(f"| 多轮回忆正确率 | {report.multi_turn_recall_rate:.1%} |")
        if report and report.memory_summary_completeness is not None:
            lines.append(f"| 记忆总结完整率 | {report.memory_summary_completeness:.1%} |")
        if report and report.asr_robustness_rate is not None:
            lines.append(f"| ASR错字容错率 | {report.asr_robustness_rate:.1%} |")
        if report and report.avg_oral_style_score is not None:
            lines.append(f"| 口语化评分 | {report.avg_oral_style_score:.2f} |")
        if report and report.avg_safety_score is not None:
            lines.append(f"| 安全适配评分 | {report.avg_safety_score:.2f} |")
        lines.append("")

        # Individual case details
        if report and report.case_results:
            lines.append("**用例详情**:")
            lines.append("")
            lines.append("| case_id | 通过 | TTFT(ms) | 关键点命中 | 错误 |")
            lines.append("|---|---|---|---|---|")
            for cr in report.case_results:
                hit_count = len(cr.expected_points_hit)
                total_points = len(cr.expected_points_hit) + len(cr.expected_points_miss)
                points_str = f"{hit_count}/{total_points}" if total_points > 0 else "-"
                error_str = cr.error[:30] if cr.error else ""
                lines.append(
                    f"| {cr.case_id} | {'✓' if cr.passed else '✗'} | "
                    f"{cr.speed.ttft_ms or '-'} | {points_str} | {error_str} |"
                )
            lines.append("")

    # --- Section 2: Per-metric aggregated results ---
    lines.append("## 2. 各指标汇总结果")
    lines.append("")
    metrics = final["metrics"]
    lines.append("| 指标 | 原始值 | 归一化得分 | 适用样本组 |")
    lines.append("|---|---|---|---|")

    metric_applicability = {
        "TTFT": "所有样本组",
        "Tokens/sec": "所有样本组",
        "Hallucination": "事实问答、任务执行",
        "Tool call accuracy": "工具调用",
        "Multi-turn recall": "多轮记忆",
        "Memory summary": "记忆总结",
        "ASR robustness": "ASR错字输入",
        "Oral style": "事实问答、任务执行、多轮记忆、ASR错字输入、陪伴安抚",
        "Safety": "陪伴安抚、安全风险",
    }

    for metric_name, metric_data in metrics.items():
        value_str = str(metric_data["value"]) if "value" in metric_data else str(metric_data.get("value_ms", ""))
        if metric_name == "TTFT":
            value_str = f"{metric_data['value_ms']}ms"
        lines.append(
            f"| {metric_name} | {value_str} | {metric_data['score']:.4f} | "
            f"{metric_applicability.get(metric_name, '')} |"
        )
    lines.append("")

    # --- Section 3: Final weighted score ---
    lines.append("## 3. 最终加权评分")
    lines.append("")
    lines.append("| 指标 | 权重 | 得分 | 加权得分 |")
    lines.append("|---|---|---|---|")

    for metric_name, metric_data in metrics.items():
        weight = SCORE_WEIGHTS.get(_metric_to_weight_key(metric_name), 0)
        weighted = weight * metric_data["score"]
        lines.append(f"| {metric_name} | {weight:.0%} | {metric_data['score']:.4f} | {weighted:.4f} |")

    lines.append("")
    lines.append(f"**最终评分: {final['final_score']:.4f}** (满分 1.0)")
    lines.append("")

    # Grade
    score = final["final_score"]
    if score >= 0.85:
        grade = "优秀"
    elif score >= 0.70:
        grade = "良好"
    elif score >= 0.50:
        grade = "可接受"
    else:
        grade = "偏差"
    lines.append(f"**等级: {grade}**")
    lines.append("")

    report_text = "\n".join(lines)

    # Write to file if output_dir specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Markdown report
        report_file = output_dir / f"llm_eval_{provider_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        report_file.write_text(report_text, encoding="utf-8")

        # JSON data
        json_data = {
            "timestamp": timestamp,
            "provider": provider_name,
            "model": model_name,
            "final_score": final,
            "categories": {cat: asdict(r) for cat, r in category_reports.items()},
        }
        json_file = output_dir / f"llm_eval_{provider_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        json_file.write_text(json.dumps(json_data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

        logger_path_msg = f"Report written to {report_file} and {json_file}"

    return report_text


def _pct(numerator: int, denominator: int) -> str:
    """Format a percentage."""
    return f"{numerator / denominator:.1%}" if denominator > 0 else "N/A"


def _metric_to_weight_key(metric_name: str) -> str:
    """Map metric display name to SCORE_WEIGHTS key."""
    mapping = {
        "TTFT": "ttft",
        "Tokens/sec": "tokens_per_sec",
        "Hallucination": "hallucination",
        "Tool call accuracy": "tool_call_accuracy",
        "Multi-turn recall": "multi_turn_recall",
        "Memory summary": "memory_summary",
        "ASR robustness": "asr_robustness",
        "Oral style": "oral_style",
        "Safety": "safety",
    }
    return mapping.get(metric_name, "")
