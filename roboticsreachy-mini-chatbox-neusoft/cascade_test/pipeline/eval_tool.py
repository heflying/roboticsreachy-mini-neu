"""多模型回复人工评分工具。

从多个模型产出的 output 目录加载结果，逐题对比展示回复，人工打分后按维度计算
平均分排名。支持盲评切换和断点续评。

Usage:
    cd project_root

    # 对比两个模型
    python cascade_test/pipeline/eval_tool.py \
        --models qwen cascade_test/pipeline/output-qwen \
        --models gemma cascade_test/pipeline/output-gemma \
        --config cascade_test/pipeline/eval_dimensions.yaml \
        --output cascade_test/pipeline/eval_results.json

    # 断点续评（指定已有结果文件）
    python cascade_test/pipeline/eval_tool.py \
        --models qwen cascade_test/pipeline/output-qwen \
        --models gemma cascade_test/pipeline/output-gemma \
        --output cascade_test/pipeline/eval_results.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import gradio as gr
import yaml

# Ensure project root on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

MAX_DIMS = 10


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------


def load_dimensions(config_path: Path) -> list[dict[str, Any]]:
    """从 YAML 加载评分维度配置。"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    dims = config.get("dimensions", [])
    if not dims:
        raise ValueError("配置文件中没有定义 dimensions")
    return dims


def load_model_data(model_dir: Path) -> dict[str, dict[str, Any]]:
    """直接扫描输出目录，不依赖 summary.json。

    目录结构：
        model_dir/分类/子分类/000001_transcript.txt
        model_dir/分类/子分类/000001_response.txt
        model_dir/分类/子分类/000001_audio.wav

    Returns:
        {rel_path: {transcript, response, audio_path}}
        rel_path 形式如 "回复幻觉率/冲突信息幻觉/000001.wav"
    """
    data: dict[str, dict[str, Any]] = {}
    if not model_dir.is_dir():
        raise FileNotFoundError(f"目录不存在: {model_dir}")

    for transcript_file in sorted(model_dir.rglob("*_transcript.txt")):
        stem = transcript_file.name.replace("_transcript.txt", "")  # "000001"
        response_file = transcript_file.with_name(f"{stem}_response.txt")
        audio_file = transcript_file.with_name(f"{stem}_audio.wav")

        rel_dir = transcript_file.parent.relative_to(model_dir)
        # 构造与输入 wav 同名的 rel_path，保持与输入数据对齐
        rel_path = str(rel_dir / f"{stem}.wav").replace("\\", "/")

        # 读取 transcript（必读，否则跳过）
        try:
            transcript = transcript_file.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if not transcript:
            continue

        # 读取 response（可选）
        response = ""
        if response_file.exists():
            try:
                response = response_file.read_text(encoding="utf-8").strip()
            except Exception:
                pass

        audio_path = str(audio_file) if audio_file.exists() else ""

        data[rel_path] = {
            "transcript": transcript,
            "response": response,
            "audio_path": audio_path,
        }
    return data


def load_input_audio_map(input_dir: Path) -> dict[str, str]:
    """扫描输入目录，建立 rel_path → 音频文件路径的映射。

    目录结构:
        input_dir/分类/子分类/xxx.wav

    Returns:
        {rel_path: audio_file_path}
        rel_path 形式如 "真实老年人对话/Elderly0014S0046W0003.wav"
    """
    audio_map: dict[str, str] = {}
    if not input_dir.is_dir():
        print(f"WARNING: 输入目录不存在: {input_dir}")
        return audio_map

    for wav_file in sorted(input_dir.rglob("*.wav")):
        rel_path = str(wav_file.relative_to(input_dir)).replace("\\", "/")
        audio_map[rel_path] = str(wav_file)

    return audio_map


def match_questions(models_data: list[tuple[str, dict[str, dict[str, Any]]]]) -> list[dict[str, Any]]:
    """按 rel_path 对齐多模型数据，只保留所有模型都有结果的题目。

    Returns:
        [{rel_path, category, transcript, models: {name: {transcript, response, audio_path}}}]
    """
    if not models_data:
        return []

    # 找所有模型共有的 rel_path
    common_keys = set(models_data[0][1].keys())
    for _, data in models_data[1:]:
        common_keys &= set(data.keys())

    common_keys = sorted(common_keys)

    questions = []
    for rel in common_keys:
        # 从第一个模型的对应 rel 提取 transcript（所有模型 transcript 应一致）
        first_model_data = models_data[0][1][rel]
        # 从 rel_path 提取分类（如 "回复幻觉率/冲突信息幻觉/000001.wav"）
        parts = Path(rel).parts
        category = "/".join(parts[:-1]) if len(parts) > 1 else ""

        models: dict[str, dict[str, Any]] = {}
        for name, data in models_data:
            d = data[rel]
            models[name] = {
                "transcript": d["transcript"],
                "response": d["response"],
                "audio_path": d["audio_path"],
            }

        questions.append({
            "rel_path": rel,
            "category": category,
            "transcript": first_model_data["transcript"],
            "models": models,
        })

    return questions


def load_results(path: Path) -> dict[str, Any]:
    """加载已有评分结果文件。"""
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_results(path: Path, data: dict[str, Any]) -> None:
    """保存评分结果。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Gradio 界面
# ---------------------------------------------------------------------------


def _model_label(i: int, name: str, blind: bool) -> str:
    if blind:
        return f"模型 {chr(65 + i)}"
    return name


def build_app(
    questions: list[dict[str, Any]],
    dimensions: list[dict[str, Any]],
    model_names: list[str],
    output_path: Path,
    existing_scores: dict[str, Any],
    input_audio_map: dict[str, str],
) -> gr.Blocks:
    """构建 Gradio 评分界面。"""
    num_questions = len(questions)
    if num_questions == 0:
        raise ValueError("没有可评分的题目（所有模型共有的 rel_path 为空）")
    num_models = len(model_names)
    if num_models == 0:
        raise ValueError("至少需要一个模型")

    num_dims = len(dimensions)

    # 确定从哪个题目开始
    current_i = existing_scores.get("_current_index", 0)
    current_i = max(0, min(current_i, num_questions - 1))

    # 转换已有评分为内部格式 {rel_path: {model: {dim: score}}}
    saved_scores: dict[str, dict[str, dict[str, int]]] = {}
    for rel, model_scores in existing_scores.items():
        if rel.startswith("_"):
            continue
        if isinstance(model_scores, dict):
            saved_scores[rel] = {}
            for mname, dim_scores in model_scores.items():
                if isinstance(dim_scores, dict):
                    saved_scores[rel][mname] = dim_scores

    css = """
    .eval-header { text-align: center; margin-bottom: 10px; }
    .question-box { background: #f5f5f5; padding: 15px; border-radius: 8px; margin-bottom: 10px; }
    .model-col { border: 1px solid #ddd; border-radius: 8px; padding: 10px; margin: 5px; }
    .dim-row { margin: 5px 0; }
    .audio-box { margin-top: 8px; }
    footer { visibility: hidden; }
    """

    with gr.Blocks(css=css, title="LLM 回复人工评分") as app:
        # ── 状态 ──
        current_idx = gr.State(current_i)
        blind_mode = gr.State(False)
        scores_state = gr.State(saved_scores)

        # ── 顶部信息栏 ──
        with gr.Row(elem_classes="eval-header"):
            progress_label = gr.Markdown("", elem_id="progress_label")
            blind_toggle = gr.Checkbox(label="盲评模式", value=False)
            save_status = gr.Markdown("", elem_id="save_status")

        # ── 原始输入音频 ──
        input_audio = gr.Audio(label="原始输入音频", type="filepath", visible=False, interactive=False)

        # ── 模型回复区（预创建 num_models 列）──
        model_name_labels: list[gr.Markdown] = []
        model_transcript_boxes: list[gr.Textbox] = []
        model_response_boxes: list[gr.Textbox] = []
        model_audio_players: list[gr.Audio] = []

        with gr.Row():
            for mi in range(num_models):
                model_name_labels.append(gr.Markdown("", visible=False))
        with gr.Row():
            for mi in range(num_models):
                model_transcript_boxes.append(
                    gr.Textbox(label="ASR转录", lines=4, interactive=False, visible=False)
                )
        with gr.Row():
            for mi in range(num_models):
                model_response_boxes.append(
                    gr.Textbox(label="回复", lines=6, interactive=False, visible=False)
                )
        with gr.Row():
            for mi in range(num_models):
                model_audio_players.append(
                    gr.Audio(label="语音", type="filepath", visible=False, elem_classes="audio-box", interactive=False)
                )

        # ── 评分表：预创建 MAX_DIMS 行 × num_models 滑块 ──
        scoring_md = gr.Markdown("### 评分")
        dim_name_mds: list[gr.Markdown] = []
        dim_sliders: list[list[gr.Slider]] = []

        for di in range(MAX_DIMS):
            dim_name_mds.append(gr.Markdown("", visible=False))
            with gr.Row():
                row_sliders: list[gr.Slider] = []
                for mi in range(num_models):
                    s = gr.Slider(1, 10, value=5, step=1, label="", visible=False, interactive=True)
                    row_sliders.append(s)
                dim_sliders.append(row_sliders)

        # ── 导航按钮 ──
        with gr.Row():
            prev_btn = gr.Button("← 上一题")
            next_btn = gr.Button("保存并下一题 →", variant="primary")
            finish_btn = gr.Button("生成最终报告", variant="secondary")

        # ── 排名区 ──
        ranking_md = gr.Markdown("")

        # ── 最终报告区 ──
        final_report_md = gr.Markdown("", visible=False)

        # 所有滑块的扁平列表
        all_sliders: list[gr.Slider] = [s for row in dim_sliders for s in row]

        # ── 渲染函数 ──
        def render_question(
            i: int,
            blind: bool,
            scores: dict[str, dict[str, dict[str, int]]],
        ) -> list[Any]:
            """根据当前索引返回所有组件的更新值。"""
            if not questions or i < 0 or i >= len(questions):
                return [gr.update(value="无题目")] * len(all_outputs)
            q = questions[i]
            dims = dimensions

            # 题目信息
            progress = f"## 题目 {i + 1} / {num_questions}  &nbsp;|&nbsp; 分类：{q.get('category', '未分类')}"
            input_audio_path = input_audio_map.get(q["rel_path"], "")
            print(f"[DEBUG] render_question(i={i}, category={q.get('category')}, rel_path={q['rel_path']})")

            # 生成排名
            model_avgs = compute_rankings(scores, dims)
            ranking_parts = ["### 当前排名"]
            for rank_idx, (mname, avg, dim_avgs) in enumerate(model_avgs, 1):
                try:
                    mlabel = _model_label(model_names.index(mname), mname, blind)
                except ValueError:
                    mlabel = mname
                rated_count = sum(
                    1
                    for rel_sc in scores.values()
                    if mname in rel_sc and all(v > 0 for v in rel_sc[mname].values())
                )
                # 总分
                ranking_parts.append(f"{rank_idx}. **{mlabel}**：总分平均 {avg:.1f} 分（已评 {rated_count} 题）")
                # 各维度平均分
                dim_parts = []
                for d in dims:
                    dname = d["name"]
                    davg = dim_avgs.get(dname, 0.0)
                    dim_parts.append(f"{dname}: {davg:.1f}")
                ranking_parts.append(f"   > {' | '.join(dim_parts)}")
            ranking_text = "\n\n".join(ranking_parts)

            results: list[Any] = [gr.update(value=progress), gr.update(value=input_audio_path, visible=bool(input_audio_path)), gr.update(value=ranking_text)]

            # 模型列更新 — 顺序必须与 all_outputs 一致:
            # model_name_labels → model_transcript_boxes → model_response_boxes → model_audio_players
            for mi in range(num_models):
                if mi < num_models:
                    mname = model_names[mi]
                    label = _model_label(mi, mname, blind)
                    results.append(gr.update(value=f"### {label}", visible=True))
                else:
                    results.append(gr.update(visible=False))

            models_dict = q.get("models", {}) if isinstance(q.get("models"), dict) else {}
            for mi in range(num_models):
                if mi < num_models:
                    mname = model_names[mi]
                    md = models_dict.get(mname, {})
                    asr_text = md.get("transcript", "(无转录)") if isinstance(md, dict) else "(无转录)"
                    results.append(gr.update(value=asr_text, visible=True))
                else:
                    results.append(gr.update(visible=False))

            for mi in range(num_models):
                if mi < num_models:
                    mname = model_names[mi]
                    md = models_dict.get(mname, {})
                    response_text = md.get("response", "(无回复)") if isinstance(md, dict) else "(无回复)"
                    results.append(gr.update(value=response_text, visible=True))
                else:
                    results.append(gr.update(visible=False))

            for mi in range(num_models):
                if mi < num_models:
                    mname = model_names[mi]
                    md = models_dict.get(mname, {})
                    audio_file = md.get("audio_path") or "" if isinstance(md, dict) else ""
                    results.append(gr.update(value=audio_file, visible=True))
                else:
                    results.append(gr.update(visible=False))

            # 维度名可见性 + 内容
            for di in range(MAX_DIMS):
                if di < num_dims:
                    results.append(gr.update(value=f"**{dims[di]['name']}**", visible=True))
                else:
                    results.append(gr.update(visible=False))

            # 滑块值 + 可见性 — 必须与 all_sliders 数量严格一致（MAX_DIMS × num_models）
            for di in range(MAX_DIMS):
                for mi in range(num_models):
                    if di < num_dims and mi < num_models:
                        mname = model_names[mi]
                        dim_name = dims[di]["name"]
                        prev_score = scores.get(q["rel_path"], {}).get(mname, {}).get(dim_name, 5)
                        results.append(gr.update(value=prev_score, visible=True))
                    else:
                        results.append(gr.update(visible=False))

            # 上一题按钮状态
            results.append(gr.update(interactive=i > 0))

            return results

        # ── 统一输出列表 ──
        all_outputs = (
            [progress_label, input_audio, ranking_md]
            + model_name_labels + model_transcript_boxes + model_response_boxes + model_audio_players
            + dim_name_mds
            + all_sliders
            + [prev_btn]
        )

        def on_navigate(i, blind, scores):
            return render_question(i, blind, scores)

        # 初始渲染
        app.load(
            fn=on_navigate,
            inputs=[current_idx, blind_mode, scores_state],
            outputs=all_outputs,
        )

        # ── 翻页事件 ──
        def go_prev(i):
            return max(0, i - 1)

        prev_btn.click(
            fn=go_prev,
            inputs=[current_idx],
            outputs=[current_idx],
        ).then(
            fn=on_navigate,
            inputs=[current_idx, blind_mode, scores_state],
            outputs=all_outputs,
        )

        # ── 保存并下一题 ──
        def save_and_next(*args: Any) -> list[Any]:
            """收集所有滑块值，保存评分，前进到下一题。"""
            # args: slider_vals ... + current_idx + scores_state + blind_mode
            # all_sliders 的长度为 MAX_DIMS * num_models，需按此解析
            n_sliders = MAX_DIMS * num_models
            slider_vals = list(args[:n_sliders])
            i = int(args[n_sliders])
            old_scores = args[n_sliders + 1]
            blind = bool(args[n_sliders + 2])
            print(f"[DEBUG] save_and_next: i={i}, blind={blind}, n_args={len(args)}")

            q = questions[i]
            new_scores = copy.deepcopy(old_scores)

            if q["rel_path"] not in new_scores:
                new_scores[q["rel_path"]] = {}

            idx = 0
            for di in range(num_dims):
                dim_name = dimensions[di]["name"]
                for mi in range(num_models):
                    mname = model_names[mi]
                    if mname not in new_scores[q["rel_path"]]:
                        new_scores[q["rel_path"]][mname] = {}
                    new_scores[q["rel_path"]][mname][dim_name] = int(slider_vals[idx])
                    idx += 1

            # 保存到文件
            output_data: dict[str, Any] = {}
            for k, v in new_scores.items():
                output_data[k] = v
            output_data["_current_index"] = i + 1
            output_data["_models"] = model_names
            output_data["_dimensions"] = [
                {"name": d["name"], "weight": d.get("weight", 1.0)} for d in dimensions
            ]
            save_results(output_path, output_data)

            next_i = min(i + 1, num_questions - 1)
            status = f"✅ 已保存（题目 {i + 1}/{num_questions}）"
            print(f"[DEBUG] save_and_next: next_i={next_i}, num_questions={num_questions}")

            render_results = render_question(next_i, blind, new_scores)
            print(f"[DEBUG] save_and_next: returning {3 + len(render_results)} values")
            return [next_i, new_scores, status] + render_results

        next_btn.click(
            fn=save_and_next,
            inputs=all_sliders + [current_idx, scores_state, blind_mode],
            outputs=[current_idx, scores_state, save_status] + all_outputs,
        )

        # ── 盲评切换 ──
        def toggle_blind(blind: bool) -> bool:
            return not blind

        blind_toggle.change(
            fn=toggle_blind,
            inputs=[blind_mode],
            outputs=[blind_mode],
        ).then(
            fn=on_navigate,
            inputs=[current_idx, blind_mode, scores_state],
            outputs=all_outputs,
        )

        # ── 生成最终报告 ──
        def generate_final_report(
            scores: dict[str, dict[str, dict[str, int]]],
            blind: bool,
        ) -> tuple[Any, str]:
            """生成最终评分报告，保存到 JSON 文件。"""
            rankings = compute_rankings(scores, dimensions)

            # 构建 Markdown 报告
            lines = ["# LLM 评测最终报告\n"]
            lines.append(f"**模型数量**: {len(rankings)}")
            lines.append(f"**评测题目数**: {num_questions}")
            lines.append(f"**评分维度**: {', '.join(d['name'] for d in dimensions)}\n")
            lines.append("## 排名总览\n")

            for rank_idx, (mname, avg, dim_avgs) in enumerate(rankings, 1):
                try:
                    mlabel = _model_label(model_names.index(mname), mname, blind)
                except ValueError:
                    mlabel = mname
                lines.append(f"{rank_idx}. **{mlabel}** — 总分: **{avg:.1f}**")

                # 各维度得分
                for d in dimensions:
                    dname = d["name"]
                    davg = dim_avgs.get(dname, 0.0)
                    lines.append(f"   - {dname}: {davg:.1f}")
                lines.append("")

            report_md = "\n".join(lines)

            # 保存为 JSON
            summary = {
                "models": model_names,
                "dimensions": [{"name": d["name"], "weight": d.get("weight", 1.0)} for d in dimensions],
                "num_questions": num_questions,
                "rankings": [
                    {
                        "rank": rank_idx,
                        "model": mname,
                        "total_avg": avg,
                        "dim_avgs": dim_avgs,
                    }
                    for rank_idx, (mname, avg, dim_avgs) in enumerate(rankings, 1)
                ],
            }
            summary_path = output_path.parent / f"{output_path.stem}_summary.json"
            save_results(summary_path, summary)

            status_msg = f"✅ 报告已生成并保存至: {summary_path}"
            # 同时更新内容和可见性
            return gr.update(value=report_md, visible=True), status_msg

        finish_btn.click(
            fn=generate_final_report,
            inputs=[scores_state, blind_mode],
            outputs=[final_report_md, save_status],
        )

    return app


def compute_rankings(
    scores: dict[str, dict[str, dict[str, int]]],
    dimensions: list[dict[str, Any]],
) -> list[tuple[str, float, dict[str, float]]]:
    """从评分数据计算各模型加权总分和各维度平均分，按总分降序排列。

    加权总分计算方式：每题先算加权分，再跨题平均。

    Returns:
        [(model_name, weighted_total_avg, {dim_name: dim_avg, ...}), ...]
    """
    if not scores:
        return []

    dim_names = [d["name"] for d in dimensions]
    dim_weights = {d["name"]: d.get("weight", 1.0) for d in dimensions}

    # 按模型收集数据
    # per_model: {mname: {"dim_avgs": {dim: [scores]}, "per_question_weighted": [weighted_per_q]}}
    per_model: dict[str, dict[str, Any]] = {}

    for rel, model_scores in scores.items():
        for mname, dim_scores in model_scores.items():
            if mname not in per_model:
                per_model[mname] = {"dim_scores": {d: [] for d in dim_names}, "q_weighted": []}

            # 收集各维度得分
            for dim_name in dim_names:
                score = dim_scores.get(dim_name, 0)
                per_model[mname]["dim_scores"][dim_name].append(score)

            # 计算本题加权分
            total_s = 0.0
            total_w = 0.0
            for dim_name in dim_names:
                w = dim_weights.get(dim_name, 1.0)
                v = dim_scores.get(dim_name, 0)
                total_s += v * w
                total_w += w
            if total_w > 0:
                per_model[mname]["q_weighted"].append(total_s / total_w)

    # 计算总分和各维度平均分
    rankings = []
    for mname, data in per_model.items():
        # 各维度平均分
        dim_avgs = {}
        for dim_name in dim_names:
            vals = data["dim_scores"][dim_name]
            dim_avgs[dim_name] = sum(vals) / len(vals) if vals else 0.0

        # 加权总分 = 各题加权分的平均
        qw = data["q_weighted"]
        weighted_avg = sum(qw) / len(qw) if qw else 0.0

        rankings.append((mname, weighted_avg, dim_avgs))

    rankings.sort(key=lambda x: x[1], reverse=True)
    return rankings


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="多模型回复人工评分工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例:
  python cascade_test/pipeline/eval_tool.py \\
      --models qwen output/ \\
      --models gemma output-gemma/ \\
      --output results.json

  # 断点续评
  python cascade_test/pipeline/eval_tool.py \\
      --models qwen output/ \\
      --models gemma output-gemma/ \\
      --output results.json
        """,
    )
    parser.add_argument(
        "--models",
        nargs=2,
        action="append",
        metavar=("NAME", "DIR"),
        required=True,
        help="模型名称和输出目录，可多次指定",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="评分维度 YAML 配置文件路径，默认: cascade_test/pipeline/eval_dimensions.yaml",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="评分结果 JSON 输出路径，默认: cascade_test/pipeline/eval_results.json",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Gradio 服务端口（默认 7860）",
    )
    parser.add_argument(
        "--input",
        default="input",
        help="原始输入音频文件夹路径，默认: input（相对于运行目录）",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="生成 Gradio 公开分享链接",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent

    # 配置路径
    config_path = Path(args.config) if args.config else script_dir / "eval_dimensions.yaml"
    output_path = Path(args.output) if args.output else script_dir / "eval_results.json"
    input_dir = Path(args.input)

    # 加载输入原始音频
    print(f"加载输入音频: {input_dir}")
    input_audio_map = load_input_audio_map(input_dir)
    print(f"  → {len(input_audio_map)} 个原始音频文件")

    # 加载维度
    print(f"加载维度配置: {config_path}")
    dimensions = load_dimensions(config_path)
    print(f"维度: {[d['name'] for d in dimensions]}")

    # 加载多模型数据
    models_data: list[tuple[str, dict[str, dict[str, Any]]]] = []
    for model_name, model_dir in args.models:
        dir_path = Path(model_dir)
        print(f"加载模型 [{model_name}]: {dir_path}")
        data = load_model_data(dir_path)
        ok_count = len(data)
        print(f"  → {ok_count} 条有效结果")
        models_data.append((model_name, data))

    model_names = [m[0] for m in models_data]

    # 匹配题目
    questions = match_questions(models_data)
    print(f"匹配成功: {len(questions)} 题（所有模型共有）")

    if not questions:
        print("ERROR: 没有可评分的题目")
        sys.exit(1)

    # 尝试加载已有结果（支持增量：新模型/新维度不会丢失旧评分）
    existing_scores = load_results(output_path)
    if existing_scores:
        prev_idx = existing_scores.get("_current_index", 0)
        prev_models = existing_scores.get("_models", [])
        prev_dims = [d["name"] for d in existing_scores.get("_dimensions", [])]
        curr_dims = [d["name"] for d in dimensions]

        new_models = [m for m in model_names if m not in prev_models]
        removed_models = [m for m in prev_models if m not in model_names]
        new_dims = [d for d in curr_dims if d not in prev_dims]
        removed_dims = [d for d in prev_dims if d not in curr_dims]

        info_parts = [f"加载已有评分结果: {output_path}（已完成 {prev_idx}/{len(questions)} 题）"]
        if new_models:
            info_parts.append(f"新增模型: {new_models}（旧模型评分保留）")
        if removed_models:
            info_parts.append(f"移除模型: {removed_models}")
        if new_dims:
            info_parts.append(f"新增维度: {new_dims}（默认 5 分，需逐题重评）")
        if removed_dims:
            info_parts.append(f"移除维度: {removed_dims}")
        print(" | ".join(info_parts))
    else:
        print("新评分任务")

    # 启动 Gradio
    app = build_app(questions, dimensions, model_names, output_path, existing_scores, input_audio_map)
    print(f"\n启动评分界面: http://localhost:{args.port}")
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
