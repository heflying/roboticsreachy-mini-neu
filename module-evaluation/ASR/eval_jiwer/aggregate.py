"""结果汇总脚本。

读取 results/ 目录下所有模型 JSON，
生成对比汇总。

用法:
    python aggregate.py --results-dir results
    python aggregate.py --results-dir results --output comparison.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 ASR 评测结果")
    parser.add_argument(
        "--results-dir", default="results", help="结果 JSON 目录"
    )
    parser.add_argument(
        "--output", default="comparison.json", help="汇总输出文件"
    )
    return parser.parse_args(argv)


def load_results(results_dir: Path) -> list[dict]:
    """加载目录下所有 JSON 结果文件。"""
    results = []
    for f in sorted(results_dir.glob("*.json")):
        with open(f, "r", encoding="utf-8") as fh:
            results.append(json.load(fh))
    return results


def build_comparison(results: list[dict]) -> dict:
    """生成对比表。"""
    if not results:
        return {"error": "No results found"}

    # 提取每个模型的摘要
    models = []
    for r in results:
        meta = r.get("meta", {})
        hw = r.get("hardware", {})

        model_entry = {
            "model_name": meta.get("model_name", "unknown"),
            "model_type": meta.get("model_type", "unknown"),
            "timestamp": meta.get("timestamp", ""),
            "cpu": hw.get("cpu_model", "unknown"),
            "datasets": {},
        }

        for ds_result in r.get("results", []):
            ds_name = ds_result.get("dataset_name", "unknown")
            agg_cer = ds_result.get("aggregate_cer", {})
            agg_timing = ds_result.get("aggregate_timing", {})

            model_entry["datasets"][ds_name] = {
                "num_utterances": ds_result.get("num_utterances", 0),
                "num_evaluated": ds_result.get("num_evaluated", 0),
                "cer": agg_cer.get("cer"),
                "accuracy": agg_cer.get("accuracy"),
                "sub_rate": agg_cer.get("sub_rate"),
                "del_rate": agg_cer.get("del_rate"),
                "ins_rate": agg_cer.get("ins_rate"),
                "ser": agg_cer.get("ser"),
                "avg_rtf": agg_timing.get("avg_rtf"),
                "avg_ttfc_s": agg_timing.get("avg_ttfc_s"),
                "avg_e2e_s": agg_timing.get("avg_e2e_latency_s"),
                "avg_convergence_s": agg_timing.get("avg_convergence_latency_s"),
                "p50_ttfc_s": agg_timing.get("p50_ttfc_s"),
                "p90_ttfc_s": agg_timing.get("p90_ttfc_s"),
                "p95_ttfc_s": agg_timing.get("p95_ttfc_s"),
                "total_audio_hours": round(
                    agg_timing.get("total_audio_duration_s", 0) / 3600, 3
                ),
            }

        models.append(model_entry)

    return {
        "comparison_timestamp": results[0].get("meta", {}).get("timestamp", ""),
        "num_models": len(models),
        "models": models,
    }


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return 1

    results = load_results(results_dir)
    print(f"Loaded {len(results)} result files from {results_dir}")

    comparison = build_comparison(results)

    output_path = results_dir / args.output if not Path(args.output).is_absolute() else Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    print(f"\nComparison written to: {output_path}")

    # 打印文本摘要
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    for model in comparison.get("models", []):
        print(f"\n--- {model['model_name']} ({model['model_type']}) ---")
        print(f"    CPU: {model['cpu']}")
        for ds_name, ds in model.get("datasets", {}).items():
            print(f"  [{ds_name}]")
            print(f"    CER: {ds.get('cer')} | Acc: {ds.get('accuracy')}")
            print(f"    Sub: {ds.get('sub_rate')} | Del: {ds.get('del_rate')} | Ins: {ds.get('ins_rate')}")
            print(f"    SER: {ds.get('ser')} | Utterances: {ds.get('num_evaluated')}")
            print(f"    RTF: {ds.get('avg_rtf')} | TTFC(p50): {ds.get('p50_ttfc_s')}s | E2E(avg): {ds.get('avg_e2e_s')}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
