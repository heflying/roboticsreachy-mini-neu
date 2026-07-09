"""评测管线入口。

用法:
    cd eval_jiwer
    python run_eval.py \
        --manifest manifests/seniortalk_test.jsonl manifests/aishell1_test.jsonl \
        --max-utterances 100 \
        --output-dir results

输出: results/{model_name}_{timestamp}.json
"""

import json
import logging
import sys
from pathlib import Path

import config
from utils import get_hardware_info, setup_logging, now_iso
from jsonl_loader import JsonlLoader
from vad.segmenter import SileroSegmenter
from eval.runner import (
    EvaluationRunner,
    create_asr,
    build_result,
)

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    args = config.parse_args(argv)
    setup_logging(args.verbose)

    model_name = Path(args.sherpa_model_dir).name

    logger.info("=" * 60)
    logger.info(f"eval_jiwer — ASR Evaluation Pipeline")
    logger.info(f"Model: {model_name} ({args.model_type})")
    logger.info(f"Manifests: {args.manifest}")
    logger.info("=" * 60)

    # 硬件信息
    hw_info = get_hardware_info()
    logger.info(f"CPU: {hw_info['cpu_model']}")
    logger.info(f"Threads: {args.sherpa_num_threads}")

    # 创建 ASR
    logger.info("Initializing ASR model...")
    asr = create_asr(args)
    asr.warmup()
    logger.info(f"ASR ready: {asr.model_info}")

    # 创建 VAD
    segmenter = SileroSegmenter(
        threshold=args.vad_threshold,
        min_speech_duration_ms=args.vad_min_speech_ms,
        min_silence_duration_ms=args.vad_min_silence_ms,
    )

    # 创建 runner
    runner = EvaluationRunner(
        asr=asr,
        segmenter=segmenter,
        strip_punctuation=not args.keep_punctuation,
    )

    # 逐个 manifest 评测
    manifest_paths = config.get_manifest_list(args.manifest)
    dataset_results = []

    for manifest_path in manifest_paths:
        logger.info(f"\n{'=' * 40}")
        logger.info(f"Manifest: {manifest_path}")

        loader = JsonlLoader(
            manifest_path=manifest_path,
            max_hours=args.max_hours,
            max_utterances=args.max_utterances,
        )

        result = runner.evaluate_dataset(loader, dry_run=args.dry_run)
        dataset_results.append(result)

        # 打印摘要
        agg_cer = result.get("aggregate_cer", {})
        agg_timing = result.get("aggregate_timing", {})
        logger.info(f"  CER: {agg_cer.get('cer', 'N/A')}")
        logger.info(f"  Sub/Del/Ins: {agg_cer.get('sub_rate')}/{agg_cer.get('del_rate')}/{agg_cer.get('ins_rate')}")
        logger.info(f"  Avg RTF: {agg_timing.get('avg_rtf', 'N/A')}")
        logger.info(f"  Avg TTFC: {agg_timing.get('avg_ttfc_s', 'N/A')}s")
        logger.info(f"  Avg E2E: {agg_timing.get('avg_e2e_latency_s', 'N/A')}s")

    # 构建最终结果
    final = build_result(model_name, args, dataset_results, hw_info)
    if args.dry_run:
        logger.info("Dry run complete, skipping output")
        return 0

    # 写入 JSON
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = now_iso().replace(":", "").replace("-", "")
    filename = f"{model_name}_{timestamp}.json"
    output_path = output_dir / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    logger.info(f"\nResults written to: {output_path}")
    logger.info("Evaluation complete.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
