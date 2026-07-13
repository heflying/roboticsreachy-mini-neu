"""Main runner for LLM evaluation tests.

Usage:
    cd project_root
    python -m cascade_test.LLM.runner --provider qwen-flash
    python -m cascade_test.LLM.runner --provider ollama-qwen2.5-0.5b --category 01-事实问答
    python -m cascade_test.LLM.runner --provider spark-ultra --output-dir ./eval_results
"""

from __future__ import annotations

import asyncio
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Ensure project root is on sys.path for imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Skip .env loading during tests
os.environ.setdefault("REACHY_MINI_SKIP_DOTENV", "1")

from cascade_test.LLM.framework import (
    BASE_DIR,
    CATEGORIES,
    TestCase,
    CaseResult,
    CategoryReport,
    create_llm_provider,
    get_available_llm_providers,
    load_test_cases,
    run_single_case,
    rate_ttft,
    rate_tokens_per_sec,
)
from cascade_test.LLM.scoring import (
    score_case,
    aggregate_category,
)
from cascade_test.LLM.report import generate_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s:%(lineno)d | %(message)s",
)
logger = logging.getLogger(__name__)


async def run_evaluation(
    provider_name: str,
    category: Optional[str] = None,
    output_dir: Optional[Path] = None,
    warmup: bool = True,
) -> Dict[str, CategoryReport]:
    """Run the full LLM evaluation suite.

    Args:
        provider_name: LLM provider name from cascade.yaml (e.g. "qwen-flash",
            "ollama-qwen2.5-0.5b", "spark-ultra").
        category: If provided, only run this category.
        output_dir: Directory for report output.
        warmup: Whether to warm up the LLM before testing.

    Returns:
        Dict mapping category name -> CategoryReport.
    """
    logger.info("Creating LLM provider: %s", provider_name)

    try:
        llm = create_llm_provider(provider_name)
    except RuntimeError as e:
        logger.error("Failed to create LLM provider: %s", e)
        print(f"\nERROR: {e}\n")
        return {}

    model_name = getattr(llm, "model", "unknown")

    # Warmup
    if warmup:
        logger.info("Warming up LLM...")
        try:
            await llm.warmup()
            logger.info("Warmup complete.")
        except Exception as e:
            logger.warning("Warmup failed (non-critical): %s", e)

    # Load test cases
    cases = load_test_cases(category)
    if not cases:
        logger.error("No test cases found. Please populate the test case directories.")
        return {}

    logger.info("Loaded %d test cases.", len(cases))

    # Group cases by category
    cases_by_category: Dict[str, List[TestCase]] = {}
    for tc in cases:
        cases_by_category.setdefault(tc.category, []).append(tc)

    # Run cases
    category_reports: Dict[str, CategoryReport] = {}

    for cat, cat_cases in cases_by_category.items():
        logger.info("Running category: %s (%d cases)", cat, len(cat_cases))
        results: List[CaseResult] = []

        for i, tc in enumerate(cat_cases):
            logger.info(
                "  [%d/%d] Running case: %s", i + 1, len(cat_cases), tc.case_id
            )

            # Determine if tools should be provided
            provide_tools = cat == "03-工具调用"

            result = await run_single_case(
                llm=llm,
                tc=tc,
                provide_tools=provide_tools,
            )

            # Score the result
            score_case(result, tc)

            # Log summary
            ttft_str = f"{result.speed.ttft_ms}ms" if result.speed.ttft_ms else "N/A"
            logger.info(
                "  → TTFT: %s, Tokens: %d, Pass: %s",
                ttft_str,
                result.speed.total_tokens,
                result.passed,
            )

            results.append(result)

        # Aggregate
        report = aggregate_category(results, cat)
        category_reports[cat] = report

        logger.info(
            "Category %s: %d/%d passed",
            cat,
            report.passed_cases,
            report.total_cases,
        )

    # Generate report
    report_text = generate_report(
        category_reports=category_reports,
        provider_name=provider_name,
        model_name=model_name,
        output_dir=output_dir,
    )

    # Print report to console
    print("\n" + "=" * 60)
    print(report_text)
    print("=" * 60 + "\n")

    return category_reports


def main():
    """CLI entry point."""
    # Dynamically get available providers from cascade.yaml
    available_providers = get_available_llm_providers()

    parser = argparse.ArgumentParser(
        description="LLM evaluation test runner for Reachy Mini cascade pipeline"
    )
    parser.add_argument(
        "--provider",
        required=True,
        choices=available_providers,
        help="LLM provider to evaluate (from cascade.yaml)",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Only run a specific category (e.g. '01-事实问答')",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write report files (default: cascade_test/LLM/reports/)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip LLM warmup before testing",
    )
    parser.add_argument(
        "--load-dotenv",
        action="store_true",
        help="Load .env file (needed for API keys in real runs)",
    )

    args = parser.parse_args()

    # Allow .env loading for real provider runs
    if args.load_dotenv:
        os.environ.pop("REACHY_MINI_SKIP_DOTENV", None)
        from dotenv import load_dotenv
        load_dotenv(override=True)

    output_dir = Path(args.output_dir) if args.output_dir else BASE_DIR / "reports"

    asyncio.run(
        run_evaluation(
            provider_name=args.provider,
            category=args.category,
            output_dir=output_dir,
            warmup=not args.no_warmup,
        )
    )


if __name__ == "__main__":
    main()
