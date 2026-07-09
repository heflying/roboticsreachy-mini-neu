#!/usr/bin/env python3
"""Run elder-care memory evaluation scenarios."""

from __future__ import annotations
import asyncio
import argparse
from pathlib import Path

from dotenv import load_dotenv

from reachy_mini_conversation_app.memory.eval import run_cases, load_cases, write_reports, summarize_results


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        default="tests/memory_scenarios/eldercare_smoke.json",
        help="JSON case file path.",
    )
    parser.add_argument(
        "--mode",
        choices=["case", "offline", "qwen-extractor", "realtime-headless"],
        default="case",
        help="Execution mode. qwen-extractor requires --allow-real-api.",
    )
    parser.add_argument("--db", help="SQLite file or directory for evaluation DBs.")
    parser.add_argument("--report-dir", default="eval_reports", help="Directory for JSON/Markdown reports.")
    parser.add_argument("--case-id", action="append", help="Run only the selected case id. May be repeated.")
    parser.add_argument("--limit", type=int, help="Run at most N cases.")
    parser.add_argument("--keep-db", action="store_true", help="Keep generated per-case SQLite DB files.")
    parser.add_argument(
        "--allow-real-api",
        action="store_true",
        help="Allow calls to real Qwen/DashScope APIs for qwen-extractor mode.",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit non-zero when any case fails, errors, or is skipped.",
    )
    return parser.parse_args()


async def main_async() -> int:
    """Run the evaluation and return a process exit code."""
    load_dotenv(override=True)
    args = parse_args()
    cases = load_cases(args.cases)
    results = await run_cases(
        cases,
        mode=args.mode,
        db_path=args.db,
        keep_db=args.keep_db,
        allow_real_api=args.allow_real_api,
        case_ids=set(args.case_id or []) or None,
        limit=args.limit,
    )
    json_path, md_path = write_reports(results, Path(args.report_dir))
    summary = summarize_results(results)
    print(
        "Memory eval complete: "
        f"{summary['passed']}/{summary['total']} passed, "
        f"{summary['failed']} failed, {summary['skipped']} skipped, {summary['errors']} errors."
    )
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    if args.fail_on_error and (
        summary["failed"] > 0 or summary["errors"] > 0 or summary["skipped"] > 0
    ):
        return 1
    return 0


def main() -> None:
    """CLI entrypoint."""
    raise SystemExit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
