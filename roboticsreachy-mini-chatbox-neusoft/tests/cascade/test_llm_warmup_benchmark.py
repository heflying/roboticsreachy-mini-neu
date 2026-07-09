"""Benchmark: sentence-by-sentence warmup vs one-shot LLM generation.

Compares two approaches:
1. Incremental: warmup sentence-by-sentence (simulating ASR sentence pauses),
   then call generate() with all messages.
2. One-shot: call generate() directly with all messages (no warmup).

Usage:
    python -m tests.cascade.test_llm_warmup_benchmark [incremental|oneshot|both]

    # Default: run both
    python -m tests.cascade.test_llm_warmup_benchmark

    # Run only incremental warmup test
    python -m tests.cascade.test_llm_warmup_benchmark incremental

    # Run only one-shot test
    python -m tests.cascade.test_llm_warmup_benchmark oneshot
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from typing import Any, Dict, List

# Ensure project root is on sys.path so imports work when running directly
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# --- Test sentences (simulating multi-sentence user input) ---
TEST_SENTENCES = [
    "你好，我想了解一下今天的天气。",
    "另外，帮我查一下明天北京的天气预报。",
    "还有，下周会有雨？",
    "顺便问一下，气温大概在什么范围？",
    "需要带伞出门吗？",
    "风力大不大？",
    "紫外线强不强？",
    "空气质量怎么样？",
    "适合户外运动吗？",
    "温差大不大？",
]

SYSTEM_INSTRUCTIONS = "你是一个友好的助手，请用简洁的中文回答问题。"

# Number of repetitions for stable timing
NUM_REPEATS = 1


async def _init_llm() -> Any:
    """Initialize LLM provider from cascade.yaml config."""
    from reachy_mini_conversation_app.cascade.config import get_config
    from reachy_mini_conversation_app.cascade.provider_factory import init_provider

    config = get_config()
    extra_kwargs: Dict[str, Any] = {"system_instructions": SYSTEM_INSTRUCTIONS}
    if config.SPARK_APP_ID and "spark" in config.llm_provider.lower():
        extra_kwargs["app_id"] = config.SPARK_APP_ID

    return init_provider("llm", extra_kwargs)


async def _collect_generate_output(llm: Any, messages: List[Dict[str, Any]], temperature: float) -> str:
    """Run generate() and collect full text output."""
    text_parts: list[str] = []
    async for chunk in llm.generate(messages=messages, temperature=temperature):
        if chunk.type == "text_delta" and chunk.content:
            text_parts.append(chunk.content)
    return "".join(text_parts)


def _print_header(llm: Any, temperature: float) -> None:
    """Print benchmark header with config info."""
    from reachy_mini_conversation_app.cascade.config import get_config
    config = get_config()

    print("=" * 70)
    print(f"LLM Benchmark  |  Provider: {config.llm_provider}  |  Model: {getattr(llm, 'model', 'N/A')}")
    print("=" * 70)
    print(f"  Temperature: {temperature}")
    print(f"  Sentences:   {len(TEST_SENTENCES)}")
    print(f"  Repeats:     {NUM_REPEATS}")
    print(f"  System:      {SYSTEM_INSTRUCTIONS[:50]}...")
    print()
    print("Test sentences:")
    for i, s in enumerate(TEST_SENTENCES, 1):
        print(f"  [{i}] {s}")
    print()


# ---------------------------------------------------------------------------
# Route 1: Incremental warmup
# ---------------------------------------------------------------------------

async def test_incremental(num_repeats: int = NUM_REPEATS) -> None:
    """Warmup sentence-by-sentence, then generate with full text.

    Simulates the real pipeline where ASR sentence pauses trigger warmup calls.
    All sentences are accumulated in a SINGLE user message content.
    """
    llm = await _init_llm()
    from reachy_mini_conversation_app.cascade.config import get_config
    temperature = get_config().llm_temperature

    _print_header(llm, temperature)
    print("Route: INCREMENTAL (sentence-by-sentence warmup + final generate)")
    print("-" * 70)

    # Pre-warm connection
    print("Pre-warming LLM connection...")
    await llm.warmup()
    print("Connection ready.\n")

    results: list[Dict[str, Any]] = []

    for i in range(num_repeats):
        conversation_history: List[Dict[str, Any]] = []
        accumulated_text = ""
        total_warmup_ms = 0.0
        warmup_times: list[float] = []

        for sentence in TEST_SENTENCES:
            accumulated_text += sentence
            warmup_messages = conversation_history + [
                {"role": "user", "content": accumulated_text}
            ]
            t0 = time.perf_counter()
            await llm.warmup(messages=warmup_messages, temperature=temperature)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            warmup_times.append(elapsed_ms)
            total_warmup_ms += elapsed_ms

        full_messages = conversation_history + [
            {"role": "user", "content": accumulated_text}
        ]
        t0 = time.perf_counter()
        output = await _collect_generate_output(llm, full_messages, temperature)
        generate_ms = (time.perf_counter() - t0) * 1000

        result = {
            "output": output,
            "total_warmup_ms": total_warmup_ms,
            "warmup_times_ms": warmup_times,
            "generate_ms": generate_ms,
            "total_ms": total_warmup_ms + generate_ms,
        }
        results.append(result)

        print(f"  Round {i + 1}: warmup={total_warmup_ms:.0f}ms "
              f"({[f'{t:.0f}' for t in warmup_times]}) "
              f"+ generate={generate_ms:.0f}ms "
              f"= total={result['total_ms']:.0f}ms")

    # Summary
    avg_warmup = sum(r["total_warmup_ms"] for r in results) / num_repeats
    avg_generate = sum(r["generate_ms"] for r in results) / num_repeats
    avg_total = sum(r["total_ms"] for r in results) / num_repeats
    per_sentence = [
        sum(r["warmup_times_ms"][j] for r in results) / num_repeats
        for j in range(len(TEST_SENTENCES))
    ]

    print()
    print(f"  Avg warmup total:    {avg_warmup:.0f}ms")
    print(f"  Per-sentence warmup: {[f'{t:.0f}ms' for t in per_sentence]}")
    print(f"  Avg generate:        {avg_generate:.0f}ms")
    print(f"  Avg total:           {avg_total:.0f}ms")
    print()
    print(f"  Sample output: {results[-1]['output'][:100]}...")


# ---------------------------------------------------------------------------
# Route 2: One-shot generate
# ---------------------------------------------------------------------------

async def test_oneshot(num_repeats: int = NUM_REPEATS) -> None:
    """Generate directly with all messages in one shot, no warmup."""
    llm = await _init_llm()
    from reachy_mini_conversation_app.cascade.config import get_config
    temperature = get_config().llm_temperature

    _print_header(llm, temperature)
    print("Route: ONE-SHOT (single generate, no warmup)")
    print("-" * 70)

    # Pre-warm connection
    print("Pre-warming LLM connection...")
    await llm.warmup()
    print("Connection ready.\n")

    full_text = "".join(TEST_SENTENCES)
    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": full_text}
    ]

    results: list[Dict[str, Any]] = []

    for i in range(num_repeats):
        t0 = time.perf_counter()
        output = await _collect_generate_output(llm, messages, temperature)
        generate_ms = (time.perf_counter() - t0) * 1000

        result = {
            "output": output,
            "generate_ms": generate_ms,
            "total_ms": generate_ms,
        }
        results.append(result)

        print(f"  Round {i + 1}: generate={generate_ms:.0f}ms")

    # Summary
    avg_generate = sum(r["generate_ms"] for r in results) / num_repeats
    avg_total = sum(r["total_ms"] for r in results) / num_repeats

    print()
    print(f"  Avg generate: {avg_generate:.0f}ms")
    print(f"  Avg total:    {avg_total:.0f}ms")
    print()
    print(f"  Sample output: {results[-1]['output'][:100]}...")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_both(num_repeats: int = NUM_REPEATS) -> None:
    """Run both routes and compare."""
    llm = await _init_llm()
    from reachy_mini_conversation_app.cascade.config import get_config
    temperature = get_config().llm_temperature

    _print_header(llm, temperature)
    print("Route: BOTH (incremental vs one-shot comparison)")
    print("-" * 70)

    # Pre-warm connection
    print("Pre-warming LLM connection...")
    await llm.warmup()
    print("Connection ready.\n")

    inc_results: list[Dict[str, Any]] = []
    one_results: list[Dict[str, Any]] = []

    for i in range(num_repeats):
        print(f"--- Round {i + 1}/{num_repeats} ---")

        # Incremental
        conversation_history: List[Dict[str, Any]] = []
        accumulated_text = ""
        total_warmup_ms = 0.0
        warmup_times: list[float] = []

        for sentence in TEST_SENTENCES:
            accumulated_text += sentence
            warmup_messages = conversation_history + [
                {"role": "user", "content": accumulated_text}
            ]
            t0 = time.perf_counter()
            await llm.warmup(messages=warmup_messages, temperature=temperature)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            warmup_times.append(elapsed_ms)
            total_warmup_ms += elapsed_ms

        full_messages = conversation_history + [
            {"role": "user", "content": accumulated_text}
        ]
        t0 = time.perf_counter()
        inc_output = await _collect_generate_output(llm, full_messages, temperature)
        inc_generate_ms = (time.perf_counter() - t0) * 1000

        inc_results.append({
            "total_warmup_ms": total_warmup_ms,
            "warmup_times_ms": warmup_times,
            "generate_ms": inc_generate_ms,
            "total_ms": total_warmup_ms + inc_generate_ms,
        })

        # One-shot
        full_text = "".join(TEST_SENTENCES)
        one_messages: List[Dict[str, Any]] = [
            {"role": "user", "content": full_text}
        ]
        t0 = time.perf_counter()
        one_output = await _collect_generate_output(llm, one_messages, temperature)
        one_generate_ms = (time.perf_counter() - t0) * 1000

        one_results.append({
            "generate_ms": one_generate_ms,
            "total_ms": one_generate_ms,
        })

        print(f"  Incremental: warmup={total_warmup_ms:.0f}ms + generate={inc_generate_ms:.0f}ms = {total_warmup_ms + inc_generate_ms:.0f}ms")
        print(f"  One-shot:    generate={one_generate_ms:.0f}ms")
        print()

    # Summary
    avg_inc_total = sum(r["total_ms"] for r in inc_results) / num_repeats
    avg_inc_warmup = sum(r["total_warmup_ms"] for r in inc_results) / num_repeats
    avg_inc_gen = sum(r["generate_ms"] for r in inc_results) / num_repeats
    avg_one_total = sum(r["total_ms"] for r in one_results) / num_repeats
    avg_one_gen = sum(r["generate_ms"] for r in one_results) / num_repeats

    per_sentence = [
        sum(r["warmup_times_ms"][j] for r in inc_results) / num_repeats
        for j in range(len(TEST_SENTENCES))
    ]

    print("=" * 70)
    print(f"SUMMARY (averaged over {num_repeats} rounds)")
    print("=" * 70)
    print()
    print(f"  Incremental warmup + generate:")
    print(f"    Avg warmup:    {avg_inc_warmup:.0f}ms  (per-sentence: {[f'{t:.0f}ms' for t in per_sentence]})")
    print(f"    Avg generate:  {avg_inc_gen:.0f}ms")
    print(f"    Avg total:     {avg_inc_total:.0f}ms")
    print()
    print(f"  One-shot generate:")
    print(f"    Avg generate:  {avg_one_gen:.0f}ms")
    print(f"    Avg total:     {avg_one_total:.0f}ms")
    print()

    diff = avg_inc_total - avg_one_total
    pct = (diff / avg_one_total * 100) if avg_one_total > 0 else 0
    label = "SLOWER" if diff > 0 else "FASTER"
    print(f"  Incremental is {abs(diff):.0f}ms {label} than one-shot ({pct:+.1f}%)")
    print()

    gen_diff = avg_inc_gen - avg_one_gen
    gen_pct = (gen_diff / avg_one_gen * 100) if avg_one_gen > 0 else 0
    if gen_diff < 0:
        print(f"  Generate-only: warmup saves {-gen_diff:.0f}ms ({gen_pct:.1f}%)")
    else:
        print(f"  Generate-only: warmup adds {gen_diff:.0f}ms ({gen_pct:+.1f}%)")

    print()
    print(f"  Sample incremental output: {inc_output[:100]}...")
    print(f"  Sample one-shot output:    {one_output[:100]}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM warmup benchmark")
    parser.add_argument(
        "route",
        nargs="?",
        default="oneshot",
        choices=["incremental", "oneshot", "both"],
        help="Which route to test (default: both)",
    )
    parser.add_argument(
        "-n", "--repeats",
        type=int,
        default=NUM_REPEATS,
        help=f"Number of repetitions (default: {NUM_REPEATS})",
    )
    args = parser.parse_args()

    if args.route == "incremental":
        asyncio.run(test_incremental(args.repeats))
    elif args.route == "oneshot":
        asyncio.run(test_oneshot(args.repeats))
    else:
        asyncio.run(run_both(args.repeats))


if __name__ == "__main__":
    main()
