"""Interactive CLI chat script for testing the Router + LLM pipeline.

Uses the project's cascade.yaml + .env configuration to create a Router and
an LLM provider, then enters a REPL loop for multi-turn conversation.
Before each LLM response, the Router classifies the user input and prints
the routing decision.

Usage:
    cd project_root
    python cascade_test/debug/debug_router.py
    python cascade_test/debug/debug_router.py --provider ollama-qwen2.5-0.5b
    python cascade_test/debug/debug_router.py --provider spark-ultra --temperature 0.5
"""

from __future__ import annotations

import asyncio
import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path for imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env for API keys
os.environ.pop("REACHY_MINI_SKIP_DOTENV", None)
from dotenv import load_dotenv

load_dotenv(override=True)

from reachy_mini_conversation_app.cascade.llm.base import LLMChunk, LLMProvider
from reachy_mini_conversation_app.cascade.router.base import RouteResult
from reachy_mini_conversation_app.cascade.router.composite import CompositeRouter

# ---------------------------------------------------------------------------
# Configuration — adjust these as needed
# ---------------------------------------------------------------------------

# System instructions override: set to a string to replace the profile default,
# or leave as None to use whatever create_llm_provider() injects.
SYSTEM_INSTRUCTIONS: Optional[str] = None

# Default LLM provider (reads from .env CASCADE_LLM_PROVIDER if not overridden via --provider)
DEFAULT_PROVIDER: Optional[str] = None

# Default temperature
DEFAULT_TEMPERATURE: float = 0.7

SHOW_REASON: bool = True

# ---------------------------------------------------------------------------
# Provider creation
# ---------------------------------------------------------------------------


def create_router() -> CompositeRouter:
    """Create a CompositeRouter instance."""
    print("Initializing Router...")
    router = CompositeRouter()
    print("Router initialized.")
    return router


def create_chat_llm(provider_name: Optional[str] = None) -> LLMProvider:
    """Create a chat LLM provider using the cascade framework.

    Reuses cascade_test.LLM.framework.create_llm_provider() which reads
    cascade.yaml and .env to resolve API keys and model parameters.
    """
    from cascade_test.LLM.framework import create_llm_provider, get_available_llm_providers

    # Resolve provider name: CLI arg > module constant > .env CASCADE_LLM_PROVIDER
    name = provider_name or DEFAULT_PROVIDER or os.getenv("CASCADE_LLM_PROVIDER")
    if not name:
        available = get_available_llm_providers()
        print("ERROR: No LLM provider specified.")
        print("  Use --provider <name> or set CASCADE_LLM_PROVIDER in .env")
        print(f"  Available providers: {', '.join(available)}")
        sys.exit(1)

    print(f"Initializing chat LLM provider: {name}")
    llm = create_llm_provider(name)

    # Override system instructions if configured above
    if SYSTEM_INSTRUCTIONS is not None and hasattr(llm, "system_instructions"):
        llm.system_instructions = SYSTEM_INSTRUCTIONS

    return llm


# ---------------------------------------------------------------------------
# Router test cases — (input, expected_decision) pairs
# ---------------------------------------------------------------------------

_ROUTER_TEST_CASES: List[tuple[str, str]] = [
    # TODO: Add test cases here, e.g.
    ("现在几点", "no_privacy"),
    ("明天的天气怎么样", "no_privacy"),
    ("python是什么语言，我怎么学习它", "no_privacy"),
    ("我喜欢看水浒传，里面的宋江最后是什么结局，是好还是坏", "no_privacy"),
    ("今天有关于大连的新闻么，特别是关于高新园区的新闻", "no_privacy"),

    ("我叫王丽，很高兴认识你", "privacy"),
    ("我今天身体有些不舒服，应该吃点什么药呢", "privacy"),
    ("我家住在高新园区黄浦路10号", "privacy"),
    ("我想起来了，我把钱包忘到车里了", "privacy"),
    ("银行卡放到桌子底下了，会不会被压坏啊", "privacy"),
]


# ---------------------------------------------------------------------------
# Router fixed-input test
# ---------------------------------------------------------------------------


async def run_router_tests(router: CompositeRouter) -> None:
    """Run fixed-input router tests before entering the interactive chat loop.

    Only prints cases where the router decision does not match the expected result.
    """
    if not _ROUTER_TEST_CASES:
        print("[No router test cases configured, skipping]\n")
        return

    error_count = 0
    total = len(_ROUTER_TEST_CASES)

    print(f"Running {total} router test cases...")
    for text, expected in _ROUTER_TEST_CASES:
        messages: List[Dict[str, Any]] = [{"role": "user", "content": text}]
        result = await router.route(messages, SHOW_REASON)
        if result.decision != expected:
            error_count += 1
            print(f"  MISMATCH: input={text!r} | expected={expected} | got={result.decision} | reason={result.reason}")

    passed = total - error_count
    print(f"Router tests: {passed}/{total} passed, {error_count} failed\n")


# ---------------------------------------------------------------------------
# Chat loop
# ---------------------------------------------------------------------------


async def chat_repl(router: CompositeRouter, llm: LLMProvider, temperature: float) -> None:
    """Run the interactive chat REPL with router classification."""
    # Run fixed-input router tests first
    await run_router_tests(router)

    messages: List[Dict[str, Any]] = []

    # If the provider has system_instructions, show them
    sys_instr = getattr(llm, "system_instructions", None)
    if sys_instr:
        print(f"\n[System instructions loaded ({len(sys_instr)} chars)]")
    else:
        print("\n[No system instructions]")

    model_name = getattr(llm, "model", "unknown")
    print(f"Chat LLM: {model_name}")
    print(f"Router LLM: ollama-qwen2.5-1.5b-instruct")
    print(f"Temperature: {temperature}")
    print(f"Type 'quit' or 'exit' to leave.\n")

    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Bye!")
            break

        # Append user message to history
        messages.append({"role": "user", "content": user_input})

        # Router classification
        route_start = time.perf_counter()
        route_result: RouteResult = await router.route(messages, SHOW_REASON)
        route_time = (time.perf_counter() - route_start) * 1000
        reason_str = f" | {route_result.reason}" if route_result.reason else ""
        print(f"  [Router: {route_result.decision}{reason_str} | {route_time:.0f}ms]")

        # Stream LLM response
        assistant_text = ""
        tool_calls: List[Dict[str, Any]] = []
        first_token = True
        ttft: Optional[float] = None
        request_start = time.perf_counter()

        try:
            async for chunk in llm.generate(
                messages=messages,
                tools=None,
                temperature=temperature,
                token=None,
            ):
                if chunk.type == "text_delta" and chunk.content:
                    if first_token:
                        ttft = time.perf_counter() - request_start
                        print("AI> ", end="", flush=True)
                        first_token = False
                    print(chunk.content, end="", flush=True)
                    assistant_text += chunk.content
                elif chunk.type == "tool_call" and chunk.tool_call:
                    if first_token:
                        ttft = time.perf_counter() - request_start
                        print("AI> ", end="", flush=True)
                        first_token = False
                    tool_calls.append(chunk.tool_call)
                    func = chunk.tool_call.get("function", {})
                    name = func.get("name", "?")
                    args = func.get("arguments", {})
                    print(f"[tool_call: {name}({args})]", end="", flush=True)
                elif chunk.type == "done":
                    pass
        except Exception as e:
            print(f"\n[Error: {e}]")
            # Remove the failed user message so history stays consistent
            messages.pop()
            continue

        total_time = time.perf_counter() - request_start

        # Finish the line after streaming and print timing
        if not first_token:
            ttft_str = f"{ttft * 1000:.0f}ms" if ttft else "N/A"
            print(f"\n  [TTFT: {ttft_str} | Total: {total_time * 1000:.0f}ms]")

        # Append assistant response to history
        if assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})
        elif tool_calls:
            # For tool calls, store a minimal assistant message so history remains valid
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls,
            })


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive CLI chat for testing Router + LLM pipeline",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Chat LLM provider name from cascade.yaml (default: read from .env CASCADE_LLM_PROVIDER)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    # Initialize Router first (it uses a fixed provider name, not affected by env)
    router = create_router()

    # Then initialize chat LLM (may reset config singleton via create_llm_provider)
    llm = create_chat_llm(args.provider)

    asyncio.run(chat_repl(router, llm, args.temperature))


if __name__ == "__main__":
    main()
