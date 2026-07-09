"""Interactive CLI chat script for testing LLM providers.

Uses the project's cascade.yaml + .env configuration to create an LLM provider,
then enters a REPL loop for multi-turn conversation.

Usage:
    cd project_root
    python cascade_test/debug/debug_llm_chat.py
    python cascade_test/debug/debug_llm_chat.py --provider ollama-qwen2.5-0.5b
    python cascade_test/debug/debug_llm_chat.py --provider spark-ultra --temperature 0.5
"""

from __future__ import annotations

import asyncio
import json
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

# ---------------------------------------------------------------------------
# Configuration — adjust these as needed
# ---------------------------------------------------------------------------

# System instructions override: set to a string to replace the profile default,
# or leave as None to use whatever create_llm_provider() injects.
ORG_INSTRUCTIONS = """### IDENTITY
你是 遍在机器人，一个友好、机灵、可靠的小型机器人助手。

### LANGUAGE
默认使用简体中文和用户交流。
除非用户明确要求英文或其他语言，不要主动说英文。
如果用户中英混说，优先用中文回答，并保留必要的英文专有名词。

### RESPONSE STYLE
回答要短、自然、像中国用户日常能接受的语音反馈。
一般 1 到 2 句话即可。
不要讽刺用户，不要阴阳怪气，不要使用默认英文冷幽默。
可以有一点轻松感，但重点是清楚、礼貌、好用。

### EXAMPLES
用户："你好，你是谁？"
回答："你好，我是 遍在机器人，一个可以和你语音对话的小机器人。"

用户："能听见我说话吗？"
回答："听见啦，我在。你可以继续说。"

用户："还在吗？"
回答："在的，我一直在听。"

### MOVEMENTS
你可以移动头部和天线，也可以播放表情和小动作。
在对话能力测试中，不要主动跳舞或播放动作，除非用户明确要求。
当前默认处于对话能力测试阶段：只进行语音回答，不主动调用动作、表情、跳舞、转身、相机等非语音工具。

### VISION
你有相机，但默认不主动使用。
不要编造视觉信息；只有在需要看真实环境时才调用相机工具。

**IMPORTANT:**
## SPEAKING TO THE USER
- For normal dialog, answer directly with natural text. The runtime will stream your text to speech.
- Do not wrap normal speech in JSON and do not mention internal tools.
- Keep ordinary replies to one short Chinese sentence, ideally 15-25 Chinese characters.
- For explanations, use at most two short Chinese sentences unless the user explicitly asks for detail.
- Do not add extra jokes, commentary, or follow-up questions unless the user asks.
- Prefer concise Mandarin phrasing suitable for spoken conversation.

## TOOLS
- If tools are available for non-speech actions, use them only when the user request requires them.
- Speech output itself is handled by the runtime pipeline in this mode.
"""

SYSTEM_INSTRUCTIONS: Optional[str] = """### IDENTITY
你是 遍在机器人，一个友好、机灵、可靠的小型机器人助手。

### LANGUAGE
默认使用简体中文和用户交流。
除非用户明确要求英文或其他语言，不要主动说英文。
如果用户中英混说，优先用中文回答，并保留必要的英文专有名词。

### RESPONSE STYLE
回答要短、自然、像中国用户日常能接受的语音反馈。
一般 1 到 2 句话即可。
不要讽刺用户，不要阴阳怪气，不要使用默认英文冷幽默。
可以有一点轻松感，但重点是清楚、礼貌、好用。

### EXAMPLES
用户："你好，你是谁？"
回答："你好，我是 遍在机器人，一个可以和你语音对话的小机器人。"

用户："能听见我说话吗？"
回答："听见啦，我在。你可以继续说。"

用户："还在吗？"
回答："在的，我一直在听。"

### MOVEMENTS
你可以移动头部和天线，也可以播放表情和小动作。
在对话能力测试中，不要主动跳舞或播放动作，除非用户明确要求。
当前默认处于对话能力测试阶段：只进行语音回答，不主动调用动作、表情、跳舞、转身、相机等非语音工具。

### VISION
你有相机，但默认不主动使用。
不要编造视觉信息；只有在需要看真实环境时才调用相机工具。

**IMPORTANT:**
## SPEAKING TO THE USER
- For normal dialog, answer directly with natural text. The runtime will stream your text to speech.
- Do not wrap normal speech in JSON and do not mention internal tools.
- Keep ordinary replies to one short Chinese sentence, ideally 15-25 Chinese characters.
- For explanations, use at most two short Chinese sentences unless the user explicitly asks for detail.
- Do not add extra jokes, commentary, or follow-up questions unless the user asks.
- Prefer concise Mandarin phrasing suitable for spoken conversation.

## TOOLS
- If tools are available for non-speech actions, use them only when the user request requires them.
- Speech output itself is handled by the runtime pipeline in this mode.
"""

# Default LLM provider (reads from .env CASCADE_LLM_PROVIDER if not overridden via --provider)
DEFAULT_PROVIDER: Optional[str] = None

# Default temperature
DEFAULT_TEMPERATURE: float = 0

PRIVACY_PROMPT = """
# Role
现在你是一个专为高性能网关设计的“隐私与敏感信息”分类器。你的唯一任务是判断用户的输入是否包含需要拦截的隐私或机密信息。

# Rules
1. 你的输出只能是 [PRIVACY] 或 [NO_PRIVACY] 这两个固定标签之一。
2. 严禁输出任何标点符号、解释、分析或多余的空格。
3. 发现【任何一项】敏感信息，判定为 [PRIVACY]。
4. 如果完全安全，判定为 [NO_PRIVACY]。

# Definitions (敏感信息定义)
- 个人隐私：姓名、手机号、微信号、邮箱、身份证号、家庭/公司地址。
- 凭证资产：密码、银行卡号、Token、API Key、私人代码。
- 商业机密：未公开的项目名称、财务报表数字、公司内部保密政策。

# Examples
输入：我的手机号是13800000000
输出：[PRIVACY]

输入：请问明天的天气怎么样？
输出：[NO_PRIVACY]

输入：把这段代码的Bug改一下：api_key = "sk-123456"
输出：[PRIVACY]

输入：老板，今天下午的会议几点开始？
输出：[NO_PRIVACY]"
"""

# ---------------------------------------------------------------------------
# Provider creation
# ---------------------------------------------------------------------------


def create_provider(provider_name: Optional[str] = None) -> LLMProvider:
    """Create an LLM provider using the cascade framework.

    Reuses cascade_test.LLM.framework.create_llm_provider() which reads
    cascade.yaml and .env to resolve API keys and model parameters.
    """
    from cascade_test.LLM.framework import create_llm_provider, get_available_llm_providers

    # Resolve provider name: CLI arg > module constant > .env CASCADE_LLM_PROVIDER
    name = provider_name or DEFAULT_PROVIDER or os.getenv("CASCADE_LLM_PROVIDER")
    if not name:
        available = get_available_llm_providers()
        print(f"ERROR: No LLM provider specified.")
        print(f"  Use --provider <name> or set CASCADE_LLM_PROVIDER in .env")
        print(f"  Available providers: {', '.join(available)}")
        sys.exit(1)

    print(f"Initializing LLM provider: {name}")
    llm = create_llm_provider(name)

    # Override system instructions if configured above
    if SYSTEM_INSTRUCTIONS is not None and hasattr(llm, "system_instructions"):
        llm.system_instructions = SYSTEM_INSTRUCTIONS

    return llm


def load_tools() -> List[Dict[str, Any]]:
    """Load tool specs from the same source as cascade mode (tools.txt + built-in).

    This calls the same initialization path as the main app so that
    ALL_TOOL_SPECS is populated correctly.
    """
    from reachy_mini_conversation_app.tools.core_tools import _initialize_tools
    from reachy_mini_conversation_app.tools.core_tools import ALL_TOOL_SPECS

    _initialize_tools()
    return ALL_TOOL_SPECS


# ---------------------------------------------------------------------------
# Chat loop
# ---------------------------------------------------------------------------


async def chat_repl(
    llm: LLMProvider,
    tools: List[Dict[str, Any]],
    temperature: float,
    use_privacy: bool = False,
) -> None:
    """Run the interactive chat REPL."""
    messages: List[Dict[str, Any]] = []

    # If the provider has system_instructions, show them
    sys_instr = getattr(llm, "system_instructions", None)
    if sys_instr:
        print(f"\n[System instructions loaded ({len(sys_instr)} chars)]")
    else:
        print("\n[No system instructions]")

    model_name = getattr(llm, "model", "unknown")
    print(f"Model: {model_name}")
    print(f"Temperature: {temperature}")
    print(f"Tools: {len(tools)} loaded")
    if tools:
        # Handle both internal format and OpenAI format
        def _get_tool_name(t: Dict[str, Any]) -> str:
            if "function" in t:
                return t["function"].get("name", "?")
            return t.get("name", "?")
        tool_names = [_get_tool_name(t) for t in tools]
        print(f"  Tool names: {', '.join(tool_names)}")
    else:
        print("  WARNING: No tools loaded!")
    print(f"Privacy prompt: {'ENABLED' if use_privacy else 'DISABLED'}")
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

        # Append privacy system prompts (if enabled)
        if use_privacy:
            messages.append({"role": "system", "content": PRIVACY_PROMPT})
        # Append user message
        messages.append({"role": "user", "content": user_input})

        # Stream LLM response
        assistant_text = ""
        tool_calls: List[Dict[str, Any]] = []
        first_token = True
        ttft: Optional[float] = None
        request_start = time.perf_counter()

        try:
            async for chunk in llm.generate(
                messages=messages,
                tools=tools,
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

        # Delete the privacy system prompt after the turn so it doesn't affect future turns
        if use_privacy:
            messages.pop(-2)

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

        # Execute tool calls and let LLM react to results (max depth 5)
        if tool_calls:
            from reachy_mini_conversation_app.tools.core_tools import dispatch_tool_call, ToolDependencies
            from unittest.mock import MagicMock

            tool_deps = ToolDependencies(
                reachy_mini=MagicMock(),
                movement_manager=MagicMock(),
            )

            for depth in range(5):
                # Execute all tool calls
                for tc in tool_calls:
                    call_id, tool_name, arguments = llm.parse_tool_call(tc)
                    print(f"\n  [Executing: {tool_name}({json.dumps(arguments, ensure_ascii=False)})]")
                    try:
                        result = await dispatch_tool_call(tool_name, json.dumps(arguments), tool_deps)
                    except Exception as e:
                        result = {"error": str(e)}
                    display = json.dumps(result, ensure_ascii=False)
                    if len(display) > 300:
                        display = display[:300] + "..."
                    print(f"  [Result: {display}]")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": tool_name,
                        "content": json.dumps(result, ensure_ascii=False),
                    })

                # Re-invoke LLM to react to tool results (no new tools offered)
                print("\n  [Re-invoking LLM...]")
                assistant_text = ""
                tool_calls = []
                first_token = True
                ttft = None
                request_start = time.perf_counter()

                try:
                    async for chunk in llm.generate(
                        messages=messages,
                        tools=[],  # don't offer new tools on re-invocation
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
                except Exception as e:
                    print(f"\n[Error on re-invocation: {e}]")
                    break

                total_time = time.perf_counter() - request_start
                if not first_token:
                    ttft_str = f"{ttft * 1000:.0f}ms" if ttft else "N/A"
                    print(f"\n  [TTFT: {ttft_str} | Total: {total_time * 1000:.0f}ms]")

                # Append the follow-up assistant response
                if assistant_text:
                    messages.append({"role": "assistant", "content": assistant_text})
                elif tool_calls:
                    messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})

                # If LLM didn't make new tool calls, we're done
                if not tool_calls:
                    break


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive CLI chat for testing LLM providers",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="LLM provider name from cascade.yaml (default: read from .env CASCADE_LLM_PROVIDER)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--privacy",
        action="store_true",
        default=False,
        help="Enable privacy prompt (default: disabled)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    llm = create_provider(args.provider)
    tools = load_tools()
    asyncio.run(chat_repl(llm, tools, args.temperature, args.privacy))


if __name__ == "__main__":
    main()
