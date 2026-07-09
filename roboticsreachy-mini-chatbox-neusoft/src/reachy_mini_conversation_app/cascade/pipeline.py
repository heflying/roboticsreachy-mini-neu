"""LLM response processing and tool execution pipeline."""

from __future__ import annotations
import json
import base64
import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict, List, AsyncIterator
from pathlib import Path
from dataclasses import dataclass, field

from reachy_mini_conversation_app.cascade.llm import LLMProvider
from reachy_mini_conversation_app.cascade.tts import TTSProvider
from reachy_mini_conversation_app.cascade.config import get_config
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies, dispatch_tool_call
from reachy_mini_conversation_app.cascade.turn_result import TurnItem, PipelineResult


PROMPT_LOG = Path("prompt.log")


def _log_prompt(messages: list[dict[str, Any]], tools: list[dict[str, Any]], system: str | None, depth: int) -> None:
    """Append a human-readable snapshot of the LLM request to prompt.log."""
    import datetime

    lines: list[str] = []
    lines.append(f"\n{'='*80}")
    lines.append(f"LLM REQUEST  depth={depth}  {datetime.datetime.now().isoformat(timespec='milliseconds')}")
    lines.append(f"{'='*80}")

    # System instructions
    if system:
        lines.append(f"\n--- SYSTEM ({len(system)} chars) ---")
        lines.append(system)

    # Tools
    lines.append(f"\n--- TOOLS ({len(tools)}) ---")
    for t in tools:
        fn = t.get("function", t)
        lines.append(f"  - {fn.get('name', '?')}: {fn.get('description', '')[:120]}")

    # Messages
    lines.append(f"\n--- MESSAGES ({len(messages)}) ---")
    for i, msg in enumerate(messages):
        role = msg.get("role", "?")
        # Tool result message
        if role == "tool":
            content = msg.get("content", "")
            if len(content) > 300:
                content = content[:300] + "..."
            lines.append(f"[{i}] {role} ({msg.get('name','?')}): {content}")
        # Assistant with tool calls
        elif "tool_calls" in msg:
            text = msg.get("content", "") or ""
            tc_summary = ", ".join(
                tc.get("function", {}).get("name", "?") for tc in msg["tool_calls"]
            )
            lines.append(f"[{i}] {role}: {text[:200]}  [tool_calls: {tc_summary}]")
        # User with image
        elif isinstance(msg.get("content"), list):
            parts = []
            for p in msg["content"]:
                if isinstance(p, dict) and p.get("type") == "image":
                    parts.append("<image>")
                elif isinstance(p, dict) and p.get("type") == "text":
                    parts.append(p.get("text", "")[:200])
                else:
                    parts.append(str(p)[:200])
            lines.append(f"[{i}] {role}: {' | '.join(parts)}")
        else:
            content = str(msg.get("content", ""))
            if len(content) > 500:
                content = content[:500] + "..."
            lines.append(f"[{i}] {role}: {content}")

    lines.append("")

    with PROMPT_LOG.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))


if TYPE_CHECKING:
    from reachy_mini_conversation_app.cascade.speech_output import SpeechOutput
    from reachy_mini_conversation_app.cascade.interrupt_coordinator import TurnCancellationToken


logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """Bundle of references passed through the LLM/tool pipeline."""

    llm: LLMProvider
    tts: TTSProvider
    speech_output: SpeechOutput | None
    conversation_history: list[dict[str, Any]]
    tool_specs: list[dict[str, Any]]
    deps: ToolDependencies
    result: PipelineResult
    # Task 7: Turn-level cancellation support
    token: TurnCancellationToken | None = field(default=None)
    turn_id: int = field(default=0)


def _track_cost(ctx: PipelineContext, provider: Any) -> None:
    """Accumulate provider cost into the pipeline result."""
    if hasattr(provider, "last_cost") and provider.last_cost > 0:
        ctx.result.cost += provider.last_cost
        provider.last_cost = 0.0


def _turn_cancelled(ctx: PipelineContext) -> bool:
    """Whether the current turn has been cancelled."""
    return bool(ctx.token and ctx.token.cancelled)


def _rollback_history(ctx: PipelineContext, checkpoint: int) -> None:
    """Remove partial assistant/tool history emitted by a cancelled turn."""
    if len(ctx.conversation_history) > checkpoint:
        removed = len(ctx.conversation_history) - checkpoint
        del ctx.conversation_history[checkpoint:]
        logger.info("Rolled back %s history entries for cancelled turn %s", removed, ctx.turn_id)


def _mark_interrupted(ctx: PipelineContext) -> None:
    """Add a system message noting that the AI output was interrupted by the user."""
    ctx.conversation_history.append({"role": "system", "content": "[用户打断了AI的回复]"})
    logger.info("Added interruption marker for turn %s", ctx.turn_id)


async def process_llm_response(ctx: PipelineContext) -> PipelineResult:
    """Process LLM response with retry on failure."""
    max_retries = 2
    for attempt in range(1 + max_retries):
        try:
            await _process_llm_response_once(ctx)
            return ctx.result
        except Exception as e:
            if attempt < max_retries:
                logger.warning("LLM failed (attempt %d), retrying: %s", attempt + 1, e)
                if ctx.speech_output:
                    await ctx.speech_output.speak("Give me a moment.", token=ctx.token, turn_id=ctx.turn_id)
                await asyncio.sleep(2)
            else:
                logger.error("LLM failed after %d attempts: %s", max_retries + 1, e)
                if ctx.speech_output:
                    await ctx.speech_output.speak(
                        "Sorry, I'm having trouble responding right now.",
                        token=ctx.token,
                        turn_id=ctx.turn_id,
                    )
    return ctx.result


async def warmup_llm(ctx: PipelineContext, partial_text: str) -> None:
    """LLM warmup using provider's warmup method with full context.

    This calls ctx.llm.warmup() with the complete context (conversation_history
    + tools) to trigger prompt processing. The provider internally uses max_tokens=1
    to minimize cost while still loading the prompt into LLM memory/GPU.

    Args:
        ctx: PipelineContext with complete conversation_history, tool_specs, etc.
        partial_text: Current accumulated partial transcript for warmup.

    Note:
        This function does NOT modify conversation_history. The partial_text
        is only used for warmup and will be re-sent with the complete transcript
        after the user finishes speaking.
    """
    if not partial_text.strip():
        logger.debug("Skipping LLM warmup: empty partial text")
        return

    # Build temporary messages for warmup (not added to conversation_history)
    warmup_messages = ctx.conversation_history + [
        {"role": "user", "content": partial_text}
    ]

    logger.debug(f"LLM warmup started with partial: '{partial_text[:50]}...'")

    try:
        # Call LLM provider's warmup method
        await ctx.llm.warmup(
            messages=warmup_messages,
            tools=ctx.tool_specs,  # Embed tools in system prompt for KV cache warmup
            temperature=get_config().llm_temperature,
        )

        logger.debug("LLM warmup completed")

    except Exception as e:
        # Warmup failures should not block the main flow
        logger.warning(f"LLM warmup failed (non-critical): {e}")


async def process_streaming_dialog_response(ctx: PipelineContext) -> PipelineResult:
    """Stream direct LLM text into TTS, with tool call support (max 3 tool calls).

    Task 7: Token/turn_id integration for interrupt support.
    """
    from reachy_mini_conversation_app.cascade.timing import tracker
    from reachy_mini_conversation_app.cascade.quick_reply import get_quick_reply

    history_checkpoint = len(ctx.conversation_history)
    system = getattr(ctx.llm, "system_instructions", None)

    last_user_text = ""
    for message in reversed(ctx.conversation_history):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            last_user_text = message["content"]
            break

    quick_reply = get_quick_reply(last_user_text)
    if quick_reply:
        tracker.mark("llm_quick_reply", {"text_len": len(quick_reply)})
        tracker.mark("llm_complete", {"text_len": len(quick_reply), "tool_calls": 0, "quick_reply": True})
        if ctx.speech_output:
            await ctx.speech_output.speak(quick_reply, token=ctx.token, turn_id=ctx.turn_id)
        if _turn_cancelled(ctx):
            logger.info("Discarding cancelled quick reply for turn %s", ctx.turn_id)
            _rollback_history(ctx, history_checkpoint)
            _mark_interrupted(ctx)
            return ctx.result
        ctx.conversation_history.append({"role": "assistant", "content": quick_reply})
        ctx.result.turn_items.append(TurnItem(kind="speak", text=quick_reply))
        return ctx.result

    # Main loop: LLM generation + tool calls (max 3 iterations)
    for tool_depth in range(3):
        _log_prompt(ctx.conversation_history, ctx.tool_specs if tool_depth == 0 else [], system, tool_depth)

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        first_speech_chunk = True

        async def text_deltas() -> AsyncIterator[str]:
            nonlocal first_speech_chunk
            async for chunk in ctx.llm.generate(
                messages=ctx.conversation_history,
                tools=ctx.tool_specs if tool_depth == 0 else None,
                temperature=get_config().llm_temperature,
                token=ctx.token,
            ):
                if _turn_cancelled(ctx):
                    logger.info(f"LLM generation cancelled at turn {ctx.turn_id}")
                    break

                if chunk.type == "text_delta" and chunk.content:
                    text_parts.append(chunk.content)
                    if first_speech_chunk:
                        tracker.mark("llm_first_speech_chunk")
                        first_speech_chunk = False
                    yield chunk.content
                elif chunk.type == "tool_call" and chunk.tool_call:
                    tool_calls.append(chunk.tool_call)
                    logger.info(f"Tool call in streaming mode: {chunk.tool_call['function']['name']}")
                elif chunk.type == "done":
                    tracker.mark("llm_complete", {"text_len": len("".join(text_parts)), "tool_calls": len(tool_calls)})
                    break

        # Stream text to TTS
        spoken_text = ""
        if ctx.speech_output and hasattr(ctx.speech_output, "speak_stream"):
            spoken_text = await ctx.speech_output.speak_stream(
                text_deltas(),
                token=ctx.token,
                turn_id=ctx.turn_id,
            )
        else:
            async for _delta in text_deltas():
                pass
            spoken_text = "".join(text_parts)
            if ctx.speech_output and spoken_text:
                await ctx.speech_output.speak(spoken_text, token=ctx.token, turn_id=ctx.turn_id)

        # Add assistant message to history
        if text_parts or tool_calls:
            assistant_message: dict[str, Any] = {"role": "assistant"}
            if text_parts:
                assistant_message["content"] = "".join(text_parts)
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
            ctx.conversation_history.append(assistant_message)

        # If no tool calls, record spoken text and we're done
        if not tool_calls:
            if spoken_text:
                ctx.result.turn_items.append(TurnItem(kind="speak", text=spoken_text))
            break

        # Execute tool calls
        logger.info(f"Executing {len(tool_calls)} tool call(s) (depth={tool_depth + 1})")
        await execute_tool_calls(tool_calls, ctx)

        # Continue loop to let LLM react to tool results
        logger.info("Tool execution complete, continuing LLM generation...")

    _track_cost(ctx, ctx.llm)
    _track_cost(ctx, ctx.tts)

    if _turn_cancelled(ctx):
        logger.info("Discarding cancelled streaming dialog output for turn %s", ctx.turn_id)
        _rollback_history(ctx, history_checkpoint)
        _mark_interrupted(ctx)
        return ctx.result

    return ctx.result


async def _process_llm_response_once(ctx: PipelineContext, _depth: int = 0) -> None:
    """Single attempt at processing LLM response with streaming, tool calls, and TTS."""
    history_checkpoint = len(ctx.conversation_history)
    # Log the full prompt for debugging
    system = getattr(ctx.llm, "system_instructions", None)
    _log_prompt(ctx.conversation_history, ctx.tool_specs, system, _depth)

    # Generate streaming response
    text_chunks: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    async for chunk in ctx.llm.generate(
        messages=ctx.conversation_history,
        tools=ctx.tool_specs,
        temperature=get_config().llm_temperature,
        token=ctx.token,
    ):
        if _turn_cancelled(ctx):
            logger.info("LLM generation cancelled at turn %s", ctx.turn_id)
            _rollback_history(ctx, history_checkpoint)
            _mark_interrupted(ctx)
            return
        if chunk.type == "text_delta" and chunk.content:
            text_chunks.append(chunk.content)
            logger.debug(f"LLM text delta: {chunk.content}")

        elif chunk.type == "tool_call" and chunk.tool_call:
            tool_calls.append(chunk.tool_call)
            logger.info(f"LLM tool call: {chunk.tool_call}")

        elif chunk.type == "done":
            logger.debug("LLM generation complete")
            break

    # Aggregate LLM cost after generator completes
    _track_cost(ctx, ctx.llm)

    if _turn_cancelled(ctx):
        logger.info("Discarding cancelled LLM response for turn %s", ctx.turn_id)
        _rollback_history(ctx, history_checkpoint)
        _mark_interrupted(ctx)
        return

    # Create assistant message with text, tool calls...
    assistant_message: Dict[str, Any] = {"role": "assistant"}
    full_text = ""
    if text_chunks:
        full_text = "".join(text_chunks)
        assistant_message["content"] = full_text
    if tool_calls:
        assistant_message["tool_calls"] = tool_calls

    logger.debug(
        f"process_llm_response: text_chunks={len(text_chunks)}, tool_calls={len(tool_calls)}, full_text_len={len(full_text)}"
    )

    if text_chunks or tool_calls:
        ctx.conversation_history.append(assistant_message)
        logger.debug(f"Added assistant message to history, history_len={len(ctx.conversation_history)}")

    # Handle text-only responses: send text directly to TTS
    # No longer using the speak tool — LLM text goes directly to TTS.
    if full_text and not tool_calls:
        logger.info("LLM returned text — sending directly to TTS")
        if ctx.speech_output:
            await ctx.speech_output.speak(full_text, token=ctx.token, turn_id=ctx.turn_id)
        ctx.result.turn_items.append(TurnItem(kind="speak", text=full_text))
    elif tool_calls and full_text:
        # Tool calls with text — record assistant text
        ctx.result.turn_items.append(TurnItem(kind="assistant", text=full_text))
    if tool_calls:
        # Process normal tool calls
        await execute_tool_calls(tool_calls, ctx)

        # Re-invoke LLM so it can react to tool results (generate a verbal response)
        if _depth < 5:
            logger.info("Tool calls executed — re-invoking LLM to react to tool results")
            await _process_llm_response_once(ctx, _depth=_depth + 1)

    if _turn_cancelled(ctx):
        logger.info("Discarding cancelled tool/assistant state for turn %s", ctx.turn_id)
        _rollback_history(ctx, history_checkpoint)
        _mark_interrupted(ctx)


async def execute_tool_calls(
    tool_calls: list[dict[str, Any]],
    ctx: PipelineContext,
) -> None:
    """Execute tool calls and handle see_image_through_camera specially."""
    camera_image_bytes: bytes | None = None

    # First pass: execute all tools and add ALL tool results to conversation
    # This must be done before adding any other messages (OpenAI requires all tool
    # responses immediately after the assistant message with tool_calls)
    for tool_call in tool_calls:
        if _turn_cancelled(ctx):
            logger.info("Stopping tool execution for cancelled turn %s", ctx.turn_id)
            return
        call_id = ""
        tool_name = "unknown"
        try:
            call_id, tool_name, arguments = ctx.llm.parse_tool_call(tool_call)

            logger.info(f"Executing tool: {tool_name}({arguments})")

            # Execute tool
            result = await dispatch_tool_call(
                tool_name,
                json.dumps(arguments),
                ctx.deps,
            )

            if _turn_cancelled(ctx):
                logger.info("Discarding tool result for cancelled turn %s", ctx.turn_id)
                return

            # Do not log full result if the tool returned base64 (huge)
            if "b64_im" in result:
                logger.info("Tool result: [image in base64, not shown]")
            else:
                logger.info(f"Tool result: {result}")

            # Add tool result to conversation
            ctx.conversation_history.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_name,
                    "content": json.dumps(result),
                }
            )
            logger.debug(
                f"Added tool result to history: name={tool_name}, history_len={len(ctx.conversation_history)}"
            )

            # Special handling for see_image_through_camera - store frame, replace heavy b64
            if tool_name == "see_image_through_camera":
                if "b64_im" in result:
                    b64_im = result["b64_im"]
                    camera_image_bytes = base64.b64decode(b64_im)
                    frame_index = len(ctx.result.captured_frames)
                    ctx.result.captured_frames.append(camera_image_bytes)
                    # Replace the heavy b64 blob in conversation history with a lightweight marker
                    ctx.conversation_history[-1]["content"] = json.dumps(
                        {"status": "image_captured", "frame_index": frame_index}
                    )
                    ctx.result.turn_items.append(TurnItem(kind="image", image_jpeg=camera_image_bytes))
                    logger.info("see_image_through_camera: stored frame %d, will add image to conversation", frame_index)
                else:
                    logger.warning(f"see_image_through_camera returned error: {result}")

            # Other tools
            elif tool_name != "see_image_through_camera":
                ctx.result.turn_items.append(
                    TurnItem(kind="tool", tool_name=tool_name, tool_content=json.dumps(result))
                )

        except Exception as e:
            logger.exception(f"Error executing tool {tool_name}: {e}")

            # Add error to conversation
            ctx.conversation_history.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_name,
                    "content": json.dumps({"error": str(e)}),
                }
            )

    # After all tool results are added, add camera image as user message and call LLM
    if camera_image_bytes is not None:
        if _turn_cancelled(ctx):
            logger.info("Skipping camera follow-up for cancelled turn %s", ctx.turn_id)
            return
        ctx.conversation_history.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": camera_image_bytes,  # Will be converted to provider format in LLM
                    }
                ],
            }
        )
        logger.info("Camera image added to conversation - calling LLM to analyze it")
        await process_llm_response(ctx)
