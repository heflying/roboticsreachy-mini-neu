"""Main cascade handler orchestrating ASR → LLM → TTS pipeline."""

from __future__ import annotations
import os
import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Union

from reachy_mini_conversation_app.cascade import pipeline
from reachy_mini_conversation_app.cascade.asr import ASRProvider, StreamingASRProvider
from reachy_mini_conversation_app.cascade.llm import LLMProvider
from reachy_mini_conversation_app.cascade.tts import TTSProvider
from reachy_mini_conversation_app.cascade.config import get_config
from reachy_mini_conversation_app.cascade.pipeline import PROMPT_LOG, PipelineContext
from reachy_mini_conversation_app.tools.core_tools import (
    ToolDependencies,
    get_tool_specs,
)
from reachy_mini_conversation_app.cascade.turn_result import TurnItem, TurnResult, PipelineResult
from reachy_mini_conversation_app.cascade.provider_factory import (
    init_asr_provider,
    init_llm_provider,
    init_tts_provider,
    init_transcript_analysis,
)
from reachy_mini_conversation_app.cascade.transcript_analysis import (
    NoOpTranscriptManager,
    TranscriptAnalysisManager,
)
from reachy_mini_conversation_app.cascade.turn_controller import TurnController
from reachy_mini_conversation_app.cascade.interrupt_coordinator import TurnCancellationToken
from reachy_mini_conversation_app.cascade.router import CompositeRouter
from reachy_mini_conversation_app.scheduler.engine import Scheduler
from reachy_mini_conversation_app.scheduler.store import SchedulerStore
from reachy_mini_conversation_app.scheduler.models import AlertEvent
from reachy_mini_conversation_app.tools.set_alarm import set_scheduler
from reachy_mini_conversation_app.proactive.engine import (
    DecisionEngine, RobotState, DecisionResult
)


if TYPE_CHECKING:
    from reachy_mini_conversation_app.cascade.speech_output import SpeechOutput


logger = logging.getLogger(__name__)


def convert_tool_specs_to_chat_format(realtime_specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert tool specs from Realtime API format to Chat Completions API format."""
    chat_specs = []
    for spec in realtime_specs:
        if spec["type"] == "function":
            chat_spec = {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": spec["parameters"],
                },
            }
            chat_specs.append(chat_spec)
    return chat_specs


class CascadeHandler:
    """Main handler for cascade pipeline mode."""

    def __init__(self, deps: ToolDependencies):
        """Initialize cascade handler."""
        self.deps = deps

        # Speech output backend (set by console or Gradio frontend)
        self.speech_output: SpeechOutput | None = None

        # Initialize providers based on config
        self.asr = init_asr_provider()
        self.llm = init_llm_provider()
        self.tts = init_tts_provider()
        self.router = CompositeRouter()

        # Conversation state
        self.conversation_history: List[Dict[str, Any]] = []
        self.processing_lock = asyncio.Lock()
        self.running = False

        # Event loop for async operations
        self.loop: asyncio.AbstractEventLoop | None = None
        self.loop_thread: threading.Thread | None = None

        # Track last partial transcript to avoid log spam
        self._last_partial_transcript = ""

        # Store streaming status based on config
        self.is_streaming_asr = get_config().is_asr_streaming()

        # Dynamic tool gating based on available capabilities
        exclusion_list: list[str] = []
        if deps.vision_manager is None:
            exclusion_list.append("describe_camera_image")

        # Get tool specs and convert to Chat Completions format
        # Note : get_tool_specs() returns Realtime API format, so we need Chat Completions format
        raw_tool_specs = get_tool_specs(exclusion_list=exclusion_list)
        if os.getenv("CASCADE_DIALOG_ONLY_TOOLS", "1").strip().lower() not in {"0", "false", "no", "off"}:
            # Dialog-only mode: no tools exposed — LLM text goes directly to TTS
            raw_tool_specs = []
            logger.info("Cascade dialog-only tool mode enabled: no tools exposed (text → TTS directly)")
        self.tool_specs = convert_tool_specs_to_chat_format(raw_tool_specs)
        tool_names = {
            spec.get("function", {}).get("name")
            for spec in self.tool_specs
            if isinstance(spec.get("function"), dict)
        }
        streaming_dialog_requested = (
            os.getenv("CASCADE_STREAMING_DIALOG", "1").strip().lower() not in {"0", "false", "no", "off"}
        )
        self.streaming_dialog_enabled = streaming_dialog_requested
        if self.streaming_dialog_enabled:
            logger.info("Cascade streaming dialog mode enabled: speech is managed by the pipeline (tools=%d)", len(tool_names))

        # Side-channel storage for see_image_through_camera frames (JPEG bytes, indexed)
        self._captured_frames: list[bytes] = []

        # Transcript analysis (NoOp if no reactions configured)
        self.transcript_manager: TranscriptAnalysisManager | NoOpTranscriptManager = (
            init_transcript_analysis(deps)
        )

        # Cost tracking
        self.cumulative_cost: float = 0.0
        self._turn_cost: float = 0.0

        # Turn result tracking
        self._current_turn_items: list[TurnItem] = []
        self._turn_results: list[TurnResult] = []

        # Turn controller for interrupt handling
        self._turn_controller: TurnController | None = None

        # Sentence pause warmup state
        self._accumulated_partial: str = ""  # Accumulated partial transcript from sentence pauses
        self._warmup_task: asyncio.Task[None] | None = None  # Current warmup task

        # Scheduler (alarm/calendar) integration
        self.scheduler = Scheduler(store=SchedulerStore())
        set_scheduler(self.scheduler)  # Make scheduler available to tools
        self._pending_alerts: list[AlertEvent] = []  # Alerts pending processing

        # ── State management (dialogue / active / resting) ──────────
        # 初始状态设为 DIALOGUE，启动即进入对话态等待用户输入
        self._robot_state: RobotState = RobotState.DIALOGUE
        self._dialogue_timeout_task: asyncio.Task[None] | None = None  # 60s timeout in dialogue state
        self._active_to_resting_task: asyncio.Task[None] | None = None  # 120s timeout in active state
        self._wobble_task: asyncio.Task[None] | None = None  # Active state wobble
        self._last_user_speak_time: float = 0.0

        # Proactive decision engine (non-dialogue states)
        self.decision_engine = DecisionEngine(self.scheduler.alert_queue)

        # VAD reset callback — set by GradioApp during initialization.
        # When entering DIALOGUE state, handler calls this to reset the
        # ContinuousVADRecorder's VADStateMachine for clean speech detection.
        self._vad_reset_callback: Callable[[], None] | None = None

        logger.info(f"Cascade handler initialized (streaming_asr={self.is_streaming_asr})")

    # ─────────────────────────────────────────────────────────────────────────────
    # Transcript Analysis Helpers (fire-and-forget, never block pipeline)
    # ─────────────────────────────────────────────────────────────────────────────

    def _get_stable_text(self, partial: str) -> str:
        """Get stable text for analysis (if ASR supports it)."""
        if hasattr(self.asr, "get_stable_text"):
            stable = self.asr.get_stable_text()
            if stable and stable != partial:
                logger.debug(f"📌 Using stable text for analysis: '{stable[:60]}...'")
                return stable  # type: ignore[no-any-return]
        return partial

    async def _on_transcript_partial(self, text: str) -> None:
        """Notify partial transcript for real-time reactions (streaming only)."""
        await self.transcript_manager.analyze_partial(text)

    def _on_transcript_final(self, text: str) -> None:
        """Notify final transcript (fire-and-forget, parallel with LLM)."""
        task = asyncio.create_task(self.transcript_manager.analyze_final(text))
        if hasattr(self.transcript_manager, '_pending_tasks'):
            self.transcript_manager._pending_tasks.append(task)

    def _on_turn_complete(self) -> None:
        """Reset transcript analysis between conversation turns."""
        self.transcript_manager.reset()

    # ─────────────────────────────────────────────────────────────────────────────
    # Sentence Pause Warmup (for reduced TTFB)
    # ─────────────────────────────────────────────────────────────────────────────

    def on_sentence_pause(self, partial_text: str | None = None) -> None:
        """Handle sentence pause event detected by VAD.

        When VAD detects a silence gap >= sentence_pause_threshold_ms during speech,
        this method is called. It uses the supplied partial transcript or the
        latest cached ASR partial transcript, then triggers LLM warmup asynchronously.

        Note:
            This method may be called from the VAD recording thread (non-async).
            It schedules the warmup task on the event loop.
        """
        partial = partial_text or self._last_partial_transcript

        if not partial.strip():
            logger.debug("[WARMUP] Ignoring sentence pause: empty partial transcript")
            return

        # Save accumulated partial for later concatenation
        self._accumulated_partial = partial

        # Trigger warmup asynchronously (fire-and-forget)
        if self.loop:
            asyncio.run_coroutine_threadsafe(
                self._trigger_warmup(partial),
                self.loop
            )

    async def _trigger_warmup(self, partial_text: str) -> None:
        """Trigger LLM warmup with the accumulated partial transcript.

        Args:
            partial_text: Current partial transcript for warmup.

        Note:
            This runs asynchronously in the background. Failures are logged
            but do not block the main pipeline flow.

            Sentence-level warmup is only effective for local LLM providers
            (location=local) that can reuse KV cache across requests. Cloud
            providers have no cache reuse benefit and are skipped automatically.
        """
        from reachy_mini_conversation_app.cascade.config import get_config

        cfg = get_config()
        if not cfg.enable_sentence_warmup:
            logger.debug("Sentence warmup disabled, skipping")
            return

        # Only local LLM providers benefit from sentence warmup
        llm_info = cfg.get_llm_provider_info()
        if llm_info.get("location") != "local":
            logger.debug("Sentence warmup skipped: LLM provider is not local (location=%s)", llm_info.get("location"))
            return

        # Cancel any existing warmup task
        if self._warmup_task and not self._warmup_task.done():
            self._warmup_task.cancel()
            logger.debug("Cancelled previous warmup task")

        # Create new warmup task
        self._warmup_task = asyncio.create_task(self._execute_warmup(partial_text))

    async def _execute_warmup(self, partial_text: str) -> None:
        """Execute LLM warmup with full context.

        Args:
            partial_text: Partial transcript to use for warmup.
        """
        try:
            logger.info(f"[WARMUP] Starting LLM warmup with partial: '{partial_text[:50]}...'")

            ctx = PipelineContext(
                llm=self.llm,
                tts=self.tts,
                speech_output=None,  # warmup doesn't need TTS
                conversation_history=self.conversation_history,
                tool_specs=self.tool_specs,
                deps=self.deps,
                result=PipelineResult(),
                token=None,  # warmup doesn't need cancellation
                turn_id=0,
            )

            await pipeline.warmup_llm(ctx, partial_text)
            logger.info("[WARMUP] Completed successfully")

        except asyncio.CancelledError:
            logger.debug("[WARMUP] Task cancelled (likely superseded by newer partial)")
        except Exception as e:
            logger.warning(f"[WARMUP] Failed (non-critical): {e}")

    async def _fence_pending_warmup(self) -> None:
        """Fence pending warmup before the final LLM request.

        Current policy uses a 0ms wait to avoid adding user-visible latency.
        If warmup is still running, it is cancelled immediately.
        """
        if self._warmup_task is None:
            return

        warmup_task = self._warmup_task
        if warmup_task.done():
            self._warmup_task = None
            return

        warmup_wait_timeout_s = 0.0
        try:
            await asyncio.wait_for(asyncio.shield(warmup_task), timeout=warmup_wait_timeout_s)
        except asyncio.TimeoutError:
            logger.debug("[WARMUP] Final fence timeout reached, cancelling pending warmup task")
            warmup_task.cancel()
            try:
                await warmup_task
            except asyncio.CancelledError:
                pass
        finally:
            self._warmup_task = None

    @property
    def turn_results(self) -> list[TurnResult]:
        """Completed conversation turns (read by UI poller)."""
        return self._turn_results

    def _aggregate_cost(self, provider: Union[ASRProvider, LLMProvider, TTSProvider], provider_name: str) -> None:
        """Aggregate cost from a provider if it tracks costs."""
        if hasattr(provider, "last_cost") and provider.last_cost > 0:
            cost = provider.last_cost
            self.cumulative_cost += cost
            self._turn_cost += cost
            logger.info(f"Cost ({provider_name}): ${cost:.4f} | Cumulative: ${self.cumulative_cost:.4f}")
            provider.last_cost = 0.0  # Reset for next call

    async def _run_pipeline_after_transcription(self, transcript: str) -> TurnResult:
        """Run the shared post-ASR pipeline: validate → history → LLM → TTS → result.

        Called by both manual and streaming paths after transcription is obtained.
        Caller must hold self.processing_lock.

        Task 7: Token/turn_id integration for interrupt support.
        """
        from reachy_mini_conversation_app.cascade.timing import tracker

        if not transcript.strip():
            logger.warning("Empty transcript, ignoring")
            if self.deps.movement_manager:
                self.deps.movement_manager.set_listening(False)
            return TurnResult()

        # Reset tracker at the actual turn boundary (not at VAD speech_start).
        # Preserve vad_speech_end so TTFB calculation remains accurate — the
        # event was marked in the VAD recording loop before we got here, and
        # reset() would otherwise discard it.
        tracker.preserve_event("vad_speech_end")
        # Preserve ASR critical-path events so trace formula works after barge-in.
        # Without these, reset() clears B4 events and G1 becomes uncomputable,
        # causing >70% deviation in TTFB decomposition.
        tracker.preserve_event("asr_local_ready")           # local ASR detection flag
        tracker.preserve_event("asr_local_final_decode")    # local ASR B4_start
        tracker.preserve_event("asr_commit_sent")           # cloud ASR B4_start
        tracker.preserve_event("asr_result_delivered")      # B4_end / G1_start
        tracker.reset("user_conversation_turn")

        # Task 7: Start new turn and get token/turn_id for interrupt isolation
        turn_id = 0
        token = None
        if self._turn_controller is not None:
            turn_id, token = self._turn_controller.start_new_turn()
            logger.info(f"Turn {turn_id} started for pipeline processing")

        # Sync decision engine state (already DIALOGUE via state machine loop)
        logger.info("[STATE] Processing user input in DIALOGUE state")
        self.decision_engine.set_state(RobotState.DIALOGUE)

        # Inject deferred alerts from proactive engine (accumulated during active/resting)
        deferred_text = self.decision_engine.get_deferred_alerts_formatted()
        if deferred_text:
            self.conversation_history.append({"role": "system", "content": deferred_text})
            logger.info("Injected %d deferred alert(s) into conversation history", deferred_text.count("[系统提醒]"))

        # Add user message to history
        self.conversation_history.append({"role": "user", "content": transcript})

        # # Router: determine routing decision based on user input and conversation history
        # route_result = await self.router.route(self.conversation_history)
        # logger.info("Router decision: %s", route_result.decision)

        # Update robot state - done listening
        if self.deps.movement_manager:
            self.deps.movement_manager.set_listening(False)

        # Analyze final transcript (parallel with LLM, fire-and-forget)
        self._on_transcript_final(transcript)

        # TTS pre-connect moved to speech_output.py (delayed until LLM first token)
        # This reduces the pre-connect-to-use window from ~10s to ~0.5s, avoiding stale connections

        # LLM: Text → Response + Tool Calls
        logger.info("Generating LLM response...")
        tracker.mark("llm_start")
        ctx = PipelineContext(
            llm=self.llm, tts=self.tts, speech_output=self.speech_output,
            conversation_history=self.conversation_history,
            tool_specs=self.tool_specs,
            deps=self.deps,
            result=PipelineResult(),
            # Task 7: Pass token and turn_id for interrupt isolation
            token=token,
            turn_id=turn_id,
        )
        if self.streaming_dialog_enabled:
            result = await pipeline.process_streaming_dialog_response(ctx)
        else:
            result = await pipeline.process_llm_response(ctx)
            tracker.mark("llm_complete")

        # Apply pipeline outputs to handler state
        self._current_turn_items.extend(result.turn_items)
        self._captured_frames.extend(result.captured_frames)
        self._turn_cost += result.cost
        self.cumulative_cost += result.cost

        # Reset transcript analysis for next turn
        self._on_turn_complete()

        # Build and store TurnResult
        turn = TurnResult(
            transcript=transcript,
            items=list(self._current_turn_items),
            cost=self._turn_cost,
        )
        self._turn_results.append(turn)

        # After turn completes, process any pending alerts (dialogue state)
        await self._process_pending_alerts()

        # Reset dialogue timeout for next turn (don't end dialogue yet;
        # VAD will continue listening for user's next utterance in DIALOGUE)
        self._reset_dialogue_timeout()

        return turn

    # ────────────────────────────────────────────────────────────────────
    # State Management (active / resting / dialogue)
    # ────────────────────────────────────────────────────────────────────

    async def _state_machine_loop(self) -> None:
        """Flat state machine loop managing all three states.

        ACTIVE → RESTING (timeout) or DIALOGUE (alert trigger)
        RESTING → DIALOGUE (urgent alert trigger)
        DIALOGUE → ACTIVE (dialogue ended event)

        VAD audio processing is gated by DIALOGUE state via is_dialogue_active
        callback in ContinuousVADRecorder. Non-dialogue states route audio
        frames to proactive modules via on_audio_frame callback.
        """
        logger.info("State machine loop started")
        while self.running:
            state = self._robot_state
            if state == RobotState.ACTIVE:
                result = await self._run_active_state()
                if result:
                    self._robot_state = RobotState.DIALOGUE
                    await self._run_dialogue_state(result)
                    self._robot_state = RobotState.ACTIVE
            elif state == RobotState.RESTING:
                result = await self._run_resting_state()
                if result:
                    self._robot_state = RobotState.DIALOGUE
                    await self._run_dialogue_state(result)
                    self._robot_state = RobotState.ACTIVE
            elif state == RobotState.DIALOGUE:
                # DIALOGUE state: run dialogue flow (initial state or externally triggered)
                logger.info("[STATE] Entering DIALOGUE state")
                await self._run_dialogue_state(None)
                self._robot_state = RobotState.ACTIVE
            else:
                logger.error("Unknown state: %s", state)
                break

    async def _run_active_state(self) -> DecisionResult | None:
        """ACTIVE: wobble + proactive loop -> return result when should dialogue."""
        logger.info("[STATE] Entering ACTIVE state")
        self._robot_state = RobotState.ACTIVE
        self.decision_engine.set_state(RobotState.ACTIVE)

        # Start wobble (background task)
        self._wobble_task = asyncio.create_task(self._wobble_loop())

        # Start 120s timer: active -> resting
        self._reset_active_to_resting_timer()

        # Run proactive loop (blocks until start_dialog=True)
        # When it returns, we should enter dialogue
        try:
            # We need to run decision_engine.run() but also allow
            # the active->resting timer to fire and change state.
            # So run decision_engine.run() as a task, and wait for either
            # it to complete, or the state to change to RESTING.
            run_task = asyncio.create_task(
                self.decision_engine.run(RobotState.ACTIVE)
            )
            state_check_task = asyncio.create_task(
                self._wait_for_state_change(RobotState.ACTIVE)
            )

            done, pending = await asyncio.wait(
                [run_task, state_check_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel the loser
            for t in pending:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass

            if run_task in done:
                result = run_task.result()
                # Cancel state check task
                state_check_task.cancel()
                try:
                    await state_check_task
                except asyncio.CancelledError:
                    pass
                return result
            else:
                # State changed to RESTING
                run_task.cancel()
                try:
                    await run_task
                except asyncio.CancelledError:
                    pass
                return None  # Loop will pick up RESTING state

        finally:
            self._cancel_state_tasks()

    async def _run_resting_state(self) -> DecisionResult | None:
        """RESTING: hold still + proactive loop -> return result."""
        logger.info("[STATE] Entering RESTING state")
        self._robot_state = RobotState.RESTING
        self.decision_engine.set_state(RobotState.RESTING)

        # Cancel wobble
        if self._wobble_task and not self._wobble_task.done():
            self._wobble_task.cancel()
            try:
                await self._wobble_task
            except asyncio.CancelledError:
                pass

        await self._hold_resting_pose()

        # Run proactive loop
        result = await self.decision_engine.run(RobotState.RESTING)
        self._cancel_state_tasks()
        return result

    async def _wait_for_state_change(self, current_state: RobotState) -> None:
        """Wait until self._robot_state != current_state."""
        while self._robot_state == current_state:
            await asyncio.sleep(0.5)

    async def _run_dialogue_state(self, result: DecisionResult) -> None:
        """DIALOGUE: inject message, run LLM, wait for end.

        On entry, resets VAD state machine to ensure clean speech detection
        (clears stale buffers from non-dialogue period). The reset callback
        is bound to the ContinuousVADRecorder during GradioApp initialization.
        """
        logger.info("[STATE] Entering DIALOGUE state (proactive trigger: %s)", result.decision if result else "user")
        self._robot_state = RobotState.DIALOGUE
        self.decision_engine.set_state(RobotState.DIALOGUE)

        # Reset VAD state machine at dialogue entry (clear stale buffers)
        self._reset_vad()

        # Start 60s dialogue timeout timer
        self._reset_dialogue_timeout()

        # Start alert listener (consumes alert_queue during dialogue state)
        self._alert_task = asyncio.create_task(self._alert_listener())

        # Inject system message from DecisionResult
        if result.system_message:
            self.conversation_history.append({
                "role": "system",
                "content": result.system_message,
            })
            logger.info("Injected proactive system message into conversation history")

        # Run LLM pipeline (robot speaks proactively)
        await self._proactive_llm_turn()

        # After speaking, wait for user input or 60s timeout
        # The timeout coroutine will set self._robot_state = ACTIVE
        # and cancel the current processing.
        # For now, just wait for the timeout event.
        if self._dialogue_ended_event:
            await self._dialogue_ended_event.wait()

        # Clean up dialogue state
        self._cancel_dialogue_tasks()
        self._dialogue_ended_event = None  # Clear for next DIALOGUE session
        logger.info("Dialogue state ended")

    async def _on_dialogue_ended(self) -> None:
        """Signal that dialogue has ended (called by timeout or user ending)."""
        if hasattr(self, '_dialogue_ended_event') and self._dialogue_ended_event:
            self._dialogue_ended_event.set()
        logger.info("[STATE] Dialogue ended, transitioning to ACTIVE state")
        self._robot_state = RobotState.ACTIVE

    def _cancel_state_tasks(self) -> None:
        """Cancel active/resting state tasks."""
        for task_name in ("_active_to_resting_task", "_wobble_task"):
            task = getattr(self, task_name, None)
            if task and not task.done():
                task.cancel()

    def _cancel_dialogue_tasks(self) -> None:
        """Cancel dialogue state tasks."""
        if self._dialogue_timeout_task and not self._dialogue_timeout_task.done():
            self._dialogue_timeout_task.cancel()
        if self._alert_task and not self._alert_task.done():
            self._alert_task.cancel()

    def _reset_vad(self) -> None:
        """Reset VAD state machine via callback (for DIALOGUE entry).

        Calls the VAD reset callback set by GradioApp, which resets the
        ContinuousVADRecorder's VADStateMachine. Thread-safe: the callback
        modifies the recorder's _vad_sm attribute (atomic in Python).
        """
        if self._vad_reset_callback:
            try:
                self._vad_reset_callback()
            except Exception as e:
                logger.warning("[VAD] Reset callback failed: %s", e)
        else:
            logger.debug("[VAD] No reset callback set, skipping VAD reset")

    def _reset_dialogue_timeout(self) -> None:
        """Reset the dialogue timeout timer.

        Cancels the existing timeout task and creates a new one.
        The _dialogue_ended_event is created only on first call (when
        entering DIALOGUE state) and reused across multiple turns within
        the same dialogue session.
        """
        if self._dialogue_timeout_task and not self._dialogue_timeout_task.done():
            self._dialogue_timeout_task.cancel()
        self._dialogue_timeout_task = asyncio.create_task(
            self._dialogue_timeout_coro()
        )
        # Create event only on first call (for dialogue entry)
        # Subsequent resets (multi-turn dialogue) reuse the same event
        if not hasattr(self, '_dialogue_ended_event') or self._dialogue_ended_event is None:
            self._dialogue_ended_event = asyncio.Event()

    async def _dialogue_timeout_coro(self) -> None:
        """60s timeout in dialogue state: no user speech -> end dialogue."""
        await asyncio.sleep(1000000000)
        if self._robot_state == RobotState.DIALOGUE:
            logger.info("Dialogue timeout (60s no user speech), ending dialogue")
            await self._on_dialogue_ended()

    def _reset_active_to_resting_timer(self) -> None:
        """Reset the 120s active->resting timer."""
        if self._active_to_resting_task and not self._active_to_resting_task.done():
            self._active_to_resting_task.cancel()
        self._active_to_resting_task = asyncio.create_task(
            self._active_to_resting_coro()
        )

    async def _active_to_resting_coro(self) -> None:
        """120s in active state with no dialogue -> transition to resting."""
        await asyncio.sleep(30)
        if self._robot_state == RobotState.ACTIVE:
            logger.info("Active->Resting timeout (120s no dialogue)")
            self._robot_state = RobotState.RESTING
            # State machine loop will pick up RESTING state

    async def _wobble_loop(self) -> None:
        """Wobble gently in active state (stub)."""
        try:
            while self._robot_state == RobotState.ACTIVE:
                logger.debug("Wobbling...")
                await asyncio.sleep(5)
        except asyncio.CancelledError:
            logger.debug("Wobble loop cancelled")
        except Exception as e:
            logger.exception("Wobble loop error: %s", e)

    async def _hold_resting_pose(self) -> None:
        """Hold a still pose in resting state (stub)."""
        logger.debug("Holding resting pose")
        await asyncio.sleep(0.1)


    async def process_audio_manual(self, audio_bytes: bytes) -> TurnResult:
        """Process recorded audio through the cascade pipeline.

        Called manually from Gradio UI.

        Args:
            audio_bytes: WAV audio bytes from Gradio recording

        Returns:
            TurnResult with transcript, displayable items, and cost

        """
        from reachy_mini_conversation_app.cascade.timing import tracker

        # Note: tracker.reset() is called in gradio_ui._stop_recording()
        # to capture user_stop_click in the same timeline

        # Reset per-turn state
        self._current_turn_items = []
        self._turn_cost = 0.0

        async with self.processing_lock:
            try:
                # Reset dialogue timeout (user is speaking)
                if self._robot_state == RobotState.DIALOGUE:
                    self._reset_dialogue_timeout()

                # Update robot state - user is speaking
                if self.deps.movement_manager:
                    self.deps.movement_manager.set_listening(True)

                # ASR: Audio → Text
                logger.info("Transcribing...")
                tracker.mark("transcribing_start")
                transcript = await self.asr.transcribe(audio_bytes, language="en")
                tracker.mark("asr_complete", {"transcript_len": len(transcript)})
                self._aggregate_cost(self.asr, "ASR")
                logger.info(f"User said: {transcript}")

                return await self._run_pipeline_after_transcription(transcript)

            except Exception as e:
                logger.exception(f"Error processing audio: {e}")
                if self.deps.movement_manager:
                    self.deps.movement_manager.set_listening(False)
                raise

    async def process_audio_streaming_start(self) -> None:
        """Initialize streaming ASR session.

        Called from Gradio UI when user starts recording with a streaming ASR provider.
        """
        if isinstance(self.asr, StreamingASRProvider):
            logger.info("Starting streaming ASR session")
            await self.asr.start_stream()

            # Update robot state - user is about to speak
            if self.deps.movement_manager:
                self.deps.movement_manager.set_listening(True)
        else:
            logger.warning("ASR provider does not support streaming")

    async def prepare_audio_streaming_session(self) -> None:
        """Pre-connect streaming ASR when the UI enters listening mode."""
        if isinstance(self.asr, StreamingASRProvider) and hasattr(self.asr, "prepare_stream"):
            logger.info("Pre-connecting streaming ASR session")
            await self.asr.prepare_stream()  # type: ignore[attr-defined]

    async def prepare_tts_session(self) -> None:
        """Pre-connect streaming TTS so the next response avoids setup latency."""
        if hasattr(self.tts, "prepare_stream"):
            logger.info("Pre-connecting streaming TTS session")
            await self.tts.prepare_stream()  # type: ignore[attr-defined]

    async def process_audio_streaming_chunk(self, chunk: bytes) -> str | None:
        """Send audio chunk to streaming ASR and get partial transcript.

        Called from Gradio UI during recording to stream audio in real-time.

        Args:
            chunk: Audio chunk bytes (WAV format)

        Returns:
            Partial transcript if available, None otherwise

        """
        if isinstance(self.asr, StreamingASRProvider):
            await self.asr.send_audio_chunk(chunk)
            partial = await self.asr.get_partial_transcript()

            # Log partial transcript (debounced to reduce spam)
            if partial and partial != self._last_partial_transcript:
                logger.info(f"🎤 Partial: {partial}")
                self._last_partial_transcript = partial

            # Analyze partial transcript (debounced, fire-and-forget)
            if partial:
                # Use stable text for entity extraction to avoid noisy draft tokens
                stable_text = self._get_stable_text(partial)
                await self._on_transcript_partial(stable_text)

            return partial
        return None

    async def process_audio_streaming_end(self) -> TurnResult:
        """Finalize streaming session, get final transcript, and run LLM pipeline.

        Called from Gradio UI when user stops recording with a streaming ASR provider.

        Returns:
            TurnResult with transcript, displayable items, and cost

        """
        from reachy_mini_conversation_app.cascade.timing import tracker

        # Reset per-turn state
        self._current_turn_items = []
        self._turn_cost = 0.0

        async with self.processing_lock:
            try:
                # Reset dialogue timeout (user is speaking)
                if self._robot_state == RobotState.DIALOGUE:
                    self._reset_dialogue_timeout()

                # Get final transcript from streaming ASR
                if isinstance(self.asr, StreamingASRProvider):
                    logger.info("Finalizing streaming ASR session")
                    tracker.mark("transcribing_start")
                    transcript = await self.asr.end_stream()
                    tracker.mark("asr_complete", {"transcript_len": len(transcript)})
                    self._aggregate_cost(self.asr, "ASR")
                else:
                    logger.warning("ASR provider does not support streaming, this shouldn't happen")
                    return TurnResult()

                await self._fence_pending_warmup()

                # end_stream() returns a full re-transcription of all audio,
                # so we should NOT concatenate _accumulated_partial (which is
                # already part of that full transcript). Only keep it as a
                # fallback if end_stream returns an empty or much shorter result.
                if self._accumulated_partial and not transcript:
                    transcript = self._accumulated_partial
                    logger.info(f"[WARMUP] Used accumulated partial as fallback: '{transcript[:30]}...'")
                self._accumulated_partial = ""  # Reset for next turn

                logger.info(f"User said: {transcript}")

                turn = await self._run_pipeline_after_transcription(transcript)

                # Reset partial transcript tracking (streaming-specific)
                self._last_partial_transcript = ""

                return turn

            except Exception as e:
                logger.exception(f"Error processing streaming audio: {e}")
                if self.deps.movement_manager:
                    self.deps.movement_manager.set_listening(False)
                raise

    def _run_event_loop(self, ready: threading.Event) -> None:
        """Run the asyncio event loop in a background thread."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.call_soon(ready.set)
        logger.debug("Event loop started in background thread")

        # Set event loop for TurnController (cross-thread interrupt support)
        if self._turn_controller is not None:
            self._turn_controller.set_event_loop(self.loop)

        try:
            self.loop.run_forever()
        finally:
            self.loop.close()

    def start(self) -> None:
        """Start the cascade handler (Gradio mode)."""
        if self.running:
            logger.warning("Cascade handler already running")
            return

        logger.info("Starting cascade handler (Gradio mode)...")
        self.running = True

        # Reset prompt log for this run
        PROMPT_LOG.write_text("", encoding="utf-8")

        # Start event loop in background thread for async operations
        loop_ready = threading.Event()
        self.loop_thread = threading.Thread(target=self._run_event_loop, args=(loop_ready,), daemon=True)
        self.loop_thread.start()
        loop_ready.wait(timeout=5)

        # Start scheduler
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.scheduler.start(), self.loop)
            logger.info("Scheduler started")

        # Start state machine loop (active → resting → dialogue → active ...)
        if self.loop:
            asyncio.run_coroutine_threadsafe(self._state_machine_loop(), self.loop)

        # Warmup LLM connection (with tools so KV cache prefix is warmed up)
        if self.loop:
            logger.info("Pre-warming LLM connection...")
            asyncio.run_coroutine_threadsafe(self.llm.warmup(tools=self.tool_specs), self.loop)
            logger.info("Pre-warming TTS...")
            asyncio.run_coroutine_threadsafe(self.tts.warmup(), self.loop)

        logger.info("Cascade handler started")

    def stop(self) -> None:
        """Stop the cascade handler."""
        if not self.running:
            return

        logger.info("Stopping cascade handler...")
        self.running = False

        # Cancel state management tasks
        for task_name in ("_dialogue_timeout_task", "_active_to_resting_task", "_wobble_task"):
            task = getattr(self, task_name, None)
            if task and not task.done():
                task.cancel()

        # Stop scheduler
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.scheduler.stop(), self.loop)

        # Stop event loop
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)

        if self.loop_thread:
            self.loop_thread.join(timeout=5)

        logger.info("Cascade handler stopped")

    def clear_state(self) -> None:
        """Reset all conversation and turn state (called from UI clear button)."""
        self.conversation_history.clear()
        self._captured_frames.clear()
        self._current_turn_items.clear()
        self._turn_results.clear()

    # ─────────────────────────────────────────────────────────────────────────────
    # Turn Controller Integration (Task 7)
    # ─────────────────────────────────────────────────────────────────────────────────────

    def init_turn_controller(self, audio_playback: Any) -> None:
        """Initialize TurnController (called after Gradio UI setup).

        Args:
            audio_playback: AudioPlaybackSystem instance from Gradio UI.

        Note:
            This should be called after the Gradio UI has created the
            AudioPlaybackSystem. The TurnController manages turn-level
            lifecycle and interrupt coordination.
        """
        self._turn_controller = TurnController(self, audio_playback)
        logger.info("TurnController initialized")

    def start_new_turn(self) -> tuple[int, TurnCancellationToken] | None:
        """Start a new turn (if turn controller is initialized).

        Returns:
            (turn_id, token) if TurnController is initialized, None otherwise.

        Note:
            This should be called at the start of each user interaction.
            The token is used to check cancellation during LLM/TTS generation.
        """
        if self._turn_controller is None:
            return None
        return self._turn_controller.start_new_turn()

    def handle_barge_in(self) -> tuple[int, TurnCancellationToken] | None:
        """Handle user barge-in (interrupt current turn).

        Returns:
            (new_turn_id, new_token) if TurnController is initialized, None otherwise.

        Note:
            This is called when VAD detects user speaking during playback.
            It cancels current LLM/TTS tasks and interrupts audio playback.
            Returns a new token that can be used for the next turn.
        """
        if self._turn_controller is None:
            logger.warning("TurnController not initialized, cannot handle barge-in")
            return None
        return self._turn_controller.handle_barge_in()

    @property
    def turn_controller(self) -> TurnController | None:
        """Get TurnController (for direct access if needed)."""
        return self._turn_controller

    # ─────────────────────────────────────────────────────────────────────────────
    # Alert Listener (Scheduler → Dialogue State)
    # ─────────────────────────────────────────────────────────────────────────────

    async def _alert_listener(self) -> None:
        """Background coroutine: consume alert_queue ONLY in dialogue state.

        Runs while in DIALOGUE state. When state changes (dialogue ends),
        this task is cancelled by _on_dialogue_ended().
        """
        logger.info("Alert listener started (dialogue state)")
        while self._robot_state == RobotState.DIALOGUE and self.running:
            try:
                alert = await self.scheduler.alert_queue.get()
                logger.info(
                    "Alert received in dialogue: [%s] %s (priority=%s)",
                    alert.source,
                    alert.message,
                    alert.priority,
                )
                # Buffer alert and process via LLM pipeline
                self._pending_alerts.append(alert)
                if not self.processing_lock.locked():
                    await self._process_pending_alerts()

            except asyncio.CancelledError:
                logger.info("Alert listener cancelled (dialogue ended)")
                break
            except Exception as e:
                logger.exception("Alert listener error: %s", e)

    async def _proactive_llm_turn(self) -> None:
        """Run an LLM turn initiated by proactive engine (no user input).

        Called when entering dialogue state from active/resting.
        The system_message has already been injected into conversation_history.
        """
        from reachy_mini_conversation_app.cascade.timing import tracker

        logger.info("Starting proactive LLM turn...")

        tracker.reset("proactive_turn")

        try:
            ctx = PipelineContext(
                llm=self.llm,
                tts=self.tts,
                speech_output=self.speech_output,
                conversation_history=self.conversation_history,
                tool_specs=self.tool_specs,
                deps=self.deps,
                result=PipelineResult(),
                token=None,
                turn_id=0,
            )

            tracker.mark("llm_start")
            if self.streaming_dialog_enabled:
                await pipeline.process_streaming_dialog_response(ctx)
            else:
                await pipeline.process_llm_response(ctx)
            tracker.mark("llm_complete")

            logger.info("Proactive LLM turn completed")

        except Exception as e:
            logger.exception("Error in proactive LLM turn: %s", e)

    async def _process_pending_alerts(self) -> None:
        """Process all pending alerts: merge → inject system message → call LLM.

        Called after each turn completes or when alerts arrive while LLM is idle.
        """
        if not self._pending_alerts:
            return

        # Take all pending alerts and clear the buffer
        alerts = list(self._pending_alerts)
        self._pending_alerts.clear()

        logger.info("Processing %d pending alert(s)", len(alerts))

        # Format alerts as a single system message
        alert_context = self._format_alerts_for_llm(alerts)

        # Inject into conversation history
        self.conversation_history.append({
            "role": "system",
            "content": alert_context,
        })

        # Call LLM without user input — let LLM decide whether to speak
        try:
            ctx = PipelineContext(
                llm=self.llm,
                tts=self.tts,
                speech_output=self.speech_output,
                conversation_history=self.conversation_history,
                tool_specs=self.tool_specs,
                deps=self.deps,
                result=PipelineResult(),
                token=None,  # No cancellation token for alert-driven turns
                turn_id=0,
            )

            if self.streaming_dialog_enabled:
                await pipeline.process_streaming_dialog_response(ctx)
            else:
                await pipeline.process_llm_response(ctx)

            logger.info("Alert-driven LLM turn completed")

        except Exception as e:
            logger.exception("Error processing alert-driven LLM turn: %s", e)

    @staticmethod
    def _format_alerts_for_llm(alerts: list[AlertEvent]) -> str:
        """Format pending alerts into a system message for LLM consumption."""
        lines = [
            "[系统提醒] 以下提醒事项到了触发时间，请根据当前对话上下文判断是否适合提及：",
        ]

        for i, alert in enumerate(alerts, 1):
            priority_label = {
                "urgent": "🔴 紧急",
                "important": "🟡 重要",
                "normal": "🟢 普通",
            }.get(alert.priority, "普通")

            lines.append(f"{i}. [{priority_label}] {alert.message}")
            if alert.description:
                lines.append(f"   描述：{alert.description}")
            if alert.recurrence_rule != "once":
                lines.append(f"   （重复提醒：{alert.recurrence_rule}）")

        lines.append("")
        lines.append("原则：")
        lines.append("- 如果当前对话正在处理更紧急的事项（如急救、呼救），延后提醒")
        lines.append("- 如果当前对话自然结束或话题切换，可以顺势提及")
        lines.append("- 紧急提醒（用药/医疗）不应延迟超过合理时间")
        lines.append("- 普通提醒可以根据情况选择沉默，留到下次对话中自然提及")
        lines.append("- 多个提醒可以合并表达，自然串联")

        return "\n".join(lines)
