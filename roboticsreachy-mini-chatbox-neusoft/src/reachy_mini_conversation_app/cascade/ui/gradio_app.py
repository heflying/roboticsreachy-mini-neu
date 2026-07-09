"""Gradio UI for cascade mode."""

from __future__ import annotations
import asyncio
from concurrent.futures import Future
import logging
import threading
from typing import TYPE_CHECKING, Any, Dict, List

import cv2
import numpy as np
import gradio as gr
import numpy.typing as npt

from .audio_playback import AudioPlaybackSystem
from .audio_recording import ContinuousState, ContinuousVADRecorder, StreamingASRCallbacks


if TYPE_CHECKING:
    from reachy_mini import ReachyMini
    from reachy_mini_conversation_app.cascade.handler import CascadeHandler

from reachy_mini_conversation_app.cascade.asr import StreamingASRProvider
from reachy_mini_conversation_app.cascade.turn_result import TurnResult


logger = logging.getLogger(__name__)


class CascadeGradioUI:
    """Gradio interface for cascade pipeline."""

    def __init__(self, cascade_handler: CascadeHandler, robot: ReachyMini | None = None) -> None:
        """Initialize Gradio UI.

        Args:
            cascade_handler: Cascade pipeline handler
            robot: Robot instance (if running on robot hardware, enables robot speaker output)

        """
        self.handler = cascade_handler
        self.robot = robot

        self.shutdown_event = threading.Event()

        # Create playback system (pre-warmed threads)
        self.playback = AudioPlaybackSystem(
            robot=robot,
            head_wobbler=self.handler.deps.head_wobbler,
            shutdown_event=self.shutdown_event,
            tts_sample_rate=self.handler.tts.sample_rate,
        )

        # Initialize TurnController for interrupt handling
        self.handler.init_turn_controller(self.playback)

        # Wire speech output so handler plays audio through Gradio's playback system
        from reachy_mini_conversation_app.cascade.speech_output import GradioSpeechOutput

        self.handler.speech_output = GradioSpeechOutput(
            tts=self.handler.tts,
            playback=self.playback,
            barge_in_start_callback=lambda: self._start_barge_in_monitor(self.handler),
            barge_in_stop_callback=self._stop_barge_in_monitor,
        )

        # VAD recorder created lazily after handler.start() provides event loop
        self._vad_recorder: ContinuousVADRecorder | None = None
        self.continuous_mode = False

    def _is_streaming_asr(self) -> bool:
        """Check if the ASR provider supports streaming."""
        return isinstance(self.handler.asr, StreamingASRProvider)

    def _create_streaming_callbacks(self) -> StreamingASRCallbacks | None:
        """Create streaming ASR callbacks if provider supports it."""
        if not self._is_streaming_asr():
            return None

        def on_start() -> None:
            assert self.handler.loop is not None
            future = asyncio.run_coroutine_threadsafe(self.handler.process_audio_streaming_start(), self.handler.loop)
            try:
                future.result(timeout=5.0)
            except Exception as e:
                logger.error(f"Failed to start streaming ASR: {e}")
                raise  # Re-raise to let audio_recording.py handle it

        def on_chunk(chunk_wav: bytes) -> None:
            assert self.handler.loop is not None
            asyncio.run_coroutine_threadsafe(
                self._on_audio_chunk(chunk_wav),
                self.handler.loop
            )

        return StreamingASRCallbacks(on_start=on_start, on_chunk=on_chunk)

    async def _on_audio_chunk(self, chunk_wav: bytes) -> None:
        """Handle audio chunk from VAD.

        This method is called for each audio chunk during recording and streams
        audio into ASR. Sentence-pause events are sourced from the VAD state
        machine in ``ContinuousVADRecorder``.

        Args:
            chunk_wav: Audio chunk bytes (WAV format)
        """
        await self.handler.process_audio_streaming_chunk(chunk_wav)

    def _start_barge_in_monitor(self, handler: CascadeHandler) -> None:
        """Start barge-in monitoring during TTS playback.

        Task 9: VAD Barge-in Trigger Mechanism
        Called when TTS playback starts to enable VAD-based interruption.

        Args:
            handler: CascadeHandler instance to call handle_barge_in() on interruption.

        Lifecycle:
            - Called at playback start
            - Must be paired with _stop_barge_in_monitor() at playback end
        """
        if self._vad_recorder is not None:
            # Set callback that calls handler.handle_barge_in()
            self._vad_recorder.set_barge_in_callback(
                lambda: handler.handle_barge_in()
            )
            self._vad_recorder.enable_barge_in_detection(True)
            logger.info("[BARGE-IN][gradio] detection enabled during playback")

    def _stop_barge_in_monitor(self) -> None:
        """Stop barge-in monitoring after TTS playback ends.

        Task 9: VAD Barge-in Trigger Mechanism
        Called when TTS playback ends to disable VAD-based interruption.

        Lifecycle:
            - Called at playback end
            - Must be paired with _start_barge_in_monitor() at playback start
        """
        if self._vad_recorder is not None:
            self._vad_recorder.enable_barge_in_detection(False)
            self._vad_recorder.set_barge_in_callback(None)
            logger.info("[BARGE-IN][gradio] detection disabled after playback")

    def _prepare_streaming_asr(self) -> Future[Any] | None:
        """Pre-connect streaming ASR outside the user's speech window."""
        if not self._is_streaming_asr() or self.handler.loop is None:
            return None

        future = asyncio.run_coroutine_threadsafe(
            self.handler.prepare_audio_streaming_session(),
            self.handler.loop,
        )

        def log_prepare_result(done_future: Any) -> None:
            try:
                done_future.result()
                logger.info("Streaming ASR pre-connect ready")
            except Exception as e:
                logger.warning("Streaming ASR pre-connect failed: %s", e)

        future.add_done_callback(log_prepare_result)
        return future

    def _prepare_streaming_tts(self) -> Future[Any] | None:
        """Pre-connect streaming TTS outside the response synthesis window."""
        if self.handler.loop is None or not hasattr(self.handler, "prepare_tts_session"):
            return None

        future = asyncio.run_coroutine_threadsafe(
            self.handler.prepare_tts_session(),
            self.handler.loop,
        )

        def log_prepare_result(done_future: Any) -> None:
            try:
                done_future.result()
                logger.info("Streaming TTS pre-connect ready")
            except Exception as e:
                logger.warning("Streaming TTS pre-connect failed: %s", e)

        future.add_done_callback(log_prepare_result)
        return future

    def _prepare_realtime_providers(self) -> list[Future[Any]]:
        """Prepare realtime cloud providers for the next turn."""
        futures = [
            future
            for future in (self._prepare_streaming_asr(),)
            if future is not None
        ]
        return futures

    def _get_vad_recorder(self) -> ContinuousVADRecorder:
        """Get or create VAD recorder (lazy initialization).

        Binds is_dialogue_active callback so VAD only processes audio in
        DIALOGUE state. Also binds the handler's VAD reset callback so
        the state machine loop can reset VAD when entering DIALOGUE.
        """
        if self._vad_recorder is None:
            from reachy_mini_conversation_app.proactive.engine import RobotState

            self._vad_recorder = ContinuousVADRecorder(
                sample_rate=16000,
                streaming_asr_callbacks=self._create_streaming_callbacks(),
                on_speech_captured=self._on_vad_speech_captured,
                on_sentence_pause=self.handler.on_sentence_pause,
                is_dialogue_active=lambda: self.handler._robot_state == RobotState.DIALOGUE,
                on_audio_frame=self._on_proactive_audio_frame,
            )

            # Bind handler's VAD reset callback to recorder's reset_vad method
            self.handler._vad_reset_callback = self._vad_recorder.reset_vad

        return self._vad_recorder

    def _on_proactive_audio_frame(self, audio_frame: npt.NDArray[np.int16]) -> None:
        """Handle raw audio frame during non-dialogue state.

        Receives 32ms audio frames (numpy int16 array at 16kHz) when
        robot is in ACTIVE or RESTING state. Future proactive modules
        (fall detection, cough detection, emotion analysis) will consume
        these frames. Currently a placeholder that logs frame receipt.

        Args:
            audio_frame: Raw audio frame as numpy int16 array.
        """
        # Placeholder for future proactive audio analysis
        logger.debug("[PROACTIVE] Audio frame received (%d samples)", len(audio_frame))

    def _on_vad_speech_captured(self, wav_bytes: bytes) -> Any:
        """Handle speech captured by VAD recorder."""
        try:
            assert self.handler.loop is not None
            logger.info("[CONSOLE][reply] Gradio captured utterance, scheduling async processing")
            processing_future = asyncio.run_coroutine_threadsafe(self._process_audio_async(wav_bytes), self.handler.loop)
            ready_future: Future[Any] = Future()

            def complete_when_prepared(prep_futures: list[Future[Any]]) -> None:
                if not prep_futures:
                    ready_future.set_result(None)
                    return
                remaining = len(prep_futures)

                def mark_one_done(_future: Future[Any]) -> None:
                    nonlocal remaining
                    remaining -= 1
                    if remaining == 0 and not ready_future.done():
                        ready_future.set_result(None)

                for prep_future in prep_futures:
                    prep_future.add_done_callback(mark_one_done)

            def log_result(done_future: Any) -> None:
                try:
                    result = done_future.result()
                    if result["success"]:
                        logger.info(f"Continuous mode: Processed transcript: '{result.get('transcript', '')[:50]}...'")
                    else:
                        logger.error(f"Continuous mode processing error: {result.get('error')}")
                    if self.continuous_mode:
                        self._prepare_realtime_providers()
                    if not ready_future.done():
                        ready_future.set_result(None)
                except Exception as e:
                    logger.exception(f"Error processing continuous audio: {e}")
                    if not ready_future.done():
                        ready_future.set_result(None)

            processing_future.add_done_callback(log_result)
            if not ready_future.done():
                ready_future.set_result(None)
            return ready_future

        except Exception as e:
            logger.exception(f"Error processing continuous audio: {e}")
            return None

    def create_interface(self) -> gr.Blocks:
        """Create and return Gradio interface."""
        with gr.Blocks(title="Reachy Mini - Cascade Mode") as demo:
            gr.Markdown("# Reachy Mini Conversation (Cascade Mode)")

            # Chat display
            chatbot = gr.Chatbot(
                label="Conversation",
                type="messages",
                height=400,
            )

            # Status display
            status_box = gr.Textbox(
                label="Status",
                interactive=False,
                value="Ready. Click 'Start Listening' to begin.",
            )

            # Controls
            with gr.Row():
                listen_btn = gr.Button(
                    "Start Listening",
                    variant="primary",
                    scale=1,
                )
                clear_btn = gr.Button("Clear History", scale=1)

            # Listening toggle handler
            def toggle_listening(
                chat_history: List[Dict[str, Any]],
            ) -> tuple[str, List[Dict[str, Any]], gr.Button, gr.Timer]:
                """Toggle continuous VAD listening on/off."""
                recorder = self._get_vad_recorder()
                if not self.continuous_mode:
                    status = recorder.start()
                    self.continuous_mode = True
                    self._prepare_realtime_providers()
                    btn = gr.Button("Stop Listening", variant="stop")
                    return status, chat_history, btn, gr.Timer(active=True)
                else:
                    status = recorder.stop()
                    self.continuous_mode = False
                    btn = gr.Button("Start Listening", variant="primary")
                    return status, chat_history, btn, gr.Timer(active=False)

            poll_timer = gr.Timer(0.5, active=False)

            listen_btn.click(
                fn=toggle_listening,
                inputs=[chatbot],
                outputs=[status_box, chatbot, listen_btn, poll_timer],
            )

            # Polling for continuous mode updates (updates chat when VAD detects speech)
            def poll_continuous_updates(
                chat_history: List[Dict[str, Any]],
            ) -> tuple[List[Dict[str, Any]], str]:
                """Poll for updates from continuous mode processing."""
                if not self.continuous_mode:
                    return chat_history, "Ready to record..."

                # Get status based on current state
                recorder = self._get_vad_recorder()
                state_messages = {
                    ContinuousState.IDLE: "Continuous mode stopped",
                    ContinuousState.LISTENING: "Listening... (speak now)",
                    ContinuousState.RECORDING: "Recording speech...",
                    ContinuousState.PROCESSING: "Processing...",
                }
                status = state_messages.get(recorder.state, "Listening...")

                # Rebuild chat from turn results
                turns = self.handler.turn_results
                if not turns:
                    return chat_history, status

                new_history: list[dict[str, Any]] = []
                for turn in turns:
                    if turn.transcript:
                        new_history.append({"role": "user", "content": turn.transcript})
                    new_history.extend(self._turn_items_to_chat(turn))

                return new_history if new_history else chat_history, status

            # Wire up the poll timer (created earlier, before toggle handler)
            poll_timer.tick(
                fn=poll_continuous_updates,
                inputs=[chatbot],
                outputs=[chatbot, status_box],
            )

            # Clear button
            clear_btn.click(
                fn=self._clear_history,
                inputs=None,
                outputs=[chatbot, status_box],
            )

        return demo  # type: ignore[no-any-return]

    async def _process_audio_async(self, audio_bytes: bytes) -> Dict[str, Any]:
        """Process audio through cascade pipeline (async).

        Args:
            audio_bytes: Audio file bytes

        Returns:
            Dictionary with processing results

        """
        result: Dict[str, Any] = {
            "success": False,
            "transcript": None,
            "error": None,
        }

        try:
            # Choose streaming or batch processing based on ASR provider
            if self._is_streaming_asr():
                logger.info("Finalizing streaming ASR session...")
                turn = await self.handler.process_audio_streaming_end()
            else:
                turn = await self.handler.process_audio_manual(audio_bytes)

            result["transcript"] = turn.transcript

            if not turn.transcript.strip():
                logger.debug("Empty transcript (barge-in race condition), skipping")
                result["success"] = True
                return result

            # L3: 转录显示时间点（转录结果即将被 UI 显示）
            from reachy_mini_conversation_app.cascade.timing import tracker
            tracker.mark("transcript_show", {"transcript_len": len(turn.transcript)})

            # 统一在这里打印性能报告（speech_output.py 不再调用 print_summary）
            # Speech was already played during tool execution via speech_output.
            # 在有语音输出时，也需要在这里打印报告（包含 L3）
            tracker.print_summary()
            tracker.next_turn()  # 准备下一轮

            result["success"] = True

        except Exception as e:
            logger.exception(f"Error in async processing: {e}")
            result["error"] = str(e)

        return result

    def _turn_items_to_chat(self, turn: TurnResult) -> list[dict[str, Any]]:
        """Convert TurnResult items to Gradio chatbot message dicts."""
        messages: list[dict[str, Any]] = []
        for item in turn.items:
            if item.kind == "speak":
                messages.append({"role": "assistant", "content": item.text})
            elif item.kind == "assistant":
                messages.append({"role": "assistant", "content": item.text})
            elif item.kind == "image":
                rgb = self._decode_jpeg_to_rgb(item.image_jpeg)
                if rgb is not None:
                    messages.append({"role": "assistant", "content": gr.Image(value=rgb)})
            elif item.kind == "tool":
                messages.append({
                    "role": "assistant",
                    "content": item.tool_content,
                    "metadata": {"title": f"🛠️ Used tool {item.tool_name}", "status": "done"},
                })
        return messages

    @staticmethod
    def _decode_jpeg_to_rgb(jpeg_bytes: bytes) -> npt.NDArray[Any] | None:
        """Decode JPEG bytes to RGB numpy array, or None on failure."""
        try:
            np_arr = np.frombuffer(jpeg_bytes, np.uint8)
            np_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if np_img is None:
                return None
            return cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.warning(f"Failed to decode JPEG: {e}")
            return None

    def _clear_history(self) -> tuple[List[Dict[str, Any]], str]:
        """Clear conversation history."""
        self.handler.clear_state()
        return [], "History cleared"

    def launch(self, **kwargs: Any) -> None:
        """Launch Gradio interface."""
        import sys

        demo = self.create_interface()

        # Use prevent_thread_lock to allow post-launch logging
        # Store original value and override
        original_prevent = kwargs.get("prevent_thread_lock", False)
        kwargs["prevent_thread_lock"] = True

        demo.launch(**kwargs)

        # Print ready banner after server starts (now possible with prevent_thread_lock)
        logger.info("")
        logger.info("=" * 60)
        logger.info("🚀 GRADIO UI READY - You can now open the browser!")
        logger.info("=" * 60)
        logger.info("")
        sys.stdout.flush()

        # Block on the server if originally intended to block
        if not original_prevent:
            # Wait indefinitely (Gradio server runs until interrupted)
            import threading
            event = threading.Event()
            event.wait()  # Blocks until interrupted

    def close(self) -> None:
        """Close Gradio interface and shutdown all subsystems."""
        # Stop continuous mode if active
        if self._vad_recorder and self._vad_recorder.is_active:
            self._vad_recorder.stop()

        # Shutdown playback system
        self.playback.close()
