"""Pre-warmed audio playback system for low-latency TTS output.

Implements interrupt-aware audio playback with generation-based filtering (R2, R3):
- Each turn has a unique generation ID (equals turn_id)
- Audio chunks are tagged with generation when enqueued
- interrupt(new_generation) clears queue and updates current generation
- Playback thread filters chunks by generation, discarding stale ones
- Completion events are bound to generation for proper turn completion
"""

from __future__ import annotations
import asyncio
import time
import base64
import logging
import threading
from queue import Empty, Queue
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd
import numpy.typing as npt


if TYPE_CHECKING:
    from reachy_mini import ReachyMini
    from reachy_mini_conversation_app.head_wobble import HeadWobbler


logger = logging.getLogger(__name__)


class AudioPlaybackSystem:
    """Pre-warmed audio playback system with persistent threads.

    Manages audio output through either sounddevice (laptop speakers) or
    robot.media (robot speakers), with synchronized head wobbler animation.

    The system pre-initializes audio streams at construction time to eliminate
    startup latency when playback begins.
    """

    def __init__(
        self,
        robot: ReachyMini | None,
        head_wobbler: HeadWobbler | None,
        shutdown_event: threading.Event | None = None,
        tts_sample_rate: int = 24000,
    ) -> None:
        """Initialize playback system.

        Args:
            robot: Robot instance (if available, enables robot speaker detection)
            head_wobbler: Head wobbler for animation during speech
            shutdown_event: External shutdown event for coordinated shutdown.
                           If None, creates an internal event.
            tts_sample_rate: TTS audio sample rate in Hz (default 24kHz)

        """
        self.robot = robot
        self.head_wobbler = head_wobbler
        self.shutdown_event = shutdown_event or threading.Event()
        self.tts_sample_rate = tts_sample_rate

        # Generation tracking for interrupt isolation (R2)
        self._current_generation: int = 0
        self._generation_lock = threading.Lock()

        # Completion events for turn-level playback completion (R3)
        self._playback_complete_events: dict[int, threading.Event] = {}

        # Queues hold (generation, payload) tuples. payload=None is a generation-bound sentinel.
        self._audio_queue: Queue[tuple[int, npt.NDArray[np.int16] | None]] = Queue(maxsize=100)
        self._wobbler_queue: Queue[tuple[int, bytes | None]] = Queue(maxsize=100)

        self._playback_thread: threading.Thread | None = None
        self._wobbler_thread: threading.Thread | None = None
        self._use_robot_media = False

        # Stream reference for abort during interrupt (R8)
        self._stream: sd.OutputStream | None = None
        self._stream_lock = threading.Lock()

        # Detect playback mode and start threads
        self._init_playback_threads()

    @property
    def audio_queue(self) -> Queue[tuple[int, npt.NDArray[np.int16] | None]]:
        """Audio chunk queue (for direct access if needed)."""
        return self._audio_queue

    @property
    def wobbler_queue(self) -> Queue[tuple[int, bytes | None]]:
        """Wobbler chunk queue (for direct access if needed)."""
        return self._wobbler_queue

    @property
    def use_robot_media(self) -> bool:
        """Whether using robot.media for playback."""
        return self._use_robot_media

    @property
    def current_generation(self) -> int:
        """Current generation ID (equals current turn_id)."""
        with self._generation_lock:
            return self._current_generation

    def put_audio(self, chunk: npt.NDArray[np.int16], generation: int | None = None) -> None:
        """Queue an audio chunk for playback with generation tag.

        Args:
            chunk: int16 audio data
            generation: Optional generation ID. If None, uses current generation.

        """
        gen = generation if generation is not None else self.current_generation
        self._audio_queue.put((gen, chunk))

    def put_wobbler(self, chunk: bytes, generation: int | None = None) -> None:
        """Queue raw audio bytes for wobbler animation with generation tag.

        Args:
            chunk: Raw audio bytes
            generation: Optional generation ID. If None, uses current generation.

        """
        gen = generation if generation is not None else self.current_generation
        self._wobbler_queue.put((gen, chunk))

    def interrupt(self, new_generation: int) -> None:
        """Interrupt current playback, allowing only new generation audio.

        Args:
            new_generation: New generation ID (typically equals turn_id)

        Execution:
        1. Update _current_generation
        2. Clear _audio_queue and _wobbler_queue
        3. Put sentinel for playback thread to check generation
        4. Set stale completion events to unblock waiters
        5. Abort stream if sounddevice mode (for in-flight write)

        """
        with self._generation_lock:
            old_generation = self._current_generation
            self._current_generation = new_generation
            logger.info(f"AudioPlayback generation updated: {old_generation} -> {new_generation}")

        # Clear audio queue
        cleared_audio = 0
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
                cleared_audio += 1
            except Empty:
                break
        logger.debug(f"Cleared {cleared_audio} audio chunks from queue")

        # Put generation-bound sentinel so the playback thread can react promptly.
        self._audio_queue.put((new_generation, None))

        # Clear wobbler queue (R4)
        cleared_wobbler = 0
        while not self._wobbler_queue.empty():
            try:
                self._wobbler_queue.get_nowait()
                cleared_wobbler += 1
            except Empty:
                break
        logger.debug(f"Cleared {cleared_wobbler} wobbler chunks from queue")

        # Put sentinel for wobbler thread
        self._wobbler_queue.put((new_generation, None))

        # Set stale completion events to unblock waiters (R3)
        with self._generation_lock:
            stale_turn_ids = [
                tid for tid in list(self._playback_complete_events.keys())
                if tid < new_generation
            ]
            for tid in stale_turn_ids:
                event = self._playback_complete_events.pop(tid, None)
                if event is not None and not event.is_set():
                    event.set()
                    logger.debug(f"Set stale completion event for turn_id={tid}")

        self._flush_backend_buffer()

    def _flush_backend_buffer(self) -> None:
        """Clear backend-side playback buffers after an interrupt."""
        if self._use_robot_media:
            if self.robot is None or not hasattr(self.robot, "media"):
                return
            try:
                self.robot.media.stop_playing()
                time.sleep(0.05)
                self.robot.media.start_playing()
                time.sleep(0.05)
                logger.info("Flushed robot.media playback buffer after interrupt")
            except Exception as exc:
                logger.warning("Failed to flush robot.media playback buffer: %s", exc)
            return

        with self._stream_lock:
            stream = self._stream

        if stream is None:
            return

        try:
            stream.abort()
            logger.info("Aborted sounddevice output stream to flush buffered audio")
        except sd.PortAudioError as exc:
            logger.debug("Ignoring sounddevice abort error during interrupt flush: %s", exc)

    def signal_end_of_turn(self, caller_turn_id: int | None = None) -> tuple[int, threading.Event] | None:
        """Signal end of current playback session with turn_id binding.

        Args:
            caller_turn_id: The turn_id of the caller. If None, uses current generation.

        Returns:
            Tuple of (turn_id, event) if turn matches current generation.
            For stale turns, returns event that is already set.
            For exact match, returns event that will be set on playback completion.

        """
        turn_id = caller_turn_id if caller_turn_id is not None else self.current_generation

        with self._generation_lock:
            current_gen = self._current_generation

            # Stale END_OF_TURN: turn_id < current_gen (R3)
            if turn_id < current_gen:
                logger.debug(f"Stale END_OF_TURN: turn_id={turn_id} < current_gen={current_gen}")
                # Create an already-set event for stale turn
                event = threading.Event()
                event.set()
                return (turn_id, event)

            # Exact match: turn_id == current_gen (R3)
            # Create or get completion event
            if turn_id not in self._playback_complete_events:
                self._playback_complete_events[turn_id] = threading.Event()
                logger.debug(f"Created completion event for turn_id={turn_id}")

            event = self._playback_complete_events[turn_id]

        # Put sentinel in queues
        self._audio_queue.put((turn_id, None))
        self._wobbler_queue.put((turn_id, None))

        return (turn_id, event)

    def _signal_playback_complete(self, generation: int) -> None:
        """Set and clear the completion event for a finished generation, if any."""
        with self._generation_lock:
            event = self._playback_complete_events.pop(generation, None)
        if event is not None and not event.is_set():
            event.set()
            logger.debug("Set playback completion event for turn_id=%s", generation)

    def close(self) -> None:
        """Shutdown playback threads."""
        logger.info("Shutting down pre-warmed audio system...")
        self.shutdown_event.set()

        if self._playback_thread:
            self._playback_thread.join(timeout=2)
        if self._wobbler_thread:
            self._wobbler_thread.join(timeout=2)

        # Stop robot media playback if using robot.media
        if self._use_robot_media and self.robot is not None and hasattr(self.robot, "media"):
            logger.info("Stopping robot.media playback system...")
            self.robot.media.stop_playing()

        logger.info("Audio system shutdown complete")

    def _init_playback_threads(self) -> None:
        """Initialize persistent audio playback and wobbler threads (pre-warmed)."""
        # Determine playback mode based on system's default audio output device
        # Use robot.media ONLY if:
        # 1. Robot hardware is available (not simulation)
        # 2. Default output device is a robot speaker (reSpeaker, etc.)

        status = self.robot.client.get_status() if self.robot is not None else None
        robot_available = (
            self.robot is not None
            and hasattr(self.robot, "media")
            and not getattr(status, "simulation_enabled", False)
        )

        # Check if default output is a robot speaker
        default_is_robot_speaker = False
        if robot_available:
            try:
                default_device = sd.query_devices(kind="output")
                device_name = default_device["name"].lower()
                # Common robot speaker names
                robot_speaker_keywords = ["respeaker", "xvf3800", "reachy"]
                default_is_robot_speaker = any(keyword in device_name for keyword in robot_speaker_keywords)
                logger.info(f"AUDIO PREWARM: Default output device: {default_device['name']}")
                logger.debug(f"AUDIO PREWARM: Is robot speaker? {default_is_robot_speaker}")
            except Exception as e:
                logger.warning(f"Failed to detect default audio device: {e}")

        # Use robot.media only if both robot is available AND default output is robot speaker
        self._use_robot_media = robot_available and default_is_robot_speaker

        if self._use_robot_media:
            logger.info("AUDIO PREWARM: Using robot.media for playback (robot speaker is default)")
            self._init_robot_playback_threads()
        else:
            reason = "laptop/other speaker" if robot_available else "simulation/no robot"
            logger.info(f"AUDIO PREWARM: Using sounddevice for playback ({reason})")
            self._init_sounddevice_playback_threads()

    def _init_sounddevice_playback_threads(self) -> None:
        """Initialize sounddevice playback threads (laptop speakers)."""

        def open_output_stream() -> sd.OutputStream:
            stream = sd.OutputStream(
                samplerate=self.tts_sample_rate,
                channels=1,
                dtype=np.int16,
                blocksize=4096,
                latency="low",
            )
            stream.start()
            with self._stream_lock:
                self._stream = stream
            return stream

        def close_output_stream(stream: sd.OutputStream | None) -> None:
            if stream is None:
                return
            try:
                stream.stop()
            except sd.PortAudioError:
                pass
            try:
                stream.close()
            except sd.PortAudioError:
                pass
            with self._stream_lock:
                if self._stream is stream:
                    self._stream = None

        def persistent_playback_thread() -> None:
            """Run persistent audio playback thread (pre-warmed and ready)."""
            from reachy_mini_conversation_app.cascade.timing import tracker

            stream: sd.OutputStream | None = None
            try:
                # Pre-initialize sounddevice stream (happens once at startup)
                tracker.mark("audio_stream_prewarm_start")

                all_devices = sd.query_devices()
                default_output_idx = sd.default.device[1]
                logger.info(f"AUDIO PREWARM: Total {len(all_devices)} devices available")
                for i, dev in enumerate(all_devices):
                    if dev["max_output_channels"] > 0:
                        default_marker = " (DEFAULT OUTPUT)" if i == default_output_idx else ""
                        logger.info(f"  [{i}] {dev['name']} - {dev['max_output_channels']} channels{default_marker}")

                default_device = sd.query_devices(kind="output")
                logger.info(f"AUDIO PREWARM: Using default device: {default_device['name']}")

                # Create stream once (pre-warmed)
                stream = open_output_stream()
                actual_latency_ms = stream.latency * 1000
                logger.info(f"AUDIO PREWARM: Stream ready. Latency: {actual_latency_ms:.0f}ms")
                tracker.mark("audio_stream_prewarm_complete", {"stream_latency_ms": round(actual_latency_ms, 1)})

                # Main playback loop - runs forever
                while not self.shutdown_event.is_set():
                    try:
                        # Wait for chunks with timeout to allow shutdown
                        item = self._audio_queue.get(timeout=0.1)

                        generation, chunk = item
                        if chunk is None:  # Generation-bound sentinel
                            self._signal_playback_complete(generation)
                            continue

                        # Check generation against current (R2 filtering)
                        with self._generation_lock:
                            current_gen = self._current_generation

                        if generation < current_gen:
                            # Old generation, discard
                            logger.debug(f"Discarding audio gen={generation}, current={current_gen}")
                            continue

                        # Current generation, play
                        try:
                            stream.write(chunk)
                        except sd.PortAudioError as exc:
                            if self.shutdown_event.is_set():
                                break
                            logger.warning("Audio output stream interrupted; recreating output stream: %s", exc)
                            close_output_stream(stream)
                            stream = open_output_stream()
                            logger.info("Audio output stream recovered after interrupt")
                            with self._generation_lock:
                                recovered_current_gen = self._current_generation
                            if generation < recovered_current_gen:
                                logger.debug(
                                    "Dropping recovered stale audio gen=%s, current=%s",
                                    generation,
                                    recovered_current_gen,
                                )
                                continue
                            stream.write(chunk)

                    except Empty:
                        continue

            except Exception as e:
                logger.exception(f"Error in persistent playback thread: {e}")
            finally:
                close_output_stream(stream)
                logger.info("Playback thread shutdown")

        # Start persistent threads
        self._playback_thread = threading.Thread(target=persistent_playback_thread, daemon=True, name="AudioPlayback")
        self._wobbler_thread = threading.Thread(target=self._persistent_wobbler_thread, daemon=True, name="Wobbler")

        self._playback_thread.start()
        self._wobbler_thread.start()

        # Give threads time to initialize
        time.sleep(0.1)

        logger.info("Pre-warmed audio playback system initialized (sounddevice)")

    def _init_robot_playback_threads(self) -> None:
        """Initialize robot.media playback threads (robot speakers)."""
        import librosa

        # Start robot media playback (must be called before pushing audio)
        if self.robot is not None and hasattr(self.robot, "media"):
            logger.info("Starting robot.media playback system...")
            self.robot.media.start_playing()
            time.sleep(0.5)  # Give pipeline time to initialize
            logger.info("Robot.media playback system started")

        def persistent_playback_thread() -> None:
            """Run persistent audio playback thread using robot.media."""
            from reachy_mini_conversation_app.cascade.timing import tracker

            # Type guard: ensure robot and media are available
            if self.robot is None or not hasattr(self.robot, "media"):
                logger.error("Robot media not available for playback")
                return

            try:
                # Pre-initialize robot media
                tracker.mark("audio_stream_prewarm_start")

                # Get robot audio sample rate
                device_sample_rate = self.robot.media.get_output_audio_samplerate()
                logger.info(f"AUDIO PREWARM: Robot speaker sample rate: {device_sample_rate}Hz")
                tracker.mark(
                    "audio_stream_prewarm_complete", {"device": "robot.media", "sample_rate": device_sample_rate}
                )

                # Main playback loop - runs forever
                while not self.shutdown_event.is_set():
                    try:
                        # Wait for chunks with timeout to allow shutdown
                        item = self._audio_queue.get(timeout=0.1)

                        generation, chunk = item
                        if chunk is None:  # Generation-bound sentinel
                            self._signal_playback_complete(generation)
                            continue

                        # Check generation against current (R2 filtering)
                        with self._generation_lock:
                            current_gen = self._current_generation

                        if generation < current_gen:
                            # Old generation, discard
                            logger.debug(f"Discarding robot audio gen={generation}, current={current_gen}")
                            continue

                        # Convert int16 to float32 for robot.media
                        audio_float = chunk.astype(np.float32) / 32768.0

                        # Resample if needed
                        if device_sample_rate != self.tts_sample_rate:
                            audio_float = librosa.resample(
                                audio_float,
                                orig_sr=self.tts_sample_rate,
                                target_sr=device_sample_rate,
                            )

                        # Push to robot speaker
                        self.robot.media.push_audio_sample(audio_float)

                    except Empty:
                        continue

            except Exception as e:
                logger.exception(f"Error in robot playback thread: {e}")
            finally:
                logger.info("Robot playback thread shutdown")

        # Start persistent threads
        self._playback_thread = threading.Thread(
            target=persistent_playback_thread, daemon=True, name="RobotAudioPlayback"
        )
        self._wobbler_thread = threading.Thread(target=self._persistent_wobbler_thread, daemon=True, name="Wobbler")

        self._playback_thread.start()
        self._wobbler_thread.start()

        # Give threads time to initialize
        time.sleep(0.1)

        logger.info("Pre-warmed audio playback system initialized (robot.media)")

    def _persistent_wobbler_thread(self) -> None:
        """Run persistent wobbler thread (pre-warmed and ready).

        Shared by both sounddevice and robot.media playback modes.
        Supports generation filtering (R4) to discard stale wobbler data.
        """
        try:
            logger.info("WOBBLER PREWARM: Thread ready")

            # Main wobbler loop - runs forever
            while not self.shutdown_event.is_set():
                try:
                    # Wait for chunks with timeout to allow shutdown
                    item = self._wobbler_queue.get(timeout=0.1)

                    generation, chunk = item

                    if chunk is None:  # Sentinel - end of current playback session
                        # Reset wobbler between turns
                        if self.head_wobbler:
                            self.head_wobbler.reset()
                        continue

                    # Check generation against current (R4 filtering)
                    with self._generation_lock:
                        current_gen = self._current_generation

                    if generation < current_gen:
                        # Old generation, discard
                        logger.debug(f"Discarding wobbler gen={generation}, current={current_gen}")
                        continue

                    # Feed to wobbler
                    if self.head_wobbler:
                        self.head_wobbler.feed(base64.b64encode(chunk).decode("utf-8"))

                    # Rate limit to match playback
                    chunk_duration = len(chunk) / (2 * self.tts_sample_rate)
                    time.sleep(chunk_duration)

                except Empty:
                    continue

        except Exception as e:
            logger.exception(f"Error in persistent wobbler thread: {e}")
        finally:
            logger.info("Wobbler thread shutdown")
