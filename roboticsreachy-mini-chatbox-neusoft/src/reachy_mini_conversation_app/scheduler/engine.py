"""Scheduler engine — the core async loop that checks due events and delivers them."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from reachy_mini_conversation_app.scheduler.models import AlertEvent, ScheduledEvent
from reachy_mini_conversation_app.scheduler.recurrence import calc_next_trigger
from reachy_mini_conversation_app.scheduler.store import SchedulerStore


logger = logging.getLogger(__name__)

# Default check interval in seconds
DEFAULT_CHECK_INTERVAL = 1.0

# System preset: daily 20:00 review alarm
SYSTEM_DAILY_REVIEW_ALARM_ID = "__system_daily_review__"
SYSTEM_DAILY_REVIEW_TIME = (20, 0)  # hour, minute


class Scheduler:
    """Async scheduler engine that monitors scheduled_events and fires alerts.

    Usage:
        scheduler = Scheduler(store=SchedulerStore())
        await scheduler.start()
        # ... consume from scheduler.alert_queue ...
        await scheduler.stop()
    """

    def __init__(self, store: SchedulerStore | None = None):
        self.store = store or SchedulerStore()
        self.alert_queue: asyncio.Queue[AlertEvent] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._check_interval = DEFAULT_CHECK_INTERVAL
        self._fired_ids: set[str] = set()  # Track already-fired event IDs in current cycle
        self._system_alarm_registered = False

    async def start(self) -> None:
        """Start the scheduler background loop."""
        if self._running:
            logger.warning("Scheduler already running")
            return

        self._running = True
        self._register_system_alarms()
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Scheduler started (check interval: %.1fs)", self._check_interval)

    async def stop(self) -> None:
        """Stop the scheduler background loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Scheduler stopped")

    @property
    def running(self) -> bool:
        return self._running

    # ── System alarms ─────────────────────────────────────────────────

    def _register_system_alarms(self) -> None:
        """Register system-preset alarms (e.g. daily 20:00 calendar review)."""
        if self._system_alarm_registered:
            return

        # Check if the daily review alarm already exists
        existing = self.store.get(SYSTEM_DAILY_REVIEW_ALARM_ID)
        if existing and existing.status == "active":
            self._system_alarm_registered = True
            return

        now = datetime.now()
        # Calculate next 20:00
        next_trigger = now.replace(
            hour=SYSTEM_DAILY_REVIEW_TIME[0],
            minute=SYSTEM_DAILY_REVIEW_TIME[1],
            second=0,
            microsecond=0,
        )
        if next_trigger <= now:
            next_trigger = next_trigger.replace(day=now.day + 1)

        system_event = ScheduledEvent(
            id=SYSTEM_DAILY_REVIEW_ALARM_ID,
            source="alarm",
            title="每日日历播报",
            description="系统每日定时提醒，检查明日日历事件并提醒用户",
            trigger_at=next_trigger,
            recurrence_rule="daily",
            priority="normal",
            status="active",
        )

        if existing:
            self.store.update_trigger_at(SYSTEM_DAILY_REVIEW_ALARM_ID, next_trigger)
            self.store.update_status(SYSTEM_DAILY_REVIEW_ALARM_ID, "active")
        else:
            self.store.create(system_event)

        self._system_alarm_registered = True
        logger.info(
            "System daily review alarm registered (next: %s)",
            next_trigger.isoformat(),
        )

    # ── Main loop ─────────────────────────────────────────────────────

    async def _run_loop(self) -> None:
        """Main scheduler loop: check due events → push to queue → reschedule."""
        while self._running:
            try:
                due_events = self.store.get_due_events(datetime.now())

                for event in due_events:
                    if event.id in self._fired_ids:
                        continue
                    self._fired_ids.add(event.id)

                    alert = AlertEvent(
                        event_type="scheduled",
                        source=event.source,
                        schedule_id=event.id,
                        title=event.title,
                        description=event.description,
                        priority=event.priority,
                        message=self._format_message(event),
                        trigger_time=event.trigger_at or datetime.now(),
                        recurrence_rule=event.recurrence_rule,
                    )

                    await self.alert_queue.put(alert)
                    logger.info(
                        "Alert fired: [%s] %s (priority=%s, recurrence=%s)",
                        event.source,
                        event.title,
                        event.priority,
                        event.recurrence_rule,
                    )

                    # Reschedule or complete
                    if event.recurrence_rule != "once":
                        next_trigger = calc_next_trigger(
                            event.recurrence_rule,
                            event.source,
                            event.trigger_at or datetime.now(),
                        )
                        if next_trigger:
                            self.store.update_trigger_at(event.id, next_trigger)
                            logger.info(
                                "Rescheduled '%s' → next: %s",
                                event.title,
                                next_trigger.isoformat(),
                            )
                        else:
                            self.store.update_status(event.id, "completed")
                    else:
                        self.store.update_status(event.id, "completed")

                # Clean up fired_ids for events that are no longer due
                self._fired_ids.clear()

            except Exception as e:
                logger.exception("Scheduler loop error: %s", e)

            await asyncio.sleep(self._check_interval)

    # ── Public API ────────────────────────────────────────────────────

    def create_event(self, event: ScheduledEvent) -> ScheduledEvent:
        """Create a new scheduled event in the store."""
        return self.store.create(event)

    def cancel_event(self, event_id: str) -> bool:
        """Cancel (mark as cancelled) an event."""
        event = self.store.get(event_id)
        if event is None:
            return False
        self.store.update_status(event_id, "cancelled")
        return True

    def mark_handled(self, schedule_id: str) -> None:
        """Mark an alert as handled (no-op for now; events are auto-completed)."""
        pass

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _format_message(event: ScheduledEvent) -> str:
        """Format a human-readable alert message from an event."""
        parts = [event.title]
        if event.description:
            parts.append(event.description)
        return "：".join(parts) if len(parts) > 1 else parts[0]
