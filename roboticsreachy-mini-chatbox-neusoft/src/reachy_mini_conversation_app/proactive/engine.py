"""Decision engine for proactive communication (active/resting state).

Implements the sense-decide loop from 机器人主动沟通行为清单.md.

Architecture:
    - detect_event(watch, timeout_ms, mode) is the unified perception interface.
    - Rule (hard-coded) or LLM decides actions + next detect_event params.
    - CascadeHandler manages state transitions and timeouts.
    - DecisionEngine.run(state) blocks (await) until start_dialog=True.
"""

from __future__ import annotations

import asyncio
import enum
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from reachy_mini_conversation_app.scheduler.models import AlertEvent

logger = logging.getLogger(__name__)


# ── State & Result ──────────────────────────────────────────────────

class RobotState(enum.Enum):
    """Operational state of the robot."""
    DIALOGUE = "dialogue"   # In conversation with user
    ACTIVE = "active"     # Awake, monitoring, can initiate
    RESTING = "resting"   # Quiet, only urgent break-through


@dataclass
class DecisionResult:
    """Result returned by DecisionEngine.run() when it's time to start a dialogue."""
    start_dialog: bool = False
    system_message: str | None = None


@dataclass
class RuleOutput:
    """Output of a rule decision."""
    actions: list[dict[str, Any]]
    detect_event: dict[str, Any] | None   # {"watch": [...], "timeout_ms": int, "mode": "any"|"all"}
    start_dialog: bool
    system_message: str | None = None


# ── Decision Engine ──────────────────────────────────────────────────

class DecisionEngine:
    """Sense-decide loop engine for non-dialogue states.

    Usage:
        engine = DecisionEngine(alert_queue)
        result = await engine.run(RobotState.ACTIVE)
        # result.start_dialog == True → handler enters dialogue state
    """

    def __init__(self, alert_queue: asyncio.Queue[AlertEvent]) -> None:
        self._alert_queue = alert_queue
        self._event_list: list[dict[str, Any]] = []
        self._deferred_alerts: list[dict[str, Any]] = []  # Deferred alerts for next dialogue
        self._state: RobotState = RobotState.ACTIVE  # Current robot state

    def set_state(self, state: RobotState) -> None:
        """Update the current robot state (called by handler on state transitions)."""
        self._state = state

    # ── Deferred Alerts ───────────────────────────────────────

    def add_deferred(self, alert_data: dict[str, Any]) -> None:
        """Add an alert to the deferred queue (for NEXT_DIALOG or SILENT_DEFER)."""
        self._deferred_alerts.append({
            "event": "scheduled",
            "data": alert_data,
        })

    def get_deferred_alerts_formatted(self) -> str | None:
        """Get deferred alerts formatted as a system message string for LLM.

        Returns None if there are no deferred alerts.
        """
        if not self._deferred_alerts:
            return None

        lines = [
            "[系统提醒] 以下提醒事项之前因为机器人处于非对话状态而被延后，"
            "现在对话开始了，请酌情自然提及：",
        ]
        for i, da in enumerate(self._deferred_alerts, 1):
            data = da.get("data", {})
            priority = data.get("priority", "normal")
            message = data.get("message", "")
            priority_label = {"urgent": "🔴 紧急", "important": "🟡 重要", "normal": "🟢 普通"}.get(
                priority, "普通"
            )
            lines.append(f"{i}. [{priority_label}] {message}")

        # Clear after returning
        self._deferred_alerts.clear()
        return "\n".join(lines)

    # ── Public: run the sense-decide loop ──────────────────────────

    async def run(self, state: RobotState) -> DecisionResult:
        """Run the sense-decide loop until a dialogue should be initiated.

        Blocks (await) until a rule/LLM decision sets start_dialog=True.
        """
        self._event_list.clear()

        while True:
            # 1. Route: rule or LLM (currently only rule is implemented)
            output = self._route(state)

            # 2. Execute actions (light alert, etc.)
            await self._execute_actions(output.actions)

            # 3. Call detect_event if requested
            if output.detect_event is not None:
                events = await self._detect_event(
                    watch=output.detect_event["watch"],
                    timeout_ms=output.detect_event["timeout_ms"],
                    mode=output.detect_event.get("mode", "any"),
                )
                self._event_list.extend(events)

            # 4. Check if we should start a dialogue
            if output.start_dialog:
                return DecisionResult(
                    start_dialog=True,
                    system_message=output.system_message,
                )

            # 5. Otherwise continue the loop (event_list has been updated)

    # ── Routing ───────────────────────────────────────────────────────

    def _route(self, state: RobotState) -> RuleOutput:
        """Route to rule or LLM based on event_list."""
        return self._rule(state)

    def _rule(self, state: RobotState) -> RuleOutput:
        """Hard-coded rule logic."""
        if not self._event_list:
            timeout_ms = 300_000 if state == RobotState.RESTING else 60_000
            return RuleOutput(
                actions=[],
                detect_event={
                    "watch": ["scheduled"],
                    "timeout_ms": timeout_ms,
                    "mode": "any",
                },
                start_dialog=False,
            )

        event = self._event_list.pop(0)

        if event.get("event") == "scheduled":
            return self._rule_handle_scheduled(event, state)

        logger.warning("Unknown event type in rule: %s", event.get("event"))
        return RuleOutput(actions=[], detect_event=None, start_dialog=False)

    def _rule_handle_scheduled(self, event: dict[str, Any], state: RobotState) -> RuleOutput:
        """Rule for handling scheduled (alarm/calendar) events."""
        priority = event.get("data", {}).get("priority", "normal")

        if priority == "urgent":
            return RuleOutput(
                actions=[],
                detect_event=None,
                start_dialog=True,
                system_message=self._format_scheduled_message(event),
            )

        # important and normal: start dialog directly
        return RuleOutput(
            actions=[],
            detect_event=None,
            start_dialog=True,
            system_message=self._format_scheduled_message(event),
        )

    # ── detect_event ──────────────────────────────────────────────────

    async def _detect_event(
        self, watch: list[str], timeout_ms: int, mode: str = "any"
    ) -> list[dict[str, Any]]:
        """Unified perception interface."""
        if not watch:
            return []

        tasks: dict[asyncio.Task, str] = {}

        for item in watch:
            if item == "scheduled":
                task = asyncio.create_task(self._wait_scheduled(timeout_ms))
                tasks[task] = item
            else:
                # Stub: other watch items return empty immediately
                pass

        if not tasks:
            return []

        try:
            if mode == "any":
                done, pending = await asyncio.wait(
                    tasks.keys(),
                    timeout=timeout_ms / 1000.0,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()
                    try:
                        await t
                    except asyncio.CancelledError:
                        pass

                results = []
                for t in done:
                    result = t.result()
                    if result:
                        results.append(result)
                return results

            else:  # mode == "all"
                done, pending = await asyncio.wait(
                    tasks.keys(),
                    timeout=timeout_ms / 1000.0,
                    return_when=asyncio.ALL_COMPLETED,
                )
                for t in pending:
                    t.cancel()
                    try:
                        await t
                    except asyncio.CancelledError:
                        pass

                results = []
                for task in tasks.keys():
                    if task in done:
                        result = task.result()
                        results.append(result if result else {"event": tasks[task], "data": None})
                    else:
                        results.append({"event": tasks[task], "data": None})
                return results

        except Exception as e:
            logger.exception("detect_event error: %s", e)
            return []

    async def _wait_scheduled(self, timeout_ms: int) -> dict[str, Any] | None:
        """Wait for a scheduled event from the alert queue."""
        try:
            alert = await asyncio.wait_for(
                self._alert_queue.get(),
                timeout=timeout_ms / 1000.0,
            )
            return {
                "event": "scheduled",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "schedule_id": alert.schedule_id,
                    "title": alert.title,
                    "description": alert.description,
                    "priority": alert.priority,
                    "message": alert.message,
                    "trigger_time": alert.trigger_time.isoformat() if alert.trigger_time else None,
                },
            }
        except asyncio.TimeoutError:
            return None

    # ── Action Execution ────────────────────────────────────────────

    async def _execute_actions(self, actions: list[dict[str, Any]]) -> None:
        """Execute a list of actions."""
        for action in actions:
            action_type = action.get("action")
            if action_type == "light_alert":
                logger.info("Light alert triggered (LED not yet implemented)")
            elif action_type == "speak":
                pass
            else:
                logger.warning("Unknown action type: %s", action_type)

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _format_scheduled_message(event: dict[str, Any]) -> str:
        """Format a scheduled event into a system message for LLM."""
        data = event.get("data", {})
        priority = data.get("priority", "normal")
        title = data.get("title", "")
        description = data.get("description", "")
        message = data.get("message", title)

        priority_label = {
            "urgent": "紧急",
            "important": "重要",
            "normal": "普通",
        }.get(priority, "普通")

        lines = [
            f"[系统提醒] 有一个{priority_label}提醒已到时间：",
            f"提醒内容：{message}",
        ]
        if description:
            lines.append(f"详细描述：{description}")
        lines.append("请根据当前情况，自然地提及这个提醒。")

        return "\n".join(lines)
