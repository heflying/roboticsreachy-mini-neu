"""Proactive module — active/resting state decision engine.

Handles the robot's behaviour when it is NOT in dialogue mode:
- Active state: robot is awake, monitoring environment, can initiate interaction
- Resting state: robot is quiet, only urgent events break through

The decision engine consumes alerts from scheduler.alert_queue and routes
them according to priority and current state (day/night).
"""

from reachy_mini_conversation_app.proactive.engine import DecisionEngine, RobotState

__all__ = ["DecisionEngine", "RobotState"]
