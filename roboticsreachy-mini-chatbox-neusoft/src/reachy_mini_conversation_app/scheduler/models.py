"""Data models for the scheduler module."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass
class ScheduledEvent:
    """A scheduled event stored in the database (alarm or calendar event)."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    source: Literal["alarm", "calendar"] = "alarm"
    title: str = ""
    description: str = ""
    trigger_at: datetime | None = None  # None = passive query only
    recurrence_rule: str = "once"  # once / daily / weekly:mon / hourly:8 / minute:30 / monthly:15 / yearly:06-23
    priority: Literal["urgent", "important", "normal"] = "normal"
    status: Literal["active", "completed", "cancelled"] = "active"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class AlertEvent:
    """An alert that has fired and is being delivered to consumers."""

    event_type: str = "scheduled"
    source: str = "alarm"  # "alarm" | "calendar"
    schedule_id: str = ""
    title: str = ""
    description: str = ""
    priority: str = "normal"  # "urgent" | "important" | "normal"
    message: str = ""
    trigger_time: datetime = field(default_factory=datetime.now)
    recurrence_rule: str = "once"
