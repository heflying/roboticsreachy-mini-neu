"""Recurrence rule parsing and next-trigger calculation."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Literal


logger = logging.getLogger(__name__)


def calc_next_trigger(
    rule: str,
    source: Literal["alarm", "calendar"],
    current_trigger: datetime,
) -> datetime | None:
    """Calculate the next trigger time based on the recurrence rule.

    Args:
        rule: Recurrence rule string (e.g. "daily", "weekly:mon", "monthly:15")
        source: Source module ("alarm" or "calendar")
        current_trigger: The trigger time that just fired

    Returns:
        Next trigger datetime, or None if the rule is "once"

    """
    if rule == "once":
        return None

    if source == "alarm":
        return _calc_alarm_next(rule, current_trigger)
    elif source == "calendar":
        return _calc_calendar_next(rule, current_trigger)

    logger.warning(f"Unknown source '{source}' for rule '{rule}', treating as once")
    return None


def _calc_alarm_next(rule: str, current: datetime) -> datetime | None:
    """Calculate next trigger for alarm recurrence rules."""
    if rule == "daily":
        return current + timedelta(days=1)

    if rule.startswith("weekly:"):
        # weekly:mon, weekly:tue, etc. — fires every 7 days
        return current + timedelta(days=7)

    if rule.startswith("hourly:"):
        try:
            hours = int(rule.split(":")[1])
        except (IndexError, ValueError):
            hours = 1
        return current + timedelta(hours=hours)

    if rule.startswith("minute:"):
        try:
            minutes = int(rule.split(":")[1])
        except (IndexError, ValueError):
            minutes = 1
        return current + timedelta(minutes=minutes)

    logger.warning(f"Unknown alarm recurrence rule: '{rule}', treating as once")
    return None


def _calc_calendar_next(rule: str, current: datetime) -> datetime | None:
    """Calculate next trigger for calendar recurrence rules."""
    if rule == "daily":
        return current + timedelta(days=1)

    if rule.startswith("yearly:"):
        # yearly:06-23 → next year same date
        try:
            parts = rule.split(":")
            month, day = int(parts[1].split("-")[0]), int(parts[1].split("-")[1])
        except (IndexError, ValueError):
            return current.replace(year=current.year + 1)
        try:
            next_dt = current.replace(year=current.year + 1, month=month, day=day)
        except ValueError:
            # Handle leap year issues (e.g. Feb 29)
            next_dt = current.replace(year=current.year + 1, month=3, day=1)
        return next_dt

    if rule.startswith("monthly:"):
        # monthly:15 → next month same day
        try:
            day = int(rule.split(":")[1])
        except (IndexError, ValueError):
            day = current.day
        year = current.year
        month = current.month + 1
        if month > 12:
            month = 1
            year += 1
        try:
            next_dt = current.replace(year=year, month=month, day=day)
        except ValueError:
            # Day doesn't exist in next month (e.g. Jan 31 → Feb 28)
            next_dt = current.replace(year=year, month=month + 1, day=1) if month < 12 else current.replace(year=year + 1, month=1, day=1)
        return next_dt

    if rule.startswith("lunar:"):
        # lunar:01-01 — not supported yet, treat as yearly
        logger.warning("Lunar calendar not yet supported, treating as yearly")
        return current.replace(year=current.year + 1)

    logger.warning(f"Unknown calendar recurrence rule: '{rule}', treating as once")
    return None
