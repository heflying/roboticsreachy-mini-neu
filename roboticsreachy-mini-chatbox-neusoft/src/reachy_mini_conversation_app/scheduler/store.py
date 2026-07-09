"""SQLite storage for scheduled events."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from reachy_mini_conversation_app.scheduler.models import ScheduledEvent


logger = logging.getLogger(__name__)

# Default database path — same directory as the memory SQLite DB
DEFAULT_DB_PATH = Path.home() / ".reachy_mini" / "scheduler.db"


class SchedulerStore:
    """SQLite-backed store for scheduled events."""

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS scheduled_events (
                        id TEXT PRIMARY KEY,
                        source TEXT NOT NULL,
                        title TEXT NOT NULL,
                        description TEXT DEFAULT '',
                        trigger_at TEXT,
                        recurrence_rule TEXT DEFAULT 'once',
                        priority TEXT DEFAULT 'normal',
                        status TEXT DEFAULT 'active',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)
                conn.commit()
            finally:
                conn.close()

    # ── CRUD ──────────────────────────────────────────────────────────

    def create(self, event: ScheduledEvent) -> ScheduledEvent:
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """INSERT INTO scheduled_events
                       (id, source, title, description, trigger_at, recurrence_rule,
                        priority, status, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        event.id,
                        event.source,
                        event.title,
                        event.description,
                        event.trigger_at.isoformat() if event.trigger_at else None,
                        event.recurrence_rule,
                        event.priority,
                        event.status,
                        event.created_at.isoformat(),
                        event.updated_at.isoformat(),
                    ),
                )
                conn.commit()
                return event
            finally:
                conn.close()

    def get(self, event_id: str) -> Optional[ScheduledEvent]:
        with self._lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT * FROM scheduled_events WHERE id = ?", (event_id,)
                ).fetchone()
                if row is None:
                    return None
                return self._row_to_event(row)
            finally:
                conn.close()

    def get_due_events(self, now: datetime) -> List[ScheduledEvent]:
        """Get all active events with trigger_at <= now."""
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    """SELECT * FROM scheduled_events
                       WHERE trigger_at IS NOT NULL
                         AND trigger_at <= ?
                         AND status = 'active'
                       ORDER BY trigger_at ASC""",
                    (now.isoformat(),),
                ).fetchall()
                return [self._row_to_event(row) for row in rows]
            finally:
                conn.close()

    def get_active_by_source(self, source: str) -> List[ScheduledEvent]:
        """Get all active events for a given source (alarm/calendar)."""
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    """SELECT * FROM scheduled_events
                       WHERE source = ? AND status = 'active'
                       ORDER BY trigger_at ASC""",
                    (source,),
                ).fetchall()
                return [self._row_to_event(row) for row in rows]
            finally:
                conn.close()

    def query_by_date_range(self, start_date: str, end_date: str) -> List[ScheduledEvent]:
        """Get calendar events within a date range (by trigger_at or date extraction)."""
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    """SELECT * FROM scheduled_events
                       WHERE source = 'calendar'
                         AND status = 'active'
                         AND trigger_at IS NOT NULL
                         AND trigger_at >= ?
                         AND trigger_at <= ?
                       ORDER BY trigger_at ASC""",
                    (start_date, end_date),
                ).fetchall()
                return [self._row_to_event(row) for row in rows]
            finally:
                conn.close()

    def update_trigger_at(self, event_id: str, new_trigger_at: datetime | None) -> None:
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """UPDATE scheduled_events
                       SET trigger_at = ?, updated_at = ?
                       WHERE id = ?""",
                    (
                        new_trigger_at.isoformat() if new_trigger_at else None,
                        datetime.now().isoformat(),
                        event_id,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def update_status(self, event_id: str, status: str) -> None:
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """UPDATE scheduled_events
                       SET status = ?, updated_at = ?
                       WHERE id = ?""",
                    (status, datetime.now().isoformat(), event_id),
                )
                conn.commit()
            finally:
                conn.close()

    def delete(self, event_id: str) -> bool:
        with self._lock:
            conn = self._get_conn()
            try:
                cursor = conn.execute("DELETE FROM scheduled_events WHERE id = ?", (event_id,))
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    # ── helpers ───────────────────────────────────────────────────────

    def _row_to_event(self, row: sqlite3.Row) -> ScheduledEvent:
        trigger_at = None
        if row["trigger_at"]:
            try:
                trigger_at = datetime.fromisoformat(row["trigger_at"])
            except ValueError:
                pass

        return ScheduledEvent(
            id=row["id"],
            source=row["source"],
            title=row["title"],
            description=row["description"] or "",
            trigger_at=trigger_at,
            recurrence_rule=row["recurrence_rule"] or "once",
            priority=row["priority"] or "normal",
            status=row["status"] or "active",
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now(),
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else datetime.now(),
        )

    def event_to_dict(self, event: ScheduledEvent) -> dict:
        return {
            "id": event.id,
            "source": event.source,
            "title": event.title,
            "description": event.description,
            "trigger_at": event.trigger_at.isoformat() if event.trigger_at else None,
            "recurrence_rule": event.recurrence_rule,
            "priority": event.priority,
            "status": event.status,
            "created_at": event.created_at.isoformat(),
            "updated_at": event.updated_at.isoformat(),
        }
