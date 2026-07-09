"""SQLite persistence for session, profile, and care memories."""

from __future__ import annotations
import json
import uuid
import sqlite3
import threading
from typing import Any, Iterable
from pathlib import Path
from datetime import datetime, timezone

from reachy_mini_conversation_app.memory.models import (
    Turn,
    User,
    CareTask,
    MemoryNote,
    ProfileFact,
    SessionRecord,
    CareTaskOccurrence,
)


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_dumps(value: dict[str, Any] | list[Any] | None) -> str:
    return json.dumps(value or {}, ensure_ascii=False, separators=(",", ":"))


def _json_loads(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


class SQLiteMemoryStore:
    """Small SQLite store used by MemoryRuntime."""

    def __init__(self, db_path: str | Path):
        """Create a store for ``db_path`` and initialise the schema."""
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self.initialize()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._conn.close()

    def initialize(self) -> None:
        """Create tables and indexes if they do not exist."""
        with self._lock:
            self._conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA foreign_keys=ON;

                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    external_user_id TEXT NOT NULL UNIQUE,
                    display_name TEXT,
                    timezone TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    status TEXT NOT NULL,
                    reason TEXT,
                    summary TEXT,
                    summary_json TEXT NOT NULL DEFAULT '{}',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(user_id) REFERENCES users(id)
                );

                CREATE TABLE IF NOT EXISTS turns (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(session_id) REFERENCES sessions(id),
                    FOREIGN KEY(user_id) REFERENCES users(id)
                );

                CREATE TABLE IF NOT EXISTS profile_facts (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    category TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    status TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_session_id TEXT,
                    evidence TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                );

                CREATE TABLE IF NOT EXISTS memory_notes (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT,
                    note TEXT NOT NULL,
                    salience REAL NOT NULL,
                    status TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(id),
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );

                CREATE TABLE IF NOT EXISTS care_tasks (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    due_at TEXT,
                    recurrence_rule TEXT,
                    status TEXT NOT NULL,
                    source TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                );

                CREATE TABLE IF NOT EXISTS care_task_occurrences (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    occurrence_key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    source_session_id TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT,
                    FOREIGN KEY(task_id) REFERENCES care_tasks(id),
                    FOREIGN KEY(user_id) REFERENCES users(id),
                    UNIQUE(task_id, occurrence_key)
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_user_started
                    ON sessions(user_id, started_at DESC);
                CREATE INDEX IF NOT EXISTS idx_turns_session_created
                    ON turns(session_id, created_at ASC);
                CREATE INDEX IF NOT EXISTS idx_profile_user_status
                    ON profile_facts(user_id, status, category);
                CREATE INDEX IF NOT EXISTS idx_profile_user_key
                    ON profile_facts(user_id, key, status);
                CREATE INDEX IF NOT EXISTS idx_notes_user_status
                    ON memory_notes(user_id, status, salience DESC);
                CREATE INDEX IF NOT EXISTS idx_tasks_user_status_due
                    ON care_tasks(user_id, status, due_at);
                CREATE INDEX IF NOT EXISTS idx_occurrences_user_status
                    ON care_task_occurrences(user_id, status, updated_at DESC);
                """
            )
            self._conn.commit()

    def _execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        cursor = self._conn.execute(sql, tuple(params))
        self._conn.commit()
        return cursor

    def get_or_create_user(
        self,
        external_user_id: str = "default",
        display_name: str | None = None,
        timezone_name: str = "Asia/Shanghai",
    ) -> User:
        """Return an existing user or create one."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM users WHERE external_user_id = ?",
                (external_user_id,),
            ).fetchone()
            if row is None:
                now = utc_now()
                user_id = f"user_{uuid.uuid4().hex}"
                self._execute(
                    """
                    INSERT INTO users (id, external_user_id, display_name, timezone, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (user_id, external_user_id, display_name, timezone_name, now, now),
                )
                row = self._conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            return self._row_to_user(row)

    def start_session(self, user_id: str, metadata: dict[str, Any] | None = None) -> SessionRecord:
        """Create a new active session."""
        with self._lock:
            now = utc_now()
            session_id = f"sess_{uuid.uuid4().hex}"
            self._execute(
                """
                INSERT INTO sessions (id, user_id, started_at, status, metadata_json)
                VALUES (?, ?, ?, 'active', ?)
                """,
                (session_id, user_id, now, _json_dumps(metadata)),
            )
            row = self._conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            return self._row_to_session(row)

    def end_session(
        self,
        session_id: str,
        *,
        reason: str,
        summary: str | None = None,
        summary_json: dict[str, Any] | None = None,
    ) -> SessionRecord | None:
        """Mark a session as ended."""
        with self._lock:
            now = utc_now()
            self._execute(
                """
                UPDATE sessions
                SET ended_at = ?, status = 'ended', reason = ?, summary = COALESCE(?, summary),
                    summary_json = CASE WHEN ? IS NULL THEN summary_json ELSE ? END
                WHERE id = ?
                """,
                (
                    now,
                    reason,
                    summary,
                    json.dumps(summary_json, ensure_ascii=False) if summary_json is not None else None,
                    _json_dumps(summary_json),
                    session_id,
                ),
            )
            row = self._conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            return self._row_to_session(row) if row else None

    def append_turn(
        self,
        session_id: str,
        user_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> Turn:
        """Append a final transcript turn."""
        cleaned = content.strip()
        if not cleaned:
            raise ValueError("turn content cannot be empty")
        if role not in {"user", "assistant", "system"}:
            raise ValueError(f"unsupported turn role: {role}")
        with self._lock:
            now = utc_now()
            turn_id = f"turn_{uuid.uuid4().hex}"
            self._execute(
                """
                INSERT INTO turns (id, session_id, user_id, role, content, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (turn_id, session_id, user_id, role, cleaned, now, _json_dumps(metadata)),
            )
            row = self._conn.execute("SELECT * FROM turns WHERE id = ?", (turn_id,)).fetchone()
            return self._row_to_turn(row)

    def get_turns(self, session_id: str, limit: int | None = None) -> list[Turn]:
        """Return turns for a session in chronological order."""
        with self._lock:
            sql = "SELECT * FROM turns WHERE session_id = ? ORDER BY created_at ASC, rowid ASC"
            params: tuple[Any, ...] = (session_id,)
            if limit is not None:
                sql += " LIMIT ?"
                params = (session_id, limit)
            rows = self._conn.execute(sql, params).fetchall()
            return [self._row_to_turn(row) for row in rows]

    def get_recent_sessions(self, user_id: str, limit: int = 5) -> list[SessionRecord]:
        """Return recently ended sessions with summaries."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM sessions
                WHERE user_id = ? AND status = 'ended' AND summary IS NOT NULL AND summary <> ''
                ORDER BY COALESCE(ended_at, started_at) DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
            return [self._row_to_session(row) for row in rows]

    def upsert_profile_fact(
        self,
        user_id: str,
        *,
        key: str,
        value: str,
        category: str,
        confidence: float,
        status: str,
        source: str,
        source_session_id: str | None = None,
        evidence: str | None = None,
    ) -> ProfileFact:
        """Insert a profile fact, archiving the previous active fact for the same key."""
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError("profile fact key and value are required")
        with self._lock:
            now = utc_now()
            if status == "active":
                self._execute(
                    """
                    UPDATE profile_facts
                    SET status = 'archived', updated_at = ?
                    WHERE user_id = ? AND key = ? AND status = 'active' AND value <> ?
                    """,
                    (now, user_id, key, value),
                )
                existing = self._conn.execute(
                    """
                    SELECT * FROM profile_facts
                    WHERE user_id = ? AND key = ? AND status = 'active' AND value = ?
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                    (user_id, key, value),
                ).fetchone()
                if existing is not None:
                    self._execute(
                        """
                        UPDATE profile_facts
                        SET confidence = MAX(confidence, ?), source = ?, source_session_id = ?,
                            evidence = COALESCE(?, evidence), updated_at = ?
                        WHERE id = ?
                        """,
                        (confidence, source, source_session_id, evidence, now, existing["id"]),
                    )
                    row = self._conn.execute("SELECT * FROM profile_facts WHERE id = ?", (existing["id"],)).fetchone()
                    return self._row_to_profile_fact(row)

            existing_same_status = self._conn.execute(
                """
                SELECT * FROM profile_facts
                WHERE user_id = ? AND key = ? AND status = ? AND value = ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (user_id, key, status, value),
            ).fetchone()
            if existing_same_status is not None:
                self._execute(
                    """
                    UPDATE profile_facts
                    SET confidence = MAX(confidence, ?), source = ?, source_session_id = ?,
                        evidence = COALESCE(?, evidence), updated_at = ?
                    WHERE id = ?
                    """,
                    (confidence, source, source_session_id, evidence, now, existing_same_status["id"]),
                )
                row = self._conn.execute(
                    "SELECT * FROM profile_facts WHERE id = ?",
                    (existing_same_status["id"],),
                ).fetchone()
                return self._row_to_profile_fact(row)

            fact_id = f"fact_{uuid.uuid4().hex}"
            self._execute(
                """
                INSERT INTO profile_facts (
                    id, user_id, key, value, category, confidence, status, source,
                    source_session_id, evidence, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fact_id,
                    user_id,
                    key,
                    value,
                    category,
                    confidence,
                    status,
                    source,
                    source_session_id,
                    evidence,
                    now,
                    now,
                ),
            )
            row = self._conn.execute("SELECT * FROM profile_facts WHERE id = ?", (fact_id,)).fetchone()
            return self._row_to_profile_fact(row)

    def search_profile_facts(
        self,
        user_id: str,
        query: str | None = None,
        *,
        statuses: tuple[str, ...] = ("active",),
        limit: int = 20,
    ) -> list[ProfileFact]:
        """Search profile facts by key or value."""
        with self._lock:
            placeholders = ",".join("?" for _ in statuses)
            params: list[Any] = [user_id, *statuses]
            sql = f"SELECT * FROM profile_facts WHERE user_id = ? AND status IN ({placeholders})"
            if query:
                sql += " AND (key LIKE ? OR value LIKE ? OR category LIKE ?)"
                like = f"%{query.strip()}%"
                params.extend([like, like, like])
            sql += " ORDER BY updated_at DESC LIMIT ?"
            params.append(limit)
            rows = self._conn.execute(sql, params).fetchall()
            return [self._row_to_profile_fact(row) for row in rows]

    def archive_profile_fact(self, user_id: str, query: str) -> list[ProfileFact]:
        """Archive active or pending facts matching an id, key, or value query."""
        cleaned = query.strip()
        if not cleaned:
            return []
        with self._lock:
            now = utc_now()
            like = f"%{cleaned}%"
            rows = self._conn.execute(
                """
                SELECT * FROM profile_facts
                WHERE user_id = ? AND status IN ('active', 'pending_confirmation')
                  AND (id = ? OR key = ? OR key LIKE ? OR value LIKE ?)
                ORDER BY updated_at DESC
                LIMIT 20
                """,
                (user_id, cleaned, cleaned, like, like),
            ).fetchall()
            ids = [row["id"] for row in rows]
            if ids:
                self._conn.executemany(
                    "UPDATE profile_facts SET status = 'archived', updated_at = ? WHERE id = ?",
                    [(now, fact_id) for fact_id in ids],
                )
                self._conn.commit()
            return [self._row_to_profile_fact(row) for row in rows]

    def confirm_profile_fact(self, user_id: str, fact_id: str) -> ProfileFact | None:
        """Promote a pending fact to active status."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM profile_facts WHERE id = ? AND user_id = ?",
                (fact_id, user_id),
            ).fetchone()
            if row is None:
                return None
            now = utc_now()
            self._execute(
                """
                UPDATE profile_facts
                SET status = 'archived', updated_at = ?
                WHERE user_id = ? AND key = ? AND status = 'active'
                """,
                (now, user_id, row["key"]),
            )
            self._execute(
                """
                UPDATE profile_facts
                SET status = 'active', source = 'user_confirmed', updated_at = ?
                WHERE id = ?
                """,
                (now, fact_id),
            )
            row = self._conn.execute("SELECT * FROM profile_facts WHERE id = ?", (fact_id,)).fetchone()
            return self._row_to_profile_fact(row)

    def add_memory_note(
        self,
        user_id: str,
        *,
        note: str,
        session_id: str | None = None,
        salience: float = 0.5,
        status: str = "active",
        source: str = "extractor",
        expires_at: str | None = None,
    ) -> MemoryNote:
        """Insert a middle-term memory note."""
        cleaned = note.strip()
        if not cleaned:
            raise ValueError("memory note cannot be empty")
        with self._lock:
            now = utc_now()
            note_id = f"note_{uuid.uuid4().hex}"
            self._execute(
                """
                INSERT INTO memory_notes (
                    id, user_id, session_id, note, salience, status, source, created_at, expires_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (note_id, user_id, session_id, cleaned, salience, status, source, now, expires_at),
            )
            row = self._conn.execute("SELECT * FROM memory_notes WHERE id = ?", (note_id,)).fetchone()
            return self._row_to_memory_note(row)

    def get_memory_notes(self, user_id: str, limit: int = 8) -> list[MemoryNote]:
        """Return active memory notes."""
        return self.list_memory_notes(user_id, statuses=("active",), limit=limit)

    def list_memory_notes(
        self,
        user_id: str,
        *,
        statuses: tuple[str, ...] = ("active",),
        limit: int = 8,
    ) -> list[MemoryNote]:
        """Return memory notes by status."""
        with self._lock:
            placeholders = ",".join("?" for _ in statuses)
            rows = self._conn.execute(
                f"""
                SELECT * FROM memory_notes
                WHERE user_id = ? AND status IN ({placeholders})
                ORDER BY salience DESC, created_at DESC
                LIMIT ?
                """,
                (user_id, *statuses, limit),
            ).fetchall()
            return [self._row_to_memory_note(row) for row in rows]

    def delete_memory_note(self, user_id: str, query: str) -> list[MemoryNote]:
        """Archive memory notes matching an id or text query."""
        cleaned = query.strip()
        if not cleaned:
            return []
        with self._lock:
            like = f"%{cleaned}%"
            rows = self._conn.execute(
                """
                SELECT * FROM memory_notes
                WHERE user_id = ? AND status = 'active' AND (id = ? OR note LIKE ?)
                LIMIT 20
                """,
                (user_id, cleaned, like),
            ).fetchall()
            ids = [row["id"] for row in rows]
            if ids:
                self._conn.executemany(
                    "UPDATE memory_notes SET status = 'archived' WHERE id = ?",
                    [(note_id,) for note_id in ids],
                )
                self._conn.commit()
            return [self._row_to_memory_note(row) for row in rows]

    def create_care_task(
        self,
        user_id: str,
        *,
        title: str,
        task_type: str = "reminder",
        due_at: str | None = None,
        recurrence_rule: str | None = None,
        status: str = "active",
        source: str = "tool",
        metadata: dict[str, Any] | None = None,
    ) -> CareTask:
        """Create a care task or reminder."""
        cleaned = title.strip()
        if not cleaned:
            raise ValueError("care task title cannot be empty")
        with self._lock:
            now = utc_now()
            task_id = f"task_{uuid.uuid4().hex}"
            self._execute(
                """
                INSERT INTO care_tasks (
                    id, user_id, title, task_type, due_at, recurrence_rule, status,
                    source, metadata_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    user_id,
                    cleaned,
                    task_type,
                    due_at,
                    recurrence_rule,
                    status,
                    source,
                    _json_dumps(metadata),
                    now,
                    now,
                ),
            )
            row = self._conn.execute("SELECT * FROM care_tasks WHERE id = ?", (task_id,)).fetchone()
            return self._row_to_care_task(row)

    def update_care_task(self, user_id: str, task_id: str, **updates: Any) -> CareTask | None:
        """Update a care task by id."""
        allowed = {"title", "task_type", "due_at", "recurrence_rule", "status", "metadata_json", "completed_at"}
        values: dict[str, Any] = {}
        for key, value in updates.items():
            if key == "metadata":
                values["metadata_json"] = _json_dumps(value)
            elif key in allowed:
                values[key] = value
        if not values:
            return self.get_care_task(user_id, task_id)
        values["updated_at"] = utc_now()
        with self._lock:
            assignments = ", ".join(f"{key} = ?" for key in values)
            params = [*values.values(), user_id, task_id]
            self._execute(
                f"UPDATE care_tasks SET {assignments} WHERE user_id = ? AND id = ?",
                params,
            )
            return self.get_care_task(user_id, task_id)

    def get_care_task(self, user_id: str, task_id: str) -> CareTask | None:
        """Return a care task by id."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM care_tasks WHERE user_id = ? AND id = ?",
                (user_id, task_id),
            ).fetchone()
            return self._row_to_care_task(row) if row else None

    def list_care_tasks(
        self,
        user_id: str,
        *,
        statuses: tuple[str, ...] = ("active",),
        due_before: str | None = None,
        query: str | None = None,
        limit: int = 20,
    ) -> list[CareTask]:
        """List care tasks by status, due time, or title query."""
        with self._lock:
            placeholders = ",".join("?" for _ in statuses)
            params: list[Any] = [user_id, *statuses]
            sql = f"SELECT * FROM care_tasks WHERE user_id = ? AND status IN ({placeholders})"
            if due_before is not None:
                sql += " AND (due_at IS NULL OR due_at <= ?)"
                params.append(due_before)
            if query:
                sql += " AND (id = ? OR title LIKE ? OR task_type LIKE ?)"
                like = f"%{query.strip()}%"
                params.extend([query.strip(), like, like])
            sql += " ORDER BY due_at IS NULL ASC, due_at ASC, updated_at DESC LIMIT ?"
            params.append(limit)
            rows = self._conn.execute(sql, params).fetchall()
            return [self._row_to_care_task(row) for row in rows]

    def complete_care_task(
        self,
        user_id: str,
        task_id: str,
        *,
        source_session_id: str | None = None,
    ) -> CareTask | None:
        """Mark a care task as completed."""
        task = self.get_care_task(user_id, task_id)
        if task is None:
            return None
        if task.recurrence_rule:
            self.complete_care_task_occurrence(user_id, task_id, source_session_id=source_session_id)
            return self.get_care_task(user_id, task_id)
        return self.update_care_task(
            user_id,
            task_id,
            status="completed",
            completed_at=utc_now(),
        )

    def complete_care_task_occurrence(
        self,
        user_id: str,
        task_id: str,
        *,
        occurrence_key: str | None = None,
        source_session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CareTaskOccurrence | None:
        """Record one completed occurrence for a recurring care task."""
        task = self.get_care_task(user_id, task_id)
        if task is None:
            return None
        key = (occurrence_key or utc_now()[:10]).strip()
        if not key:
            raise ValueError("occurrence key cannot be empty")
        with self._lock:
            now = utc_now()
            occurrence_id = f"occ_{uuid.uuid4().hex}"
            self._execute(
                """
                INSERT INTO care_task_occurrences (
                    id, task_id, user_id, occurrence_key, status, source_session_id,
                    metadata_json, created_at, updated_at, completed_at
                )
                VALUES (?, ?, ?, ?, 'completed', ?, ?, ?, ?, ?)
                ON CONFLICT(task_id, occurrence_key) DO UPDATE SET
                    status = 'completed',
                    source_session_id = COALESCE(excluded.source_session_id, source_session_id),
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at,
                    completed_at = excluded.completed_at
                """,
                (
                    occurrence_id,
                    task_id,
                    user_id,
                    key,
                    source_session_id,
                    _json_dumps(metadata),
                    now,
                    now,
                    now,
                ),
            )
            row = self._conn.execute(
                "SELECT * FROM care_task_occurrences WHERE task_id = ? AND occurrence_key = ?",
                (task_id, key),
            ).fetchone()
            return self._row_to_care_task_occurrence(row) if row else None

    def list_care_task_occurrences(
        self,
        user_id: str,
        *,
        statuses: tuple[str, ...] = ("completed",),
        task_id: str | None = None,
        limit: int = 50,
    ) -> list[CareTaskOccurrence]:
        """List recurring care task occurrence records."""
        with self._lock:
            placeholders = ",".join("?" for _ in statuses)
            params: list[Any] = [user_id, *statuses]
            sql = f"SELECT * FROM care_task_occurrences WHERE user_id = ? AND status IN ({placeholders})"
            if task_id is not None:
                sql += " AND task_id = ?"
                params.append(task_id)
            sql += " ORDER BY updated_at DESC LIMIT ?"
            params.append(limit)
            rows = self._conn.execute(sql, params).fetchall()
            return [self._row_to_care_task_occurrence(row) for row in rows]

    def disable_care_tasks(self, user_id: str, query: str) -> list[CareTask]:
        """Disable active care tasks matching a query."""
        tasks = self.list_care_tasks(user_id, query=query, limit=20)
        for task in tasks:
            self.update_care_task(user_id, task.id, status="disabled")
        return tasks

    def _row_to_user(self, row: sqlite3.Row) -> User:
        return User(
            id=row["id"],
            external_user_id=row["external_user_id"],
            display_name=row["display_name"],
            timezone=row["timezone"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_session(self, row: sqlite3.Row) -> SessionRecord:
        return SessionRecord(
            id=row["id"],
            user_id=row["user_id"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            status=row["status"],
            reason=row["reason"],
            summary=row["summary"],
            summary_json=_json_loads(row["summary_json"]),
            metadata=_json_loads(row["metadata_json"]),
        )

    def _row_to_turn(self, row: sqlite3.Row) -> Turn:
        return Turn(
            id=row["id"],
            session_id=row["session_id"],
            user_id=row["user_id"],
            role=row["role"],
            content=row["content"],
            created_at=row["created_at"],
            metadata=_json_loads(row["metadata_json"]),
        )

    def _row_to_profile_fact(self, row: sqlite3.Row) -> ProfileFact:
        return ProfileFact(
            id=row["id"],
            user_id=row["user_id"],
            key=row["key"],
            value=row["value"],
            category=row["category"],
            confidence=float(row["confidence"]),
            status=row["status"],
            source=row["source"],
            source_session_id=row["source_session_id"],
            evidence=row["evidence"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_memory_note(self, row: sqlite3.Row) -> MemoryNote:
        return MemoryNote(
            id=row["id"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            note=row["note"],
            salience=float(row["salience"]),
            status=row["status"],
            source=row["source"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
        )

    def _row_to_care_task(self, row: sqlite3.Row) -> CareTask:
        return CareTask(
            id=row["id"],
            user_id=row["user_id"],
            title=row["title"],
            task_type=row["task_type"],
            due_at=row["due_at"],
            recurrence_rule=row["recurrence_rule"],
            status=row["status"],
            source=row["source"],
            metadata=_json_loads(row["metadata_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            completed_at=row["completed_at"],
        )

    def _row_to_care_task_occurrence(self, row: sqlite3.Row) -> CareTaskOccurrence:
        return CareTaskOccurrence(
            id=row["id"],
            task_id=row["task_id"],
            user_id=row["user_id"],
            occurrence_key=row["occurrence_key"],
            status=row["status"],
            source_session_id=row["source_session_id"],
            metadata=_json_loads(row["metadata_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            completed_at=row["completed_at"],
        )
