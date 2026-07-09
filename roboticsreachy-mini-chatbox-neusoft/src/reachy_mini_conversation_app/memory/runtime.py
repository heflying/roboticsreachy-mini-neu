"""Runtime gateway for Qwen realtime memory orchestration."""

from __future__ import annotations
import os
import asyncio
import logging
from typing import Any
from pathlib import Path

from reachy_mini_conversation_app.memory.store import SQLiteMemoryStore, utc_now
from reachy_mini_conversation_app.memory.models import (
    Turn,
    CareTask,
    MemoryNote,
    ProfileFact,
    MemoryAction,
    MemoryCandidate,
    CareTaskCandidate,
)
from reachy_mini_conversation_app.memory.safety import MemorySafetyFilter
from reachy_mini_conversation_app.memory.extractors import MemoryExtractor, create_default_extractor
from reachy_mini_conversation_app.memory.context_builder import MemoryContextBuilder


logger = logging.getLogger(__name__)

_GLOBAL_MEMORY_RUNTIME: "MemoryRuntime | None" = None


class MemoryRuntime:
    """Single entrypoint for session memory, profile memory, and care tasks."""

    def __init__(
        self,
        store: SQLiteMemoryStore,
        *,
        user_external_id: str = "default",
        user_display_name: str | None = None,
        timezone_name: str = "Asia/Shanghai",
        extractor: MemoryExtractor | None = None,
        safety_filter: MemorySafetyFilter | None = None,
        enabled: bool = True,
    ):
        """Create a memory runtime."""
        self.store = store
        self.enabled = enabled
        self.user = self.store.get_or_create_user(
            external_user_id=user_external_id,
            display_name=user_display_name,
            timezone_name=timezone_name,
        )
        self.extractor = extractor or create_default_extractor()
        self.safety_filter = safety_filter or MemorySafetyFilter()
        self.context_builder = MemoryContextBuilder(self.store)
        self.current_session_id: str | None = None
        self._ending_session_ids: set[str] = set()
        self._background_extraction_tasks: set[asyncio.Task[None]] = set()

    def start_session(self, metadata: dict[str, Any] | None = None) -> str | None:
        """Start a memory session if needed."""
        if not self.enabled:
            return None
        if self.current_session_id is not None:
            return self.current_session_id
        session = self.store.start_session(
            self.user.id,
            metadata={
                "provider": "qwen_realtime",
                **(metadata or {}),
            },
        )
        self.current_session_id = session.id
        logger.info("Memory session started: %s", session.id)
        return session.id

    def record_turn(self, role: str, content: str, metadata: dict[str, Any] | None = None) -> Turn | None:
        """Persist a final transcript turn without doing extraction."""
        if not self.enabled or not content.strip():
            return None
        session_id = self.start_session()
        if session_id is None:
            return None
        return self.store.append_turn(session_id, self.user.id, role, content, metadata=metadata)

    def record_user_transcript(self, content: str, metadata: dict[str, Any] | None = None) -> Turn | None:
        """Persist a final user transcript."""
        return self.record_turn("user", content, metadata=metadata)

    def record_assistant_transcript(self, content: str, metadata: dict[str, Any] | None = None) -> Turn | None:
        """Persist a final assistant transcript."""
        return self.record_turn("assistant", content, metadata=metadata)

    async def end_session(self, *, reason: str = "closed") -> None:
        """End the active session and run model-based extraction out of the realtime path."""
        session_id = self._claim_current_session_id()
        if session_id is None:
            return
        await self._finalize_session(session_id, reason=reason)

    def end_session_background(self, *, reason: str = "closed") -> asyncio.Task[None] | None:
        """End the active session and schedule extraction without blocking realtime shutdown."""
        session_id = self._claim_current_session_id()
        if session_id is None:
            return None
        self.store.end_session(session_id, reason=f"{reason}: extraction_pending")
        task = asyncio.create_task(
            self._finalize_session(session_id, reason=reason),
            name=f"memory-extract-{session_id}",
        )
        self._background_extraction_tasks.add(task)
        task.add_done_callback(self._background_extraction_tasks.discard)
        return task

    async def wait_for_pending_extractions(self, *, timeout_s: float | None = None) -> None:
        """Wait for currently scheduled background extraction tasks."""
        tasks = [task for task in self._background_extraction_tasks if not task.done()]
        if not tasks:
            return
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout_s)

    def _claim_current_session_id(self) -> str | None:
        session_id = self.current_session_id
        if not self.enabled or session_id is None:
            return None
        if session_id in self._ending_session_ids:
            return None
        self._ending_session_ids.add(session_id)
        self.current_session_id = None
        return session_id

    async def _finalize_session(self, session_id: str, *, reason: str) -> None:
        """Run extraction and apply writes for a claimed session id."""
        turns = self.store.get_turns(session_id)
        try:
            extraction = None
            if turns:
                memory_context = self.build_memory_context()
                extraction = await self.extractor.extract(
                    session_id=session_id,
                    turns=turns,
                    memory_context=memory_context,
                )

            if extraction and extraction.summary:
                self.store.end_session(
                    session_id,
                    reason=reason,
                    summary=extraction.summary.summary,
                    summary_json=extraction.summary.as_json(),
                )
            else:
                self.store.end_session(session_id, reason=reason)

            if extraction:
                for action in extraction.memory_actions:
                    self._apply_memory_action(action, source_session_id=session_id)
                for note in extraction.memory_notes:
                    decision = self.safety_filter.evaluate_memory_note(note, source="extractor")
                    if decision.should_store:
                        self.store.add_memory_note(
                            self.user.id,
                            note=note,
                            session_id=session_id,
                            salience=0.7,
                            status=decision.status,
                            source="extractor",
                        )
                for candidate in extraction.profile_candidates:
                    self._store_profile_candidate(candidate, source_session_id=session_id)
                for candidate in extraction.care_task_candidates:
                    self._store_care_task_candidate(candidate)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Memory extraction failed for session %s: %s", session_id, exc)
            self.store.end_session(session_id, reason=f"{reason}: extraction_failed")
        finally:
            self._ending_session_ids.discard(session_id)

    def build_memory_context(self) -> str:
        """Build active memory context for the next Qwen realtime session."""
        if not self.enabled:
            return ""
        return self.context_builder.build(self.user.id)

    def remember_user_fact(
        self,
        *,
        key: str,
        value: str,
        category: str = "preference",
        confidence: float = 0.9,
        source: str = "tool",
        evidence: str | None = None,
        confirmed: bool = False,
    ) -> ProfileFact | None:
        """Create or update a long-term profile fact through the safety filter."""
        candidate = MemoryCandidate(
            key=key,
            value=value,
            category=category,
            confidence=confidence,
            source="user_confirmed" if confirmed else source,
            evidence=evidence,
        )
        return self._store_profile_candidate(candidate, source_session_id=self.current_session_id)

    def update_user_fact(
        self,
        *,
        key_or_id: str,
        value: str,
        category: str | None = None,
        confirmed: bool = False,
    ) -> ProfileFact | None:
        """Update an existing fact by id or key."""
        matches = self.store.search_profile_facts(
            self.user.id,
            key_or_id,
            statuses=("active", "pending_confirmation"),
            limit=1,
        )
        key = matches[0].key if matches else key_or_id
        category_name = category or (matches[0].category if matches else "preference")
        return self.remember_user_fact(
            key=key,
            value=value,
            category=category_name,
            source="tool",
            confirmed=confirmed,
        )

    def forget_user_fact(self, query: str) -> list[ProfileFact]:
        """Archive matching profile facts."""
        return self.store.archive_profile_fact(self.user.id, query)

    def recall_user_memory(self, query: str | None = None, *, include_pending: bool = False) -> dict[str, Any]:
        """Return matching profile facts, memory notes, and care tasks."""
        statuses = ("active", "pending_confirmation") if include_pending else ("active",)
        facts = self.store.search_profile_facts(self.user.id, query, statuses=statuses, limit=20)
        tasks = self.store.list_care_tasks(self.user.id, statuses=statuses, query=query, limit=20)
        notes = self.store.get_memory_notes(self.user.id, limit=8) if not query else self.store.get_memory_notes(self.user.id, limit=20)
        if query:
            notes = [note for note in notes if query in note.note]
        return {
            "facts": [self._fact_to_dict(fact) for fact in facts],
            "notes": [self._note_to_dict(note) for note in notes],
            "care_tasks": [self._task_to_dict(task) for task in tasks],
        }

    def list_user_profile(self, *, include_pending: bool = False) -> list[dict[str, Any]]:
        """List stored user profile facts."""
        statuses = ("active", "pending_confirmation") if include_pending else ("active",)
        return [
            self._fact_to_dict(fact)
            for fact in self.store.search_profile_facts(self.user.id, statuses=statuses, limit=100)
        ]

    def create_care_task(
        self,
        *,
        title: str,
        task_type: str = "reminder",
        due_at: str | None = None,
        recurrence_rule: str | None = None,
        source: str = "tool",
        confirmed: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> CareTask | None:
        """Create a care task through the safety filter."""
        candidate = CareTaskCandidate(
            title=title,
            task_type=task_type,
            due_at=due_at,
            recurrence_rule=recurrence_rule,
            confidence=0.95,
            source="user_confirmed" if confirmed else source,
        )
        decision = self.safety_filter.evaluate_care_task_candidate(candidate)
        if not decision.should_store:
            logger.info("Rejected care task candidate %r: %s", title, decision.reason)
            return None
        return self.store.create_care_task(
            self.user.id,
            title=title,
            task_type=task_type,
            due_at=due_at,
            recurrence_rule=recurrence_rule,
            status=decision.status,
            source=candidate.source,
            metadata={
                "safety_reason": decision.reason,
                **(metadata or {}),
            },
        )

    def update_care_task(self, task_id: str, **updates: Any) -> CareTask | None:
        """Update a care task by id."""
        return self.store.update_care_task(self.user.id, task_id, **updates)

    def complete_care_task(
        self,
        *,
        task_id: str | None = None,
        query: str | None = None,
        source_session_id: str | None = None,
    ) -> CareTask | None:
        """Complete a care task by id or query."""
        resolved_task_id = task_id
        if resolved_task_id is None and query:
            matches = self.store.list_care_tasks(self.user.id, query=query, limit=1)
            resolved_task_id = matches[0].id if matches else None
        if resolved_task_id is None:
            return None
        return self.store.complete_care_task(
            self.user.id,
            resolved_task_id,
            source_session_id=source_session_id or self.current_session_id,
        )

    def disable_care_tasks(self, query: str) -> list[CareTask]:
        """Disable care tasks matching a query."""
        tasks = self.store.disable_care_tasks(self.user.id, query)
        if tasks:
            return tasks
        tombstone = self.store.create_care_task(
            self.user.id,
            title=query,
            task_type="reminder",
            status="disabled",
            source="user_confirmed",
            metadata={"tombstone": True, "query": query, "created_at_runtime": utc_now()},
        )
        return [tombstone]

    def list_today_care_tasks(self, *, include_completed: bool = False) -> list[dict[str, Any]]:
        """List active care tasks for model/tool use."""
        statuses = ("active", "pending_confirmation", "completed") if include_completed else ("active",)
        return [self._task_to_dict(task) for task in self.store.list_care_tasks(self.user.id, statuses=statuses)]

    def _store_profile_candidate(
        self,
        candidate: MemoryCandidate,
        *,
        source_session_id: str | None,
    ) -> ProfileFact | None:
        decision = self.safety_filter.evaluate_profile_candidate(candidate)
        if not decision.should_store:
            logger.info("Rejected profile candidate %r: %s", candidate.key, decision.reason)
            return None
        return self.store.upsert_profile_fact(
            self.user.id,
            key=candidate.key,
            value=candidate.value,
            category=candidate.category,
            confidence=candidate.confidence,
            status=decision.status,
            source=candidate.source,
            source_session_id=source_session_id,
            evidence=candidate.evidence,
        )

    def _store_care_task_candidate(self, candidate: CareTaskCandidate) -> CareTask | None:
        decision = self.safety_filter.evaluate_care_task_candidate(candidate)
        if not decision.should_store:
            logger.info("Rejected care task candidate %r: %s", candidate.title, decision.reason)
            return None
        existing = self._find_care_task(candidate.title, statuses=("active", "pending_confirmation"))
        if existing is not None and existing.title == candidate.title:
            logger.info("Skipped duplicate care task candidate %r", candidate.title)
            return existing
        return self.store.create_care_task(
            self.user.id,
            title=candidate.title,
            task_type=candidate.task_type,
            due_at=candidate.due_at,
            recurrence_rule=candidate.recurrence_rule,
            status=decision.status,
            source=candidate.source,
            metadata={
                "evidence": candidate.evidence,
                "safety_reason": decision.reason,
                "created_at_runtime": utc_now(),
            },
        )

    def _apply_memory_action(self, action: MemoryAction, *, source_session_id: str | None = None) -> None:
        if action.confidence < self.safety_filter.min_confidence:
            logger.info("Skipped low-confidence memory action %r", action)
            return

        if action.action == "forget_user_fact":
            query = _first_text(action.query, action.value, action.key)
            if query:
                self.forget_user_fact(query)
                self.store.delete_memory_note(self.user.id, query)
            return

        if action.action == "delete_memory_note":
            query = _first_text(action.query, action.value, action.title)
            if query:
                self.store.delete_memory_note(self.user.id, query)
            return

        if action.action == "remember_user_fact":
            if action.key and action.value:
                self.remember_user_fact(
                    key=action.key,
                    value=action.value,
                    category=action.category or "preference",
                    confidence=action.confidence,
                    source="extractor_action",
                    evidence=action.evidence,
                    confirmed=action.requires_confirmation is False,
                )
            return

        if action.action == "update_user_fact":
            key_or_id = _first_text(action.key, action.query)
            if key_or_id and action.value:
                self.update_user_fact(
                    key_or_id=key_or_id,
                    value=action.value,
                    category=action.category,
                    confirmed=action.requires_confirmation is False,
                )
            return

        if action.action == "create_care_task":
            if action.title:
                self._store_care_task_candidate(
                    CareTaskCandidate(
                        title=action.title,
                        task_type=action.task_type or "reminder",
                        due_at=action.due_at,
                        recurrence_rule=action.recurrence_rule,
                        confidence=action.confidence,
                        source="user_confirmed" if action.requires_confirmation is False else "extractor_action",
                        evidence=action.evidence,
                        requires_confirmation=action.requires_confirmation,
                    )
                )
            return

        if action.action == "complete_care_task":
            query = _first_text(action.query, action.title, action.value)
            if query:
                self.complete_care_task(query=query, source_session_id=source_session_id)
            return

        if action.action == "disable_care_task":
            query = _first_text(action.query, action.title, action.value)
            if query:
                self.disable_care_tasks(query)
            return

        if action.action == "update_care_task":
            query = _first_text(action.query, action.title, action.value)
            if query:
                task = self._find_care_task(query, statuses=("active", "pending_confirmation"))
                if task is not None:
                    updates: dict[str, Any] = {}
                    if action.title:
                        updates["title"] = action.title
                    if action.task_type:
                        updates["task_type"] = action.task_type
                    if action.due_at is not None:
                        updates["due_at"] = action.due_at
                    if action.recurrence_rule is not None:
                        updates["recurrence_rule"] = action.recurrence_rule
                    if updates:
                        self.update_care_task(task.id, **updates)
            return

        logger.info("Ignored unsupported memory action: %s", action.action)

    def _find_care_task(
        self,
        query: str,
        *,
        statuses: tuple[str, ...] = ("active",),
    ) -> CareTask | None:
        matches = self._find_care_tasks(query, statuses=statuses, limit=1)
        return matches[0] if matches else None

    def _find_care_tasks(
        self,
        query: str,
        *,
        statuses: tuple[str, ...] = ("active",),
        limit: int = 20,
    ) -> list[CareTask]:
        return self.store.list_care_tasks(self.user.id, statuses=statuses, query=query, limit=limit)

    @staticmethod
    def _fact_to_dict(fact: ProfileFact) -> dict[str, Any]:
        return {
            "id": fact.id,
            "key": fact.key,
            "value": fact.value,
            "category": fact.category,
            "confidence": fact.confidence,
            "status": fact.status,
            "source": fact.source,
            "evidence": fact.evidence,
        }

    @staticmethod
    def _note_to_dict(note: MemoryNote) -> dict[str, Any]:
        return {
            "id": note.id,
            "note": note.note,
            "salience": note.salience,
            "status": note.status,
            "source": note.source,
        }

    @staticmethod
    def _task_to_dict(task: CareTask) -> dict[str, Any]:
        return {
            "id": task.id,
            "title": task.title,
            "task_type": task.task_type,
            "due_at": task.due_at,
            "recurrence_rule": task.recurrence_rule,
            "status": task.status,
            "source": task.source,
            "completed_at": task.completed_at,
        }


def set_global_memory_runtime(runtime: MemoryRuntime | None) -> None:
    """Set the process-global runtime used by tools when needed."""
    global _GLOBAL_MEMORY_RUNTIME
    _GLOBAL_MEMORY_RUNTIME = runtime


def get_global_memory_runtime() -> MemoryRuntime | None:
    """Return the process-global runtime."""
    return _GLOBAL_MEMORY_RUNTIME


def create_default_memory_runtime(instance_path: str | Path | None = None) -> MemoryRuntime:
    """Create and register the default runtime for Qwen sessions."""
    enabled = os.getenv("REACHY_MINI_MEMORY_ENABLED", "1").strip().lower() not in {"0", "false", "off", "no"}
    db_path = _resolve_default_db_path(instance_path)
    runtime = MemoryRuntime(SQLiteMemoryStore(db_path), enabled=enabled)
    set_global_memory_runtime(runtime)
    return runtime


def _resolve_default_db_path(instance_path: str | Path | None = None) -> Path:
    configured = os.getenv("REACHY_MINI_MEMORY_DB_PATH")
    if configured:
        return Path(configured)
    if instance_path:
        return Path(instance_path) / "reachy_memory.sqlite3"
    return Path.home() / ".reachy_mini_conversation_app" / "reachy_memory.sqlite3"


def _first_text(*values: str | None) -> str | None:
    for value in values:
        cleaned = (value or "").strip()
        if cleaned:
            return cleaned
    return None
