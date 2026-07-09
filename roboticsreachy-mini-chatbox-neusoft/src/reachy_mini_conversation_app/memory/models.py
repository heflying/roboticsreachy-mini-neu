"""Data models for the elder-care memory runtime."""

from __future__ import annotations
from typing import Any
from dataclasses import field, dataclass


@dataclass(slots=True)
class User:
    """A local user profile owner."""

    id: str
    external_user_id: str
    display_name: str | None
    timezone: str
    created_at: str
    updated_at: str


@dataclass(slots=True)
class SessionRecord:
    """A persisted realtime conversation session."""

    id: str
    user_id: str
    started_at: str
    ended_at: str | None = None
    status: str = "active"
    reason: str | None = None
    summary: str | None = None
    summary_json: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Turn:
    """One final transcript turn from the user or assistant."""

    id: str
    session_id: str
    user_id: str
    role: str
    content: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SessionSummary:
    """Structured middle-term memory generated from one session."""

    session_id: str
    summary: str
    topics: list[str] = field(default_factory=list)
    emotions: list[str] = field(default_factory=list)
    follow_ups: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)

    def as_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "session_id": self.session_id,
            "summary": self.summary,
            "topics": self.topics,
            "emotions": self.emotions,
            "follow_ups": self.follow_ups,
            "risks": self.risks,
        }


@dataclass(slots=True)
class ProfileFact:
    """A long-term user profile fact or preference."""

    id: str
    user_id: str
    key: str
    value: str
    category: str
    confidence: float
    status: str
    source: str
    source_session_id: str | None
    evidence: str | None
    created_at: str
    updated_at: str


@dataclass(slots=True)
class MemoryNote:
    """A middle-term note that can be injected into future sessions."""

    id: str
    user_id: str
    session_id: str | None
    note: str
    salience: float
    status: str
    source: str
    created_at: str
    expires_at: str | None = None


@dataclass(slots=True)
class CareTask:
    """A reminder or care workflow item."""

    id: str
    user_id: str
    title: str
    task_type: str
    due_at: str | None
    recurrence_rule: str | None
    status: str
    source: str
    metadata: dict[str, Any]
    created_at: str
    updated_at: str
    completed_at: str | None = None


@dataclass(slots=True)
class CareTaskOccurrence:
    """One completion/skip instance for a recurring care task."""

    id: str
    task_id: str
    user_id: str
    occurrence_key: str
    status: str
    source_session_id: str | None
    metadata: dict[str, Any]
    created_at: str
    updated_at: str
    completed_at: str | None = None


@dataclass(slots=True)
class MemoryCandidate:
    """A profile fact candidate extracted by a model or explicit tool call."""

    key: str
    value: str
    category: str = "preference"
    confidence: float = 0.7
    source: str = "extractor"
    evidence: str | None = None
    requires_confirmation: bool | None = None


@dataclass(slots=True)
class CareTaskCandidate:
    """A care task candidate extracted from a transcript."""

    title: str
    task_type: str = "reminder"
    due_at: str | None = None
    recurrence_rule: str | None = None
    confidence: float = 0.7
    source: str = "extractor"
    evidence: str | None = None
    requires_confirmation: bool | None = None


@dataclass(slots=True)
class MemoryAction:
    """An explicit session-end CRUD action for stored memory."""

    action: str
    query: str | None = None
    key: str | None = None
    value: str | None = None
    category: str | None = None
    title: str | None = None
    task_type: str | None = None
    due_at: str | None = None
    recurrence_rule: str | None = None
    confidence: float = 0.7
    evidence: str | None = None
    requires_confirmation: bool | None = None


@dataclass(slots=True)
class ExtractionResult:
    """All memory material generated after a session ends."""

    summary: SessionSummary | None = None
    profile_candidates: list[MemoryCandidate] = field(default_factory=list)
    memory_notes: list[str] = field(default_factory=list)
    care_task_candidates: list[CareTaskCandidate] = field(default_factory=list)
    memory_actions: list[MemoryAction] = field(default_factory=list)
