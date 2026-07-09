"""Safety and confirmation policy for elder-care memory writes."""

from __future__ import annotations
from dataclasses import dataclass

from reachy_mini_conversation_app.memory.models import MemoryCandidate, CareTaskCandidate


SENSITIVE_CATEGORIES: set[str] = {
    "health",
    "medication",
    "safety",
    "emergency",
    "contact",
    "address",
    "phone",
    "financial",
    "legal",
}

CONFIRMED_SOURCES: set[str] = {
    "user_confirmed",
    "caregiver_confirmed",
    "tool_confirmed",
}

SENSITIVE_CARE_TASK_TERMS: tuple[str, ...] = (
    "血压",
    "血糖",
    "降压药",
    "阿司匹林",
    "服药",
    "吃药",
    "药",
    "头晕",
    "听力",
)

SENSITIVE_NOTE_TERMS: tuple[str, ...] = (
    "血压",
    "血糖",
    "降压药",
    "阿司匹林",
    "服药",
    "吃药",
    "药",
    "头晕",
    "胃",
    "胃不舒服",
    "膝盖",
    "胸口痛",
    "喘不过气",
    "住址",
    "地址",
    "门牌",
    "电话",
    "手机号",
    "保证金",
    "银行卡",
    "密码",
)


@dataclass(slots=True)
class SafetyDecision:
    """The persistence decision for a memory candidate."""

    status: str
    reason: str

    @property
    def should_store(self) -> bool:
        """Whether the candidate should be persisted."""
        return self.status in {"active", "pending_confirmation"}


class MemorySafetyFilter:
    """Classify candidates into active, pending, or rejected writes."""

    min_confidence: float = 0.45

    def evaluate_profile_candidate(self, candidate: MemoryCandidate) -> SafetyDecision:
        """Evaluate a long-term profile candidate."""
        if not candidate.key.strip() or not candidate.value.strip():
            return SafetyDecision("rejected", "empty key or value")
        if candidate.confidence < self.min_confidence:
            return SafetyDecision("rejected", "low confidence")
        if candidate.requires_confirmation is True:
            return SafetyDecision("pending_confirmation", "candidate requested confirmation")
        if candidate.category in SENSITIVE_CATEGORIES and candidate.source not in CONFIRMED_SOURCES:
            return SafetyDecision("pending_confirmation", "sensitive memory requires confirmation")
        return SafetyDecision("active", "safe to store")

    def evaluate_care_task_candidate(self, candidate: CareTaskCandidate) -> SafetyDecision:
        """Evaluate a care task candidate."""
        if not candidate.title.strip():
            return SafetyDecision("rejected", "empty title")
        if candidate.confidence < self.min_confidence:
            return SafetyDecision("rejected", "low confidence")
        if candidate.requires_confirmation is True:
            return SafetyDecision("pending_confirmation", "candidate requested confirmation")
        if candidate.task_type == "safety" and candidate.source not in CONFIRMED_SOURCES:
            return SafetyDecision("pending_confirmation", "safety task requires confirmation")
        if _is_sensitive_care_task(candidate) and candidate.source not in CONFIRMED_SOURCES:
            return SafetyDecision("pending_confirmation", "sensitive care task requires confirmation")
        if candidate.task_type in {"medication", "emergency", "medical"} and candidate.source not in CONFIRMED_SOURCES:
            return SafetyDecision("pending_confirmation", "sensitive care task requires confirmation")
        return SafetyDecision("active", "safe to store")

    def evaluate_memory_note(self, note: str, *, source: str = "extractor") -> SafetyDecision:
        """Evaluate whether a middle-term note may be injected as active context."""
        cleaned = note.strip()
        if not cleaned:
            return SafetyDecision("rejected", "empty note")
        if source not in CONFIRMED_SOURCES and any(term in cleaned for term in SENSITIVE_NOTE_TERMS):
            return SafetyDecision("pending_confirmation", "sensitive note requires confirmation")
        return SafetyDecision("active", "safe to store")


def _is_sensitive_care_task(candidate: CareTaskCandidate) -> bool:
    scoped_text = "\n".join(part for part in (candidate.title, candidate.evidence or "") if part)
    return any(term in scoped_text for term in SENSITIVE_CARE_TASK_TERMS)
