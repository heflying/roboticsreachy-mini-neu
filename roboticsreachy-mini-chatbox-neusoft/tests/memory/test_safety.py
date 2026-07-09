from reachy_mini_conversation_app.memory.models import MemoryCandidate, CareTaskCandidate
from reachy_mini_conversation_app.memory.safety import MemorySafetyFilter


def test_sensitive_profile_fact_requires_confirmation():
    """Health memories extracted from transcript stay pending."""
    decision = MemorySafetyFilter().evaluate_profile_candidate(
        MemoryCandidate(
            key="health.blood_pressure",
            value="午饭后容易头晕",
            category="health",
            confidence=0.8,
            source="extractor",
        )
    )

    assert decision.status == "pending_confirmation"


def test_confirmed_sensitive_profile_fact_can_be_active():
    """Confirmed medication memories can be active."""
    decision = MemorySafetyFilter().evaluate_profile_candidate(
        MemoryCandidate(
            key="medication.current",
            value="阿司匹林",
            category="medication",
            confidence=0.95,
            source="user_confirmed",
        )
    )

    assert decision.status == "active"


def test_low_confidence_candidate_is_rejected():
    """Low-confidence care candidates are rejected."""
    decision = MemorySafetyFilter().evaluate_care_task_candidate(
        CareTaskCandidate(title="可能要提醒什么", confidence=0.2)
    )

    assert decision.status == "rejected"


def test_sensitive_care_task_terms_require_confirmation():
    """Health-specific care tasks should not become active reminders automatically."""
    decision = MemorySafetyFilter().evaluate_care_task_candidate(
        CareTaskCandidate(
            title="复查血压",
            task_type="appointment",
            confidence=0.9,
            source="extractor_action",
            evidence="医生说血压有点高",
        )
    )

    assert decision.status == "pending_confirmation"
