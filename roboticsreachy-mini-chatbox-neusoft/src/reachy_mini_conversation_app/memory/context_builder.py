"""Build compact memory context for the next Qwen realtime session."""

from __future__ import annotations
from datetime import datetime

from reachy_mini_conversation_app.memory.store import SQLiteMemoryStore
from reachy_mini_conversation_app.memory.models import ProfileFact


REDACTED_MEMORY_TEXT = "[已删除或待确认信息]"
SENSITIVE_CONTEXT_TERMS = {
    "头晕",
    "胃",
    "胃不舒服",
    "膝盖",
    "胸口痛",
    "喘不过气",
    "血压",
    "血糖",
    "降压药",
    "阿司匹林",
    "青霉素",
    "过敏",
    "电话",
    "手机号",
    "地址",
    "住址",
    "家庭住址",
    "住在",
    "门牌",
    "门牌号",
    "银行卡",
    "密码",
    "保证金",
}
CATEGORY_CONTEXT_BLOCK_TERMS = {
    "address": {"地址", "住址", "家庭住址", "住在", "门牌", "门牌号"},
    "contact": {"电话", "手机号"},
    "phone": {"电话", "手机号"},
    "financial": {"银行卡", "密码", "保证金"},
    "medication": {"降压药", "阿司匹林", "青霉素", "过敏"},
    "health": {"头晕", "胃", "胃不舒服", "膝盖", "胸口痛", "喘不过气", "血压", "血糖"},
}


class MemoryContextBuilder:
    """Render active memory into a short instruction block."""

    def __init__(
        self,
        store: SQLiteMemoryStore,
        *,
        max_profile_facts: int = 24,
        max_session_summaries: int = 4,
        max_memory_notes: int = 6,
        max_care_tasks: int = 8,
        max_completed_occurrences: int = 4,
        max_chars: int = 3000,
    ):
        """Create a context builder."""
        self.store = store
        self.max_profile_facts = max_profile_facts
        self.max_session_summaries = max_session_summaries
        self.max_memory_notes = max_memory_notes
        self.max_care_tasks = max_care_tasks
        self.max_completed_occurrences = max_completed_occurrences
        self.max_chars = max_chars

    def build(self, user_id: str, *, now: datetime | None = None) -> str:
        """Return a compact memory block to append to realtime instructions."""
        profile = self.store.search_profile_facts(user_id, statuses=("active",), limit=self.max_profile_facts)
        blocked_facts = self.store.search_profile_facts(
            user_id,
            statuses=("pending_confirmation", "archived"),
            limit=100,
        )
        blocked_terms = sorted(
            set(_blocked_terms_from_facts(blocked_facts)).union(SENSITIVE_CONTEXT_TERMS),
            key=len,
            reverse=True,
        )
        sessions = self.store.get_recent_sessions(user_id, limit=self.max_session_summaries)
        notes = self.store.get_memory_notes(user_id, limit=self.max_memory_notes)
        care_tasks = self.store.list_care_tasks(user_id, statuses=("active",), limit=self.max_care_tasks)
        completed_occurrences = self.store.list_care_task_occurrences(
            user_id,
            statuses=("completed",),
            limit=self.max_completed_occurrences,
        )

        if not profile and not sessions and not notes and not care_tasks and not completed_occurrences:
            return ""

        lines: list[str] = [
            "",
            "[Memory context / 记忆上下文 - 本轮对话可用的已确认背景]",
            "下面只包含 active/confirmed 记忆。请自然使用这些记忆回答用户，不要朗读标题，也不要透露数据库细节。",
            "当用户直接问“你记得/还记得/怎么称呼我/我喜欢什么/提醒什么”时，必须优先根据下面的相关条目回答。",
            "pending_confirmation、未确认、已取消、已完成、已删除的信息不会作为事实列在这里；不要猜测它们仍然有效。",
        ]
        if profile:
            lines.append("长期用户画像（可直接用于回答）：")
            family_overview = _render_family_overview(profile)
            if family_overview:
                lines.append(f"- family.overview (family): {family_overview}")
            for fact in profile:
                if family_overview and fact.key.startswith("family."):
                    continue
                lines.append(f"- {fact.key} ({fact.category}): {fact.value}")
        if sessions or notes:
            lines.append("近期跨会话上下文：")
            for session in sessions:
                if session.summary:
                    summary = _sanitize_context_text(session.summary, blocked_terms)
                    if _has_useful_context(summary):
                        lines.append(f"- {summary}")
            for note in notes:
                sanitized_note = _sanitize_context_text(note.note, blocked_terms)
                if _has_useful_context(sanitized_note):
                    lines.append(f"- {sanitized_note}")
        if care_tasks:
            lines.append("今日或仍有效的照护提醒：")
            for task in care_tasks:
                due = f" 到期时间 {task.due_at}" if task.due_at else ""
                repeat = f" 重复规则 {task.recurrence_rule}" if task.recurrence_rule else ""
                lines.append(f"- [{task.task_type}] {task.title}{due}{repeat}")
        if completed_occurrences:
            lines.append("今日已完成的重复提醒实例（不表示未来重复提醒已取消）：")
            for occurrence in completed_occurrences:
                task = self.store.get_care_task(user_id, occurrence.task_id)
                title = task.title if task is not None else occurrence.task_id
                lines.append(f"- {title}：{occurrence.occurrence_key} 这次已完成；若该任务有重复规则，后续提醒仍有效。")
        lines.append("记忆策略：如果用户纠正、删除或拒绝某条记忆，必须立即服从新的说法。")

        rendered = "\n".join(lines)
        if len(rendered) <= self.max_chars:
            return rendered
        return f"{rendered[: self.max_chars - 120]}\n[Memory context truncated for realtime budget.]"


def _blocked_terms_from_facts(facts: list[ProfileFact]) -> list[str]:
    """Return values that must not leak through summaries or notes."""
    terms: set[str] = set()
    for fact in facts:
        value = fact.value.strip()
        if 1 < len(value) <= 80:
            terms.add(value)
        if fact.status == "pending_confirmation" or fact.category in {
            "health",
            "medication",
            "contact",
            "address",
            "phone",
            "financial",
            "legal",
            "safety",
            "emergency",
        }:
            terms.update(CATEGORY_CONTEXT_BLOCK_TERMS.get(fact.category, set()))
            for token in SENSITIVE_CONTEXT_TERMS:
                if token in value or (fact.evidence and token in fact.evidence):
                    terms.add(token)
    return sorted(terms, key=len, reverse=True)


def _render_family_overview(profile: list[ProfileFact]) -> str:
    facts = {fact.key: fact.value for fact in profile if fact.status == "active"}
    parts: list[str] = []
    daughter = facts.get("family.daughter.name")
    son = facts.get("family.son.name")
    grandchild = facts.get("family.grandchild.name")
    visit = facts.get("family.visit_pattern")
    if daughter:
        daughter_text = f"女儿{daughter}"
        if visit:
            daughter_text = f"{daughter_text}（{visit}）"
        parts.append(daughter_text)
    elif visit:
        parts.append(visit)
    if son:
        parts.append(f"儿子{son}")
    if grandchild:
        parts.append(f"外孙{grandchild}")
    return "；".join(parts)


def _sanitize_context_text(text: str, blocked_terms: list[str]) -> str:
    """Redact blocked terms before injecting text into realtime instructions."""
    sanitized = text
    for term in blocked_terms:
        if term:
            sanitized = sanitized.replace(term, REDACTED_MEMORY_TEXT)
    return " ".join(sanitized.split())


def _has_useful_context(text: str) -> bool:
    """Return whether a sanitized note/summary is still worth injecting."""
    stripped = text.strip()
    if not stripped:
        return False
    if REDACTED_MEMORY_TEXT in stripped:
        return False
    return stripped != REDACTED_MEMORY_TEXT
