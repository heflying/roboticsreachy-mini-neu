"""Small local replies for high-frequency dialog checks."""

from __future__ import annotations


def get_quick_reply(text: str) -> str | None:
    """Return a deterministic short reply for simple, stateless utterances."""
    normalized = "".join(ch for ch in text.strip().lower() if not ch.isspace())
    if not normalized:
        return None

    # if any(key in normalized for key in ("你是谁", "介绍一下自己")):
    #     return "我是 Reachy Mini，可以和你语音聊天的小机器人。"

    # if any(key in normalized for key in ("在吗", "听得见", "能听见", "听到吗", "能听到")):
    #     return "我在，能听见你。"

    # if any(key in normalized for key in ("反应快", "快吗", "有点慢", "太慢")):
    #     return "我会尽量简短回答，让对话更快。"

    # if normalized in {"好的", "好", "嗯", "呵呵", "哈哈"}:
    #     return "好的。"

    return None
