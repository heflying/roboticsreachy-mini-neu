"""Tests for Qwen cascade provider registration."""

from __future__ import annotations
from pathlib import Path


def test_qwen_entries_are_registered_in_static_config():
    """The repository config advertises Qwen providers and install extra."""
    cascade_yaml = Path("cascade.yaml").read_text(encoding="utf-8")
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "qwen_realtime_asr:" in cascade_yaml
    assert "provider: qwen-flash" in cascade_yaml
    assert "qwen-plus:" in cascade_yaml
    assert "qwen-flash:" in cascade_yaml
    assert "model: qwen-flash" in cascade_yaml
    assert "qwen_realtime_tts:" in cascade_yaml
    assert "DASHSCOPE_API_KEY" in cascade_yaml
    assert 'cascade_qwen = ["websockets>=13.0"]' in pyproject
    assert '"DASHSCOPE_API_KEY": config.DASHSCOPE_API_KEY' in Path(
        "src/reachy_mini_conversation_app/cascade/provider_factory.py"
    ).read_text(encoding="utf-8")
    assert "self.DASHSCOPE_API_KEY = os.getenv(\"DASHSCOPE_API_KEY\")" in Path(
        "src/reachy_mini_conversation_app/cascade/config.py"
    ).read_text(encoding="utf-8")
