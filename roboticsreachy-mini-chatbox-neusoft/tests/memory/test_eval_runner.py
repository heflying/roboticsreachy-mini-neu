import json

import pytest

from reachy_mini_conversation_app.memory.eval import run_case, run_cases, load_cases, render_markdown_report


@pytest.mark.asyncio
async def test_eval_runner_handles_router_memory(tmp_path, monkeypatch):
    """Router scenarios can be evaluated without real APIs."""
    monkeypatch.setenv("REACHY_MINI_MEMORY_WRITE_MODE", "hybrid")
    case = {
        "id": "LT-01",
        "title": "称呼偏好",
        "runner": "router",
        "turns": [{"role": "user", "text": "以后叫我张老师"}],
        "expect": {
            "profile_facts": [{"key": "preferred_name", "value": "张老师", "status": "active"}],
            "memory_context_contains": ["张老师"],
        },
    }

    result = await run_case(case, mode="case", db_path=tmp_path / "eval.db")

    assert result.status == "passed"
    assert result.metrics["context_chars"] > 0


@pytest.mark.asyncio
async def test_eval_runner_keeps_sensitive_extraction_out_of_context(tmp_path):
    """Mock extractor scenarios can verify pending sensitive memory policy."""
    case = {
        "id": "HE-04",
        "title": "用药提及",
        "runner": "extractor_mock",
        "turns": [{"role": "user", "text": "我今天早上吃了一片阿司匹林。"}],
        "mock_extraction": {
            "summary": {"summary": "用户提到未确认的用药信息。"},
            "profile_candidates": [
                {
                    "key": "medication.current",
                    "value": "阿司匹林",
                    "category": "medication",
                    "confidence": 0.9,
                }
            ],
        },
        "expect": {
            "profile_facts": [{"key": "medication.current", "status": "pending_confirmation"}],
            "memory_context_not_contains": ["阿司匹林"],
        },
    }

    result = await run_case(case, db_path=tmp_path / "eval.db")

    assert result.status == "passed"


@pytest.mark.asyncio
async def test_eval_runner_skips_real_api_without_permission(tmp_path):
    """Real Qwen extractor mode is guarded by allow_real_api."""
    result = await run_case(
        {
            "id": "REAL-01",
            "title": "真实抽取保护",
            "runner": "qwen-extractor",
            "turns": [{"role": "user", "text": "我喜欢豆浆。"}],
        },
        db_path=tmp_path / "eval.db",
        allow_real_api=False,
    )

    assert result.status == "skipped"
    assert "--allow-real-api" in (result.error or "")


@pytest.mark.asyncio
async def test_eval_runner_loads_cases_and_renders_report(tmp_path, monkeypatch):
    """Case files can be loaded and summarized into Markdown."""
    monkeypatch.setenv("REACHY_MINI_MEMORY_WRITE_MODE", "hybrid")
    case_file = tmp_path / "cases.json"
    case_file.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "LT-01",
                        "title": "称呼偏好",
                        "runner": "router",
                        "turns": [{"role": "user", "text": "以后叫我张老师"}],
                        "expect": {"memory_context_contains": ["张老师"]},
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    results = await run_cases(load_cases(case_file), db_path=tmp_path / "dbs", keep_db=True)
    report = render_markdown_report(results)

    assert results[0].status == "passed"
    assert "Memory Evaluation Report" in report
