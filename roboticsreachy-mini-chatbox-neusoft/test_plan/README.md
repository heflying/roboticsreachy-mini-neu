# 记忆测试计划

这个目录用于集中存放养老场景记忆能力的测试设计、测试用例和当前测试报告，覆盖 Qwen realtime 记忆链路。

## 目录内容

- `test_design.md`：测试策略、测试观点、通过标准和性能指标。
- `cases/single_turn_memory_cases.md`：单体 turn 测试用例，覆盖记忆增删改查、安全 gating、上下文注入和 recall。
- `cases/long_conversation_memory_cases.md`：长对话测试用例，覆盖真实养老陪伴对话里的混合信息、修正、取消、完成和敏感信息。
- `reports/qwen_memory_eval_report_20260507.md`：当前最新一轮本地测试和真实 Qwen 测试报告。

## 当前测试范围

当前产品决策是 extractor-first，也就是优先依赖 session end extractor 完成记忆写入：

- realtime session 会把用户和 assistant 的最终转写文本写入 SQLite `turns` 表。
- 会话结束时，在后台调度 extractor。
- extractor 负责写入 `profile_facts`、`memory_notes`、`care_tasks` 和 `care_task_occurrences`。
- 下一次 realtime session 只把 active/confirmed 记忆注入到 `session.update.session.instructions`。

router/native memory tools 不是当前核心记忆路径。本测试计划里，它们只作为兼容路径，不作为 extractor-only 核心用例的通过前提。

## 常用命令

本地回归：

```bash
uv run pytest tests/memory tests/test_qwen_omni_realtime.py -q
uv run ruff check src/reachy_mini_conversation_app/memory src/reachy_mini_conversation_app/qwen_omni_realtime.py scripts/run_qwen_long_memory_recall_eval.py tests/memory tests/test_qwen_omni_realtime.py
```

离线/本地单体 turn 记忆评估：

```bash
uv run python scripts/run_memory_eval.py \
  --cases tests/memory_scenarios/eldercare_expanded_25.json \
  --mode case \
  --report-dir /tmp/reachy_memory_eval_reports \
  --fail-on-error
```

真实 Qwen 长对话 recall 评估：

```bash
uv run python scripts/run_qwen_long_memory_recall_eval.py \
  --scenario tests/memory_scenarios/eldercare_long_conversation.json \
  --db /tmp/reachy_long_memory_eval_final.sqlite3 \
  --report-dir /tmp/reachy_long_memory_eval_reports \
  --tool-mode router \
  --memory-timeout-s 120 \
  --allow-real-api \
  --fail-on-error
```

