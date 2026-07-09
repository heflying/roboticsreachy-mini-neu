# Qwen 记忆评估报告 - 2026-05-07

## 总结

最新一轮真实 Qwen 长对话评估已经通过。

- 场景文件：`tests/memory_scenarios/eldercare_long_conversation.json`
- 原始报告：`/tmp/reachy_long_memory_eval_reports_0507_final/qwen_long_memory_recall_20260507_114754.md`
- 测试数据库：`/tmp/reachy_long_memory_eval_final.sqlite3`
- 对话轮数：28
- transcript 字符数：641
- extractor session-end 耗时：60559.3 ms
- 注入 memory context 字符数：986
- realtime recall 通过率：6/6

## 性能数据

### Extraction 和 Context

这一部分衡量“会话结束后的后台记忆整理阶段”。它关注 extractor 从本轮 transcript 中抽取记忆、写入 SQLite、再构建下一轮 memory context 的成本。正常产品流程里这部分已经放到后台异步执行，不应该影响用户听到 realtime 回复的速度，但它会影响记忆多快在下一轮 session 生效。

| 指标说明 | 字段 | 数值 |
|---|---|---:|
| 本轮长对话 turn 数 | turn_count | 28 |
| 本轮 transcript 总字符数 | transcript_chars | 641 |
| session end 后 extractor 抽取、写库、结束 session 的耗时 | extractor_end_session_ms | 60559.3 |
| 本地从 SQLite 构建下一轮 memory context 的耗时 | context_build_ms | 0.7 |
| 下一轮会注入 realtime instructions 的 memory context 字符数 | memory_context_chars | 986 |
| 长对话 extraction 阶段总耗时 | total_extraction_phase_ms | 60563.7 |

### Realtime Recall 聚合耗时

这一部分把所有 recall prompt 的 realtime 性能做聚合统计，展示平均值和最大值。它用于判断“整体体验是否稳定”：例如建连是否变慢、`session.update` 是否因为 memory context 变大而变慢、用户说完后模型是否仍能在目标时间内开始说话。

| 指标说明 | 字段 | 平均 ms | 最大 ms |
|---|---|---:|---:|
| WebSocket 建连耗时 | connect_ms | 223.6 | 271.7 |
| `session.update` 发送后收到 ack 的耗时 | session_update_ack_ms | 217.8 | 259.8 |
| 发送用户音频到用户转写完成的耗时 | audio_to_user_transcript_ms | 1522.1 | 2335.3 |
| 服务端确认用户停顿到 assistant 首包音频的耗时 | speech_stopped_to_first_audio_ms | 360.7 | 409.7 |
| 用户文本完成到 assistant 首包音频的耗时 | content_done_to_first_audio_ms | 559.7 | 670.2 |
| 发送用户音频到 assistant 完整 transcript 完成的耗时 | audio_to_assistant_transcript_ms | 2685.3 | 3909.1 |
| recall session 结束和后台清理/抽取调度耗时 | end_session_ms | 3113.7 | 3618.6 |
| 本地构建 memory context 的耗时 | context_build_ms | 1.1 | 1.6 |

### Realtime Recall 单 Prompt 明细

这一部分逐条展示每个 recall prompt 的耗时拆分。它用于定位具体慢在哪个 case：有些 prompt 慢可能是用户转写慢，有些可能是 assistant 回复长，有些可能是 session end 后台处理慢。后续改代码时，如果聚合指标变差，可以先看这张表找到是哪个 prompt 和哪个阶段拉高了耗时。

| 指标说明 | 字段 | identity_communication | family | routine | tea_update | care_tasks | sensitive_pending |
|---|---|---:|---:|---:|---:|---:|---:|
| WebSocket 建连耗时 | connect_ms | 271.7 | 188.9 | 198.3 | 208.9 | 233.2 | 240.8 |
| `session.update` 发送后收到 ack 的耗时 | session_update_ack_ms | 169.8 | 224.2 | 259.8 | 228.4 | 247.6 | 177.2 |
| 发送用户音频到用户转写完成的耗时 | audio_to_user_transcript_ms | 1792.2 | 1428.7 | 1164.8 | 1055.6 | 1355.8 | 2335.3 |
| 服务端确认用户停顿到 assistant 首包音频的耗时 | speech_stopped_to_first_audio_ms | 364.7 | 306.1 | 391.8 | 336.5 | 409.7 | 355.4 |
| 用户文本完成到 assistant 首包音频的耗时 | content_done_to_first_audio_ms | 539.2 | 493.5 | 595.4 | 522.1 | 670.2 | 537.7 |
| 发送用户音频到 assistant 完整 transcript 完成的耗时 | audio_to_assistant_transcript_ms | 2621.9 | 3052.4 | 2358.2 | 1883.0 | 2287.2 | 3909.1 |
| recall session 结束和后台清理/抽取调度耗时 | end_session_ms | 3542.3 | 1059.8 | 3523.4 | 3618.6 | 3504.7 | 3433.4 |
| 本轮注入 realtime instructions 的 memory context 字符数 | context_chars | 986 | 986 | 986 | 986 | 986 | 986 |

### 性能解读

- 用户停顿后到首包音频平均约 360.7 ms，最大约 409.7 ms，满足“说完后约 1 秒内开口”的目标。
- 用户文本完成到首包音频平均约 559.7 ms，最大约 670.2 ms，说明模型启动回复速度稳定。
- `session.update` ack 平均约 217.8 ms，当前 986 字符 memory context 没有造成明显配置延迟。
- 本地 `context_build_ms` 平均约 1.1 ms，SQLite 查询和上下文渲染不是瓶颈。
- extractor 约 60.6 秒，属于后台耗时，不应阻塞用户可见的 realtime 交互；后续如果长对话继续增长，需要重点观察这个指标。

## 本地回归

最近一次本地检查结果：

```text
64 passed
ruff: All checks passed
```

执行命令：

```bash
uv run pytest tests/memory tests/test_qwen_omni_realtime.py -q
uv run ruff check src/reachy_mini_conversation_app/memory src/reachy_mini_conversation_app/qwen_omni_realtime.py scripts/run_qwen_long_memory_recall_eval.py tests/memory tests/test_qwen_omni_realtime.py
```

## 最终存储快照

| 项目 | 数量 |
|---|---:|
| profile_facts | 13 |
| profile_active | 12 |
| profile_pending | 1 |
| profile_archived | 0 |
| care_tasks | 3 |
| care_active | 2 |
| care_pending | 0 |
| care_completed | 0 |
| care_disabled | 1 |
| task_occurrences | 1 |
| memory_notes | 2 |
| memory_notes_active | 0 |
| memory_notes_pending | 2 |
| memory_context_chars | 986 |

## 已存用户画像

| Key | Value | Status |
|---|---|---|
| preferred_name | 林阿姨 | active |
| communication.language_preference | 普通话 | active |
| communication.voice_style | 轻声 | active |
| communication.speaking_pace | 说慢一点 | active |
| family.daughter.name | 李敏 | active |
| family.son.name | 陈强 | active |
| family.grandchild.name | 小宝 | active |
| family.visit_pattern | 女儿李敏每周六下午常来访 | active |
| preference.likes | 红茶 | active |
| routine.wake_time | 06:40 | active |
| routine.nap | 午饭后半小时 | active |
| care_preference.reminder_style | 仅提醒一次，不重复催促 | active |
| safety.scam_risk | 提到疑似诈骗风险，需确认 | pending_confirmation |

## 照护任务状态

| Title | Type | Status | Repeat |
|---|---|---|---|
| 周五上午社区医院复诊 | appointment | active | |
| 晚饭后喝水 | hydration | active | daily |
| 睡前拉伸 | reminder | disabled | |

循环任务完成记录单独存储：

| Task | Occurrence | Status |
|---|---|---|
| 晚饭后喝水 | 2026-05-07 | completed |

## Realtime Recall 结果

| Prompt | Status | Assistant 结果摘要 |
|---|---|---|
| identity_communication | passed | 记得林阿姨，以及轻声慢语 |
| family | passed | 记得李敏、陈强、小宝，以及周六下午来访 |
| routine | passed | 记得 06:40 起床和午饭后休息 |
| tea_update | passed | 记得红茶 |
| care_tasks | passed | 记得社区医院复诊和晚饭后喝水 |
| sensitive_pending | passed | 没有暴露用药/住址等未确认敏感信息 |

## 测试解读

当前 schema 已经能覆盖本轮测试中的主要养老记忆流程：

- 普通稳定事实能作为 active profile fact 持久化。
- 敏感或不确定信息会保持 pending。
- pending note 会保留用于未来确认，但不会注入 realtime。
- 循环任务完成不会破坏 active task。
- 家庭信息在注入前会聚合，提升 realtime recall 完整性。

长对话 extractor 仍然较慢，本轮约 1 分钟。这个耗时之所以可接受，是因为正常产品流程里 session end extraction 已经改成后台调度，不应作为用户可感知同步时延。

## 剩余风险

- realtime 回答偶尔会过于谨慎，例如正确 recall active 普通偏好后又补一句“需要确认吗”。
- 明显更长的真实会话需要滚动摘要或分块抽取。
- caregiver-confirmed 的用药、联系人、紧急信息流程还需要额外补充测试。
