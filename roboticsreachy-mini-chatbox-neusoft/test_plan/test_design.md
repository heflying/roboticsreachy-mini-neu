# 养老场景记忆测试设计

## 测试观点

记忆质量不能只看 extractor 是否“抽到了”。当前测试观点是：必须把记忆作为一条完整行为链路来评估。

完整链路如下：

1. 用户和 assistant 的最终 transcript 被持久化到 `turns` 表。
2. session end extractor 读取本轮 transcript 和已有 active memory context。
3. extractor 输出 CRUD 风格的记忆动作和候选记忆。
4. runtime 执行安全策略、字段归一化、去重和任务生命周期规则。
5. SQLite 以正确表结构和正确状态存储记忆。
6. 下一次 realtime session 只注入 active/confirmed 记忆。
7. Qwen realtime 能正确 recall，并且不会泄漏 pending 或 archived 信息。

只通过存储断言但 recall 失败，不算完整通过。realtime 能答对但泄漏健康、用药、地址、金融等 pending 敏感信息，也算失败。

## 被测记忆类型

| 类型 | 数据表 | 预期用途 | 注入规则 |
|---|---|---|---|
| 长期用户画像 | `profile_facts` | 稳定称呼、沟通偏好、家庭成员、作息、普通偏好 | 只注入 `status=active` |
| 中期会话记忆 | `memory_notes` 和 `sessions.summary` | 近期上下文、回访提示、非敏感短期连续性 | 只注入 active note，并在注入前做敏感内容过滤 |
| 照护任务 | `care_tasks` | 有效提醒、复诊、喝水、运动、已确认用药任务 | 只注入 active task |
| 照护任务实例 | `care_task_occurrences` | 循环任务的某次完成/跳过记录 | 不作为 active task 注入 |
| 待确认敏感信息 | `profile_facts` / `memory_notes` 的 `pending_confirmation` | 健康、用药、地址、安全、联系人、金融线索 | 不作为已确认事实注入 |

## 核心通过标准

- 普通稳定事实能保存为 active profile fact。
- 未确认的健康、用药、地址、联系人、安全、金融、法律信息必须 pending，不能 active。
- pending 和 archived 的值不能出现在 realtime memory context 里。
- 用户修正后，新值替换旧 active 值。
- 忘记、删除、取消命令会让旧事实或旧提醒停止注入。
- 循环任务完成后写入 occurrence，同时循环任务本体保持 active。
- realtime recall 能自然使用 active memory，并在用户直接问“你记得什么”时说全相关 active 信息。
- 正常 app 流程中，session-end extraction 不能阻塞 realtime 关闭。

## 重点捕捉的失败模式

- 修正后旧值仍然 active，例如红茶更新后绿茶仍然注入。
- 敏感内容通过 summary 或 note 泄漏，例如地址、血压、阿司匹林。
- 抽象安全请求被误存为 active reminder，而不是 pending safety fact/note。
- 循环任务完成后被写成 `care_tasks.status=completed`，导致未来提醒失效。
- 家庭成员分散存储后，realtime 只 recall 一个成员。
- 长对话里某一轮包含取消/删除/完成命令，导致其他无关记忆被整体抑制。
- extractor 为同一个 key/value/status 生成重复 pending fact。

## 性能观点

realtime 交互性能和后台 extractor 性能需要分开评估。

realtime 预算：

- 用户说完到首包音频，通常应接近或低于 1 秒。
- `session.update` ack 不应随着 memory context 变大而明显失稳。
- memory context 默认应保持在 3000 字符预算内。
- 报告必须拆分 connect、session.update ack、ASR transcript、speech stop 到首包音频、content done 到首包音频、assistant transcript done、session end/background extraction 等阶段。

extractor 预算：

- 长会话 extraction 可能较慢，因为它运行在后台。
- 慢 extraction 只有在不阻塞 realtime 关闭、不影响下一次用户可见响应时才可接受。
- 对明显更长的真实会话，应增加滚动摘要或分块抽取，而不是无限扩大上下文。

报告中必须保留这些字段，方便代码变更前后对比：

| 指标 | 含义 | 关注点 |
|---|---|---|
| `connect_ms` | WebSocket 建连耗时 | 网络/服务端可用性 |
| `session_update_ack_ms` | `session.update` 到 ack 的耗时 | prompt/context/tools 变更是否影响 session 配置 |
| `audio_to_user_transcript_ms` | 音频发送到用户转写完成 | ASR 和 VAD 影响 |
| `speech_stopped_to_first_audio_ms` | 服务端确认用户停顿到首包音频 | 用户说完后的真实等待感 |
| `content_done_to_first_audio_ms` | 用户文本完成到首包音频 | 模型响应启动速度 |
| `audio_to_assistant_transcript_ms` | 音频发送到 assistant transcript done | 完整回复耗时 |
| `end_session_ms` | session 结束和 extractor/cleanup 耗时 | 是否阻塞、后台任务是否异常 |
| `context_build_ms` | 本地构建 memory context 耗时 | SQLite 查询和渲染开销 |
| `context_chars` | 注入 memory context 字符数 | 上下文膨胀风险 |

## 测试分层

| 层级 | 是否使用真实 API | 目的 |
|---|---|---|
| 单元测试 | 否 | 验证 store schema、状态策略、字段归一化、CRUD 应用 |
| 离线 case 测试 | 默认否 | 批量验证单体 turn 记忆行为，使用 mock extractor 或确定性路由 |
| Qwen extractor 测试 | 是，文本 API | 验证 extractor 的真实抽取质量和 memory_actions 输出 |
| Qwen realtime recall 测试 | 是，realtime API + 本地 TTS | 验证记忆注入、realtime recall、时延和敏感信息隔离 |
| 长对话测试 | 是，extractor + realtime recall | 在混合真实上下文下压测 schema 和记忆编排 |

## 推荐准入流程

修改记忆代码前：

- 跑单元测试和 ruff。
- 跑单体 turn 本地套件。

接入真实 Qwen realtime 前：

- 跑一次真实 extractor 长对话场景。
- 手动检查 SQLite 行数和 injected memory context。
- 跑代表性的 realtime recall prompt。

准备接受版本前：

- 跑全量 25 条两轮 recall case，或至少重跑上次失败 case 加本次改动影响范围。
- 至少跑一条长对话 recall 测试。
