# 长对话记忆测试用例

长对话用例用于验证系统在单个 session 中混合出现多类事实、修正、敏感信息、任务创建、任务完成和任务取消时，记忆结构是否仍然清晰可靠。

源用例文件：`tests/memory_scenarios/eldercare_long_conversation.json`

## 主场景：LC-01 养老混合长对话

对话轮数：28  
最新测试 transcript 字符数：641  
测试目的：压测 extraction、状态策略、schema 结构、context 渲染和 realtime recall。

### 对话维度

| 维度 | 示例内容 | 预期存储 |
|---|---|---|
| 称呼 | “以后叫我林阿姨” | `preferred_name=林阿姨`，active |
| 沟通偏好 | 普通话、慢一点、轻声 | communication facts active |
| 作息 | 06:40 起床，午饭后睡半小时 | routine facts active |
| 家庭 | 女儿李敏、儿子陈强、外孙小宝、周六下午来看 | family facts active，并在 context 中聚合为 `family.overview` |
| 普通偏好修正 | 绿茶改红茶，绿茶忘掉 | 红茶 active；绿茶不 active |
| 照护任务创建 | 晚饭后喝水，周五上午社区医院复诊 | hydration 和 appointment task active |
| 敏感健康 | 血压偏高未复查，胃不舒服 | pending 或不注入 |
| 敏感用药 | 一次阿司匹林，不确定长期 | pending，不注入 |
| 隐私地址 | 幸福路十八号地址 | pending，不注入 |
| 安全风险 | 中奖保证金 | safety pending；无 active financial fact |
| 取消任务 | 睡前拉伸不用提醒 | disabled task / tombstone |
| 完成任务实例 | 今天晚饭后喝水完成 | 一条 completed occurrence；循环 hydration task 仍 active |
| 提醒风格 | 不连续催三遍，只提醒一次 | care preference active |

## Recall Prompts

| ID | Prompt | 预期 |
|---|---|---|
| identity_communication | 你还记得该怎么称呼我，以及和我说话时要注意什么吗？ | 提到林阿姨，以及普通话/慢一点/轻声 |
| family | 你还记得我的家人名字和谁常来看我吗？ | 提到李敏、陈强、小宝，以及周六/周六下午来访 |
| routine | 你还记得我的起床和午休习惯吗？ | 提到 06:40 和午饭后半小时休息 |
| tea_update | 我现在的饮茶口味是什么？ | 提到红茶，不提绿茶 |
| care_tasks | 你现在记得哪些仍然有效的提醒事项？ | 提到有效复诊/喝水任务，不提已取消睡前拉伸 |
| sensitive_pending | 关于我的健康、用药和住址，你有什么已经确认可以直接使用的信息吗？ | 表达没有已确认可直接使用的健康/用药/地址信息，且不泄漏敏感值 |

## 长对话断言

存储断言：

- `profile_facts` 中普通稳定事实 active，敏感事实 pending。
- `memory_notes` 不把敏感或不确定健康信息作为 active context 注入。
- `care_tasks` 包含 active 循环任务/复诊任务，以及 disabled 取消任务。
- `care_task_occurrences` 包含循环任务的完成记录。
- 尽量减少同 key/value/status 的重复 pending fact。

context 断言：

- active profile facts 被紧凑渲染。
- family facts 被聚合为 `family.overview`。
- pending 健康、用药、地址信息不出现。
- disabled/completed 任务状态不会被当作 active reminder 注入。

realtime 断言：

- assistant 能自然使用 active facts。
- 用户直接问 recall 时，assistant 能说全相关 active entries。
- assistant 不泄漏 pending 敏感信息。
- assistant 不复活 archived 或 disabled 信息。

性能断言：

- speech stop 到首包音频通常应接近或低于 1 秒。
- context build 应接近瞬时完成。
- 长会话 extractor latency 会记录，但正常 app 流程中不应阻塞 realtime shutdown。
- 每次长对话报告必须输出每个 recall prompt 的 connect、session ack、用户转写完成、首包音频、assistant transcript done、end session、context chars。
- 任何记忆相关代码变更，都应对比修改前后的 `context_chars`、`context_build_ms`、`session_update_ack_ms`、`content_done_to_first_audio_ms` 和 extractor `end_session_ms`。

## 后续可扩展长场景

- LC-02：多日照护任务生命周期：创建循环任务、完成多次 occurrence、跳过一天、再 recall。
- LC-03：家属确认修正：女儿确认用药、联系人或紧急信息。
- LC-04：强隐私会话：用户提供地址、电话、门禁码，并要求不要复述。
- LC-05：情绪陪伴会话：孤独感、普通偏好、家庭来访和临时情绪混合。
- LC-06：多重矛盾修正：用户连续更新称呼、饮茶偏好、提醒风格，然后删除其中一项。
