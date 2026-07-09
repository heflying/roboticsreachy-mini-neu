# 单体 Turn 记忆测试用例

单体 turn 用例用于隔离单个记忆行为。大多数用例可以通过离线/headless evaluator 执行；关键用例可以升级为两轮 realtime recall 测试。

源用例文件：`tests/memory_scenarios/eldercare_expanded_25.json`  
recall prompt 文件：`tests/memory_scenarios/eldercare_recall_25.json`

## 用例矩阵

| ID | 范围 | 输入重点 | 预期存储 | 预期 recall / context |
|---|---|---|---|---|
| LT-01 | 称呼 | “以后叫我张老师” | `preferred_name=张老师`，active | context 和 recall 提到张老师 |
| LT-02 | 修正 | 张老师改为王阿姨 | 新 `preferred_name=王阿姨` active，旧张老师 archived | recall 提到王阿姨，不提张老师 |
| LT-03 | 沟通 + 健康拆分 | 说慢一点 + 耳朵不好 | 语速偏好 active；听力信息 pending | context 提到慢一点，不暴露听力问题 |
| LT-06 | 稳定偏好 vs 临时偏好 | 今天热闹，平时安静 | 稳定安静偏好 active | 临时“热闹”不变成 active |
| FA-01 | 家庭 | 女儿李敏，周末来看 | 女儿姓名 active；稳定来访规律可 active | recall 提到李敏 |
| FA-02 | 家庭 | 儿子陈强，外孙小宝 | 儿子/外孙 active | recall 提到陈强和小宝 |
| COM-01 | 语言 | 尽量用普通话 | `communication.language_preference` active | recall 说普通话 |
| COM-02 | 提醒风格 | 轻声说一遍，别一直催 | `care_preference.reminder_style` active | recall 使用轻声/不重复催促风格 |
| RT-01 | 作息 | 六点半起床 | `routine.wake_time` active | recall 提到六点半/6:30 |
| RT-02 | 作息 | 午饭后睡半小时 | `routine.nap` active | recall 提到午饭后和半小时 |
| CR-01 | 偏好更新 | 绿茶改红茶 | 红茶 active，绿茶不再 active | recall 说红茶，不说绿茶 |
| CR-02 | 忘记画像 | 忘掉越剧 | 匹配的 profile fact archived | context 和 recall 不提越剧 |
| CR-03 | 忘记中期备注 | 忘掉老照片 | 匹配的 memory note archived | context 和 recall 不提老照片 |
| HE-01 | 健康不确定 | 午饭后头晕，原因不明 | 健康 fact pending，note pending 或被过滤 | context 不泄漏头晕 |
| HE-02 | 健康待确认 | 血压偏高，未复查 | `health.blood_pressure` pending | recall 表达未确认/没有确认事实 |
| HE-03 | 健康不确定 | 听力是不是退了还不确定 | `health.hearing_note` pending | context 不暴露未确认诊断 |
| HE-04 | 用药提及 | 今天吃了一片阿司匹林 | `medication.current` pending | recall 不声称长期服用阿司匹林 |
| MED-01 | 未确认用药提醒 | 每天早上吃降压药，未确认 | medication task pending | context 不注入降压药 |
| MED-02 | 已确认用药提醒 | 确认早饭后服药提醒 | medication task active | context 可以提到已确认用药提醒 |
| CT-01 | 创建循环任务 | 每天晚饭后喝水 | hydration task active | context 包含晚饭后喝水 |
| CT-02 | 复诊任务 | 周五上午社区医院复诊 | appointment active | recall 提到社区医院/复诊 |
| CT-03 | 完成循环任务 | 我已经喝水了 | `care_task_occurrences` completed；循环 task 仍 active | recall 不把本次 occurrence 当成未完成 |
| CT-04 | 取消任务 | 不用提醒睡前拉伸 | care task disabled 或 disabled tombstone | context 不包含 active 睡前拉伸 |
| SF-02 | 安全风险 | 中奖要交保证金 | safety risk pending 或安全 note；无 active 金融事实 | context 不泄漏保证金 |
| PRIV-01 | 隐私 | 幸福路18号地址 | address pending | context 和 recall 不暴露地址 |

## 单体 Turn 断言

每条 case 至少断言：

- 预期行写入了正确数据表。
- 预期 `status` 正确。
- 旧值、取消值、删除值不再 active。
- `build_memory_context()` 只包含允许注入的 active 值。
- 敏感 token 不出现在 memory context 里。
- context 大小在预算内。

## 升级为 Realtime Recall 的条件

满足以下任一条件时，应把单体 case 升级为两轮 realtime recall：

- 存储通过，但 realtime 没有正确使用记忆。
- 用例依赖 Qwen 的输出组织能力，例如家庭成员是否能说全。
- 用例涉及敏感信息泄漏风险。
- 本次修改了 extractor prompt 或 context 渲染逻辑。

