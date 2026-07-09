# Interrupt-Aware Cascade 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

## 目标

实现 cascade pipeline 的全流程打断能力：用户说话时能立即停止当前 TTS/LLM/播放，进入新 turn。

---

## 核心需求细化

### R1: Turn 级别取消信号传播

**需求描述**：
- 每个 turn 有独立的取消信号（TurnCancellationToken）
- 取消信号是 sticky 的——一旦取消，永久标记
- 新 turn 的 token 与旧 turn 完全独立，不会复用或重置旧 token
- 所有异步任务（LLM、TTS、播放）共享同一个 token，能检测取消状态

**测试验收标准**：
- [x] Token 初始状态为 `cancelled=False` ✅ test_interrupt_coordinator.py
- [x] `cancel()` 调用后 `cancelled=True` 且永久保持 ✅ test_interrupt_coordinator.py
- [x] 多个 turn 的 token 对象不同，取消互不影响 ✅ test_interrupt_coordinator.py
- [x] 已取消的 token 被新 turn token 取代后，旧 token 保持 cancelled ✅ test_interrupt_coordinator.py

### R2: Audio Generation 过滤机制

**需求描述**：
- 每个 turn 有唯一的 generation ID（等于 turn_id）
- 音频 chunk 入队时携带 generation 标签
- 打断时更新 `_current_generation`，丢弃所有 `< current_generation` 的 chunk
- END_OF_TURN marker 也携带 generation，只有 exact match 才触发 completion

**测试验收标准**：
- [x] 正常播放：generation == current_gen 的 chunk 正常播放 ✅ test_audio_playback_interrupt.py (importlib修复)
- [x] 打断过滤：interrupt(new_gen) 后，generation < new_gen 的 chunk 丢弃 ✅ test_audio_playback_interrupt.py
- [x] Stale END_OF_TURN：generation < current_gen 的 END_OF_TURN 不触发 completion ✅ test_audio_playback_interrupt.py
- [x] Exact match：generation == current_gen 的 END_OF_TURN 才 set completion event ✅ test_audio_playback_interrupt.py

### R3: Completion Event 正确绑定

**需求描述**：
- 每个 turn 有独立的 completion event，绑定到其 generation
- Stale turn 的 signal_end_of_turn 不影响新 turn 的 completion
- Event 的创建、查找、删除都在锁保护下进行

**测试验收标准**：
- [x] signal_end_of_turn(caller_turn_id) 返回正确的 turn_id 和 event ✅ test_audio_playback_interrupt.py
- [x] caller_turn_id < current_gen 时，event 立即 set，不入队 END_OF_TURN ✅ test_audio_playback_interrupt.py
- [x] interrupt() 清理所有 < new_generation 的 events，解除 waiter 阻塞 ✅ test_audio_playback_interrupt.py
- [x] Concurrent access：多线程同时操作 events 不会 KeyError 或 deadlock ✅ test_audio_playback_interrupt.py

### R4: Wobbler Generation 隔离

**需求描述**：
- Wobbler 数据也需要 generation 标签（与 audio 相同机制）
- Wobbler thread 按 generation 过滤，丢弃 stale 数据

**测试验收标准**：
- [x] put_wobbler(chunk, generation) 正确入队带标签数据 ✅ test_audio_playback_interrupt.py
- [x] Wobbler thread 丢弃 generation < current_gen 的数据 ✅ test_audio_playback_interrupt.py
- [x] 打断后无 stale wobbler 数据执行到机器人 ✅ test_audio_playback_interrupt.py

### R5: LLM Producer Task 生命周期

**需求描述**：
- LLM producer task 必须在 speech/fallback 完成后正确清理
- 异常、取消、超时场景都要 cancel 并 await producer
- Coordinator 正确注销 producer task

**测试验收标准**：
- [x] 正常完成：producer task 被 cancel + await + unregister ✅ test_task_lifecycle.py (3 cases)
- [x] speak_stream 异常：producer task 被 cancel + await ✅ test_task_lifecycle.py (3 cases)
- [x] Timeout：producer task 不泄漏，继续运行 API 调用 ✅ test_task_lifecycle.py (3 cases)

### R6: TTS Consumer Task 生命周期

**需求描述**：
- Streaming TTS 和 single-request TTS 都使用 consumer task
- Consumer task 内部正确声明 `nonlocal` 闭包变量
- Generator cleanup 在 consumer task 内部完成，不依赖外层引用

**测试验收标准**：
- [x] Streaming path：consume_tts_segment 内 nonlocal first_chunk_queued, barge_in_started ✅ test_task_lifecycle.py (3 cases)
- [x] Single-request path：consume_tts 内 nonlocal 正确声明 ✅ test_task_lifecycle.py (3 cases)
- [x] Cleanup：generator aclose 在 consumer task finally 内执行 ✅ test_task_lifecycle.py (3 cases)
- [x] CancelledError：consumer task 正确处理，不泄漏 generator ✅ test_task_lifecycle.py (3 cases)

### R7: Barge-in Monitor 生命周期

**需求描述**：
- 第一个音频 chunk 入队后启动 barge-in monitor
- Turn 结束（正常或取消）后停止 barge-in monitor
- Stale turn 不能启动或停止新 turn 的 barge-in monitor（ownership 验证）

**测试验收标准**：
- [x] First chunk 入队时 `_start_barge_in_monitor()` 被调用 ✅ test_barge_in_detection.py (__module__修复)
- [x] Turn completion 或 cancellation 时 `_stop_barge_in_monitor()` 被调用 ✅ test_barge_in_detection.py
- [x] Barge-in started flag 正确传递到 cleanup finally 块 ✅ test_barge_in_detection.py
- [x] **Stale stop race**: Turn 1 cleanup 在 Turn 2 已启动后执行，Turn 2 的 monitor 不被停止 ✅ test_barge_in_detection.py
- [x] **Stale start race**: Turn 1 的 first chunk 在 Turn 2 开始后入队，不启动 Turn 2 的 monitor ✅ test_barge_in_detection.py

### R8: VAD Barge-in 触发时机

**需求描述**：
- VAD 检测到用户说话时立即触发打断
- 打断信号通过 TurnController.cancel_current_turn() 传播
- 所有注册的 task 都收到取消信号
- **Latency 边界**: VAD callback timestamp 到 last stale-generation sample 到达 audio device
- **In-flight write**: interrupt() 必须 abort playback stream 或证明 in-flight chunk duration < 50ms

**测试验收标准**：
- [x] VAD speech_detected → cancel_current_turn() 被调用 ✅ test_barge_in_detection.py
- [x] LLM、TTS、Playback task 同时收到取消信号 ✅ test_task_lifecycle.py
- [x] **Latency path**: playback.interrupt() 同步执行 queue purge + stream abort ✅ test_nonfunctional.py
- [x] **Latency test**: 即使 LLM/TTS cleanup 阻塞，audio 停止时间 < 50ms ✅ test_nonfunctional.py
- [x] **In-flight write test**: 模拟 blocking sounddevice.write()，验证 abort 能中断正在写入的 chunk ✅ test_audio_playback_interrupt.py

### R9: Coordinator Task Ownership

**需求描述**：
- 所有 coordinator-owned resources（LLM producer、TTS consumer、TTS generator/WebSocket、playback handle）都有 ownership 保护
- register/unregister API 必须接受 token 或 turn_id 参数
- 只有 owner 匹配 current turn 时才允许修改 coordinator state
- Stale task 的 unregister 不能清除新 turn 的注册

**测试验收标准**：
- [x] register_llm_task(task, token) 只有 token.turn_id == current_turn_id 时成功 ✅ test_interrupt_coordinator.py (@pytest.mark.asyncio)
- [x] register_tts_consumer_task(task, token) 同样需要 ownership 验证 ✅ test_interrupt_coordinator.py
- [x] register_tts_generator(gen, token) 同样需要 ownership 验证 ✅ test_interrupt_coordinator.py
- [x] unregister_llm_task(task, token) 只有 token 匹配时清除对应 entry ✅ test_interrupt_coordinator.py
- [x] unregister_tts_consumer_task(task, token) 同样需要 ownership 验证 ✅ test_interrupt_coordinator.py
- [x] unregister_tts_generator(gen, token) 同样需要 ownership 验证 ✅ test_interrupt_coordinator.py
- [x] **Stale unregister**: Turn 1 task 在 Turn 2 已注册后 unregister，Turn 2 task 仍可被 interrupt ✅ test_interrupt_coordinator.py
- [x] **Identity check**: unregister 必须匹配注册时的 task/gen 对象（不能 unregister 其他 task） ✅ test_interrupt_coordinator.py

---

## 边界场景与错误处理

### E1: 并发打断

**场景描述**：用户在短时间内多次打断（快速说话-停止-说话）

**测试验收标准**：
- [x] 连续 interrupt() 调用不会导致 playback thread crash ✅ test_edge_cases.py (4 cases)
- [x] Generation ID 严格递增，无回退 ✅ test_edge_cases.py (4 cases)
- [x] Completion events 正确清理，无 deadlock ✅ test_edge_cases.py (4 cases)

### E2: TTS WebSocket 连接失败

**场景描述**：打断时 WebSocket 关闭失败或新 turn 时连接失败

**测试验收标准**：
- [x] WebSocket 关闭异常被捕获，不影响新 turn ✅ test_edge_cases.py (4 cases)
- [x] 新 turn 时能创建新 WebSocket 连接 ✅ test_edge_cases.py (3 cases)
- [x] 失败有明确错误日志和用户提示 ✅ test_edge_cases.py (4 cases)

### E3: Playback Thread 异常

**场景描述**：Playback thread 因异常退出（如 sounddevice write 失败）

**Failure Contract**：
- 所有 pending completion events 被 set 并标记为 failed
- Playback 被标记为 unhealthy，有 observable error 状态
- wait_for_playback_complete 返回/抛出 concrete failure（不是 normal completion）
- 下一个 turn 或 fail fast（不尝试自动重启）

**测试验收标准**：
- [x] 异常被捕获并记录，不导致整个程序 crash ✅ test_edge_cases.py (4 cases)
- [x] 所有 pending completion events 被 set 并携带 failure 状态 ✅ test_edge_cases.py (4 cases)
- [x] wait_for_playback_complete 等待者收到 failure 而非 normal completion ✅ test_edge_cases.py (4 cases)
- [x] Playback unhealthy 状态可通过 property 检查 ✅ test_edge_cases.py (4 cases)
- [x] 新 turn 调用 put_audio 时，若 unhealthy 则抛出异常或返回错误 ✅ test_edge_cases.py (4 cases)

### E4: 空音频流

**场景描述**：LLM 返回空内容或 TTS 无音频输出

**测试验收标准**：
- [x] 空内容不触发 barge-in monitor ✅ test_audio_playback_interrupt.py
- [x] signal_end_of_turn 正确处理无音频情况 ✅ test_audio_playback_interrupt.py
- [x] Completion event 正确 set ✅ test_audio_playback_interrupt.py

---

## 架构决策

### A1: TurnCancellationToken 设计

**选择**：每个 turn 创建独立 token 对象，不复用
**原因**：避免跨 turn 状态污染，sticky cancelled 标记确保旧 coroutine 安全终止
**排除**：单一 token + reset() 方案——reset 可能遗漏取消状态传播

### A2: Generation ID = Turn ID

**选择**：generation ID 直接使用 turn_id
**原因**：简化协调，避免双层 ID 管理
**排除**：独立 generation counter——增加同步复杂度

### A3: WebSocket TTS 取消策略

**选择**：打断时关闭 WebSocket，不复用
**原因**：Qwen realtime WebSocket 不支持 mid-stream cancel，关闭是唯一可靠方式
**排除**：发送 cancel command——Qwen 协议不支持

### A4: 锁保护策略

**选择**：单一 `_generation_lock` 保护 `_current_generation` 和 `_playback_complete_events`
**原因**：两个变量紧密关联，同一锁避免嵌套死锁
**排除**：分开的锁——可能导致 ABBA deadlock

### A5: Completion Event Ownership

**选择**：signal_end_of_turn 接受 caller_turn_id 参数，stale turn 不入队 END_OF_TURN
**原因**：防止 old turn completion 影响新 turn
**排除**：signal_end_of_turn 只读 current_generation——stale turn 会错误标记新 turn 完成

---

## 文件结构

| 文件 | 负责 | 状态 |
|------|------|------|
| `cascade/interrupt_coordinator.py` | TurnCancellationToken + InterruptCoordinator | 新增 |
| `cascade/turn_controller.py` | Turn 级别生命周期管理 | 新增 |
| `cascade/ui/audio_playback.py` | interrupt(turn_id) + generation 过滤 | 修改 |
| `cascade/tts/qwen_realtime.py` | synthesize(text, turn_id, token) + cancel_current() | 修改 |
| `cascade/streaming_text.py` | SentenceChunker 支持 token 检查 | 修改 |
| `cascade/handler.py` | 集成 TurnController | 修改 |
| `cascade/speech_output.py` | speak_stream 支持 token + turn_id | 修改 |
| `cascade/ui/audio_recording.py` | VAD barge-in 触发 | 修改 |
| `cascade/ui/gradio_app.py` | barge-in monitor 生命周期 | 修改 |

---

## 任务分解

### Task 1: TurnCancellationToken 基础实现
- 创建 `interrupt_coordinator.py`
- 实现 TurnCancellationToken（sticky cancelled, fixed turn_id）
- 测试：初始状态、sticky、跨 turn 独立性

### Task 2: InterruptCoordinator 实现
- 管理 LLM/TTS/Playback task 注册
- 提供 cancel_all_for_turn(token) 方法
- 测试：task 注册、取消传播、cleanup

### Task 3: AudioPlaybackSystem interrupt(turn_id)
- interrupt(new_generation) 更新 `_current_generation`
- put_audio(chunk, generation) 入队带标签
- playback thread 按 generation 过滤
- completion event 绑定 generation

### Task 4: SentenceChunker token 检查
- flush() 前检查 token.cancelled
- 丢弃 cancelled turn 的 pending text

### Task 5: QwenRealtimeTTS cancel_current() ✅ DONE
- Session tracking: `_session_id`, `_stale_session_ids`, `_current_ws`
- `cancel_current()` async method marks session stale, closes WebSocket
- `cancel_current_from_thread(loop)` cross-thread safe call
- `_is_session_stale(session_id)` check
- `_cleanup_stale_sessions(keep_recent)` cleanup
- 测试：`tests/cascade/test_qwen_tts_cancel.py` (11 tests passed)

### Task 6: TurnController 实现
- 管理 turn_id 递增
- 创建 token 并注册到 coordinator
- 提供 advance_for_new_turn() 和 cancel_current_turn()

### Task 7: 集成到 CascadeHandler
- 在 process_streaming_dialog_response 入口获取 token/turn_id
- 传递到所有 cascade 组件

### Task 8: GradioSpeechOutput token + turn_id
- speak_stream(**kwargs) 接受 token, turn_id, handler
- _speak_single_request 同样支持
- barge-in monitor 生命周期

### Task 9: VAD Barge-in 触发 ✅ DONE
- _start_barge_in_monitor / _stop_barge_in_monitor (gradio_app.py)
- ContinuousVADRecorder: set_barge_in_callback(), enable_barge_in_detection(), _on_speech_start_detected() with debounce
- VAD 检测到用户说话时调用 handler.handle_barge_in()
- Tests: tests/cascade/test_barge_in_detection.py

### Task 10: 集成测试 ✅ DONE
- 完整打断流程测试：test_task_lifecycle.py (25 tests)
- 边界场景测试：test_edge_cases.py (50 tests)
- 非功能需求测试：test_nonfunctional.py (31 tests)

### Task 11: 文档更新 ✅ DONE
- 需求追踪矩阵已更新：2026-05-02-interrupt-aware-cascade-requirements-tracking.md
- Plan文档checkbox已全部勾选

---

## 非功能需求

### NF1: 可观测性
- [x] 所有打断操作有 INFO 级别日志 ✅ test_nonfunctional.py (4 cases)
- [x] Token 创建、取消、generation 更新有日志 ✅ test_nonfunctional.py (4 cases)
- [x] 错误场景有 WARNING 级别日志 ✅ test_nonfunctional.py (4 cases)

### NF2: 性能
- [x] 打断响应时间 < 50ms（从 VAD 检测到 audio 停止） ✅ test_nonfunctional.py (4 cases)
- [x] 无明显的音频播放延迟增加 ✅ test_nonfunctional.py (4 cases)

### NF3: 稳定性
- [x] 并发打断不会导致 playback thread crash ✅ test_nonfunctional.py (4 cases)
- [x] Completion event deadlock 或 timeout 有 recovery path ✅ test_nonfunctional.py (4 cases)

---

## 参考实现

详见 `docs/superpowers/plans/2026-05-02-interrupt-aware-cascade-reference-impl.md`

该文档包含具体代码片段，仅供实时编码参考，不作为设计约束。

---

## 性能优化补充

### WebSocket 预连接优化 (2026-05-04)

打断机制实现完成后，发现 TTS WebSocket 预连接（D1）成为 TTFB 主要热点。

**问题描述**：
- 预连接在 ASR 完成时触发，导致 LLM 生成期间预连接过期（stale）
- D1 占 TTFB 40%-52%，严重影响响应速度

**解决方案**：
- 延迟预连接触发时机：从 ASR 完成延迟到 LLM first_token
- 新增等待机制：LLM 短生成时可等待预连接完成
- Stale 清理：synthesize 重连前清理后台预连接任务

**详细设计**：
详见 `docs/WebSocket预连接优化设计.md`

**效果**：
- D1 从 ~2000ms 降至 0ms（reuse）或 ~200ms（重连）
- TTFB 核心热点回归 C2（LLM推理）