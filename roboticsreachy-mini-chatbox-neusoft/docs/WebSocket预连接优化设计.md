# WebSocket 预连接优化设计

> 版本: v1.0  
> 日期: 2026-05-04  
> 状态: 已实现

---

## 一、问题背景

### 现象

测试数据显示 TTS WebSocket 预连接（D1）成为 TTFB 的主要热点：

| 轮次 | D1 (TTS连接) | TTFB | 占比 |
|------|-------------|------|------|
| #1 | 1834ms ❌ | 4535ms | 40% |
| #2 | 2872ms ❌ | 5701ms | 50% |
| #3 | 3372ms ❌ | 6459ms | 52% |

### 根因

预连接时机过早：在 ASR 完成时触发，导致：

1. **LLM 生成期间预连接过期**：LLM 耗时 1-2 秒，预连接 idle 超过 8 秒阈值
2. **竞态条件**：LLM 短生成时，预连接还没完成就被丢弃（浪费）
3. **Stale reuse**：第二轮 reuse 第一轮的旧预连接，age > 50s

---

## 二、场景分析

### 时间变量

| 变量 | 定义 | 典型值 |
|------|------|--------|
| `T_llm` | LLM 生成时间 (first_token → complete) | 500ms ~ 3000ms |
| `T_preconnect` | 预连接时间 (start → connected) | 100ms ~ 2000ms |
| `T_wait` | 预连接完成到使用的时间差 | 0ms ~ 10000ms+ |
| `T_stale_threshold` | stale 阈值 (prepared_max_age_s) | 8000ms |

### 场景矩阵

```
                    ┌─────────────────────────────────────────────────────────┐
                    │              预连接状态 (at tts_start)                   │
                    ├──────────────┬──────────────┬──────────────┬────────────┤
                    │   未启动     │   进行中     │   已完成     │   已过期    │
LLM 生成时长        │              │              │              │            │
├───────────────────┼──────────────┼──────────────┼──────────────┼────────────┤
│ 短 (< T_preconnect│   场景 A     │   场景 B     │   不可能     │   不可能   │
│ 如 500ms)         │   重连       │   ⚠️ 竞态    │              │            │
├───────────────────┼──────────────┼──────────────┼──────────────┼────────────┤
│ 中 (~ T_preconnect│   场景 C     │   场景 D     │   场景 E     │   不可能   │
│ 如 1000ms)        │   重连       │   边界       │   ✅ reuse   │            │
├───────────────────┼──────────────┼──────────────┼──────────────┼────────────┤
│ 长 (> T_preconnect│   场景 F     │   场景 G     │   场景 H     │   场景 I   │
│ 如 2000ms+)       │   重连       │   ✅ reuse   │   ✅ reuse   │   ⚠️ stale │
└───────────────────┴──────────────┴──────────────┴──────────────┴────────────┘
```

---

## 三、设计方案

### 方案选择

| 方案 | 核心思路 | D1 优化预期 | 复杂度 | 选择 |
|------|---------|------------|--------|------|
| **A: 延迟预连接** | 在 LLM first_token 时预连接 | ~200ms | 低 | ✅ 采用 |
| B: 动态刷新 | LLM 完成前检查并刷新 | ~100ms | 中 | 备选 |
| C: 心跳保活 | 定期发送 ping 保持活跃 | 0ms | 中 | 备选 |
| D: 双连接池 | 维护 2 个备用连接 | 0ms | 高 | 不采用 |

### 方案 A 详细设计

**核心改动**：将预连接触发时机从 ASR 完成延迟到 LLM first_token

```
【旧流程】
ASR完成 → TTS预连接(t=0) → LLM生成(1-2s) → LLM完成 → TTS使用(t=10s)
         ↓                                        ↓
         预连接 age=10s > 8s → stale → 重连 2-3s

【新流程】
ASR完成 → LLM启动 → LLM first_token → TTS预连接 → LLM后续(~0.5s) → TTS使用(t=0.5s)
                                    ↓                              ↓
                                    预连接 age=0.5s < 8s → reuse ✓
```

---

## 四、实现细节

### 4.1 新增状态变量

**文件**: `cascade/tts/qwen_realtime.py`

```python
# 新增状态追踪
self._preparing: bool = False           # 预连接是否正在进行
self._prepare_task: asyncio.Task | None = None  # 预连接任务引用

# 新增配置参数
self.wait_preconnect_s = float(os.getenv("QWEN_TTS_WAIT_PRECONNECT_S", "0.5"))  # 等待预连接的最大时间
```

### 4.2 预连接触发点变更

**文件**: `cascade/speech_output.py`

**变更**: 从 handler.py (ASR完成后) 移动到 speech_output.py (LLM first_token)

```python
# 触发时机：第一个 text delta 收到时
if not tts_preconnect_triggered and hasattr(self.tts, "prepare_stream"):
    tts_preconnect_triggered = True
    # 关键：在 task 启动前设置 _preparing，避免竞态
    self.tts._preparing = True
    task = asyncio.create_task(self.tts.prepare_stream())
    self.tts._prepare_task = task
```

### 4.3 等待机制

**文件**: `cascade/tts/qwen_realtime.py`

**场景 B/C/D 处理**: synthesize 检测到 `_preparing=True` 时等待预连接完成

```python
async def synthesize(self, text: str, voice: Optional[str] = None):
    # 场景 B/C/D: 预连接正在进行中，等待它完成
    if self._preparing and self._prepare_task is not None:
        tracker.mark("tts_wait_preconnect_start")
        try:
            await asyncio.wait_for(asyncio.shield(self._prepare_task), timeout=self.wait_preconnect_s)
            tracker.mark("tts_wait_preconnect_success")
        except asyncio.TimeoutError:
            tracker.mark("tts_wait_preconnect_timeout")
            # 继续走重连流程
    
    # 场景 E/G/H: 预连接已完成，检查是否可 reuse
    if self._prepared_ws is not None:
        ...
```

### 4.4 Stale 清理

**文件**: `cascade/tts/qwen_realtime.py`

**问题**: 后台预连接任务残留导致 stale reuse

**修复**: synthesize 被迫重连前清理后台预连接

```python
# 场景 A/I: 无法 reuse，启动新连接
# 先清理残留的预连接状态（包括后台正在进行的任务）
await self._close_prepared()

async for chunk in self._synthesize_fresh(text, voice_to_use):
    yield chunk
```

---

## 五、关键修复：竞态条件

### 问题

`_preparing` 标志在 `prepare_stream` 任务**内部**才设置，但 `synthesize` 检查时任务刚启动，`_preparing` 还是 `False`

```python
# ❌ 错误顺序
task = asyncio.create_task(self.tts.prepare_stream())
# 此时 _preparing = False（任务刚启动，还没进入函数体）

# synthesize 检查
if self._preparing:  # ❌ False，不等待！
    ...
```

### 修复

在 `create_task()` **之前**设置 `_preparing = True`

```python
# ✅ 正确顺序
self.tts._preparing = True  # 先设置标志
task = asyncio.create_task(self.tts.prepare_stream())  # 再启动任务
```

---

## 六、场景覆盖验证

| 场景 | 条件 | 改进后行为 | D1 预期 |
|------|------|----------|--------|
| **A** | 未启动 + 短LLM | 等待 500ms → reuse 或重连 | 0ms 或 ~200ms |
| **B** | 进行中 + 短LLM | 等待预连接完成 → reuse ✅ | 0ms |
| **C** | 未启动 + 中LLM | 等待预连接 → reuse ✅ | 0ms |
| **D** | 进行中 + 中LLM | 等待预连接 → reuse ✅ | 0ms |
| **E** | 已完成 + 中LLM | 直接 reuse ✅ | 0ms |
| **G** | 进行中 + 长LLM | 预连接自然完成 → reuse ✅ | 0ms |
| **H** | 已完成 + 长LLM | 直接 reuse ✅ | 0ms |
| **I** | 已过期 + 长LLM | stale 检测 → 重连 | ~200ms |
| **J** | reuse 旧连接 | 清理 + 新预连接 | ~200ms |

---

## 七、新增时间点追踪

| 时间点 | 说明 |
|--------|------|
| `tts_wait_preconnect_start` | 开始等待预连接 |
| `tts_wait_preconnect_success` | 等待成功 |
| `tts_wait_preconnect_timeout` | 等待超时 |
| `tts_wait_preconnect_failed` | 等待失败 |

---

## 八、配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `QWEN_TTS_PREPARED_MAX_AGE_S` | 8.0 | stale 阈值 |
| `QWEN_TTS_WAIT_PRECONNECT_S` | 0.5 | synthesize 等待预连接的最大时间 |

---

## 九、变更文件清单

| 文件 | 变更类型 | 变更内容 |
|------|---------|---------|
| `cascade/handler.py` | 删除 | 移除 ASR 完成时的预连接触发代码 |
| `cascade/speech_output.py` | 新增 | LLM first_token 时触发预连接，设置 `_preparing` 和 `_prepare_task` |
| `cascade/tts/qwen_realtime.py` | 新增 | `_preparing`, `_prepare_task`, `wait_preconnect_s` 状态变量 |
| `cascade/tts/qwen_realtime.py` | 新增 | `synthesize` 等待预连接逻辑 |
| `cascade/tts/qwen_realtime.py` | 新增 | `_close_prepared()` 清理后台预连接任务 |
| `cascade/tts/qwen_realtime.py` | 新增 | `tts_wait_preconnect_*` 时间点追踪 |

---

## 十、测试验证要点

1. **日志验证**: 是否出现 `tts_wait_preconnect_*` 时间点
2. **D1 下降**: D1 从 ~2000ms 降至 0ms 或 ~200ms
3. **Stale 消除**: 不再出现 `tts_ws_prepared_stale (age_s>8)`
4. **多轮稳定**: 连续对话中每轮正确 reuse 或清理

---

## 十一、后续优化方向

1. **心跳保活**: 如果 LLM 生成经常超过 8s，可考虑定期 ping 保持活跃
2. **双连接池**: 极端场景下可维护备用连接，但复杂度高
3. **ASR 预连接**: 类似逻辑可应用于 ASR WebSocket

---

## 十二、变更记录

| 版本 | 日期 | 变更内容 |
|------|------|----------|
| v1.0 | 2026-05-04 | 延迟预连接 + 等待机制 + Stale 清理 |