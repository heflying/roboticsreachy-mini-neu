# 变更汇总 - WebSocket 预连接优化

> 日期: 2026-05-04  
> 变更类型: 性能优化  
> 影响范围: TTS WebSocket 连接管理

---

## 一、变更文件清单

| 文件 | 变更类型 | 关键变更 |
|------|---------|---------|
| `cascade/handler.py` | **删除代码** | 移除 ASR 完成时的预连接触发（原 208-226 行） |
| `cascade/speech_output.py` | **新增代码** | LLM first_token 时触发预连接，设置 `_preparing` 和 `_prepare_task` |
| `cascade/tts/qwen_realtime.py` | **新增状态** | `_preparing`, `_prepare_task`, `wait_preconnect_s` |
| `cascade/tts/qwen_realtime.py` | **新增逻辑** | synthesize 等待预连接机制 |
| `cascade/tts/qwen_realtime.py` | **修复清理** | `_close_prepared()` 清理后台任务 |
| `cascade/timing.py` | **新增映射** | 事件别名新增预连接相关事件 |

---

## 二、新增文档清单

| 文档 | 内容 |
|------|------|
| `docs/WebSocket预连接优化设计.md` | 完整设计文档：问题分析、场景矩阵、方案设计、实现细节 |
| `docs/级联架构性能指标设计.md` | 更新 D1 定义，新增预连接等待时间点 |
| `docs/superpowers/plans/2026-05-02-interrupt-aware-cascade.md` | 新增性能优化补充章节 |

---

## 三、代码关键变更

### 3.1 handler.py - 删除预连接触发

**变更前**（已删除）：
```python
# ASR完成后立即预连接 TTS
if hasattr(self.tts, "prepare_stream"):
    prepare_task = asyncio.create_task(self.tts.prepare_stream())
    # ...等待 0.35s 或继续 LLM
```

**变更原因**：预连接过早，LLM 生成期间导致 stale

---

### 3.2 speech_output.py - 延迟触发

**新增代码**：
```python
# 在第一个 text delta 时触发
if not tts_preconnect_triggered and hasattr(self.tts, "prepare_stream"):
    tts_preconnect_triggered = True
    # 关键：先设置标志再启动任务（避免竞态）
    self.tts._preparing = True
    task = asyncio.create_task(self.tts.prepare_stream())
    self.tts._prepare_task = task
```

**关键修复**：`_preparing = True` 必须在 `create_task()` **之前**设置

---

### 3.3 qwen_realtime.py - 新增状态与等待

**新增状态变量**：
```python
self._preparing: bool = False
self._prepare_task: asyncio.Task | None = None
self.wait_preconnect_s = 0.5  # 等待预连接的最大时间
```

**新增等待逻辑**：
```python
async def synthesize(...):
    # 场景 B/C/D: 预连接进行中，等待完成
    if self._preparing and self._prepare_task is not None:
        tracker.mark("tts_wait_preconnect_start")
        try:
            await asyncio.wait_for(asyncio.shield(self._prepare_task), timeout=0.5)
            tracker.mark("tts_wait_preconnect_success")
        except asyncio.TimeoutError:
            tracker.mark("tts_wait_preconnect_timeout")
    
    # 场景 A/I: 重连前清理后台预连接
    await self._close_prepared()
```

---

### 3.4 timing.py - 新增事件别名

**新增映射**：
```python
"tts_ws_preconnect_start": "tts_preconnect_start",
"tts_ws_preconnected": "tts_preconnect_done",
"tts_wait_preconnect_start": "tts_wait_start",
"tts_wait_preconnect_success": "tts_wait_success",
"tts_wait_preconnect_timeout": "tts_wait_timeout",
...
```

---

## 四、配置参数

| 参数 | 默认值 | 用途 |
|------|--------|------|
| `QWEN_TTS_PREPARED_MAX_AGE_S` | 8.0 | stale 阈值（秒） |
| `QWEN_TTS_WAIT_PRECONNECT_S` | 0.5 | 等待预连接超时（秒） |

---

## 五、测试验证要点

### 成功标志

| 日志关键词 | 含义 |
|------------|------|
| `tts_wait_preconnect_success` | 等待成功，D1=0ms |
| `tts_ws_reused (age_s=0.x)` | 正常 reuse |
| `D1 TTS连接建立 = 0.0ms (reuse)` | 报告显示 reuse |

### 失败标志

| 日志关键词 | 含义 |
|------------|------|
| `tts_wait_preconnect_timeout` | 等待超时，需重连 |
| `tts_ws_prepared_stale (age_s>8)` | 预连接过期 |
| `D1 TTS连接建立 > 500ms` | 未成功 reuse |

---

## 六、预期效果

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| D1 (TTS连接) | 2000-3000ms ❌ | 0ms ✅ 或 ~200ms |
| D1 占 TTFB | 40%-52% | 0%-10% |
| TTFB 核心热点 | D1 | C2 (LLM推理) |

---

## 七、后续工作

1. **验证测试**: 运行 Gradio UI 测试多轮对话
2. **性能对比**: 对比优化前后性能报告
3. **边缘场景**: 测试打断场景下预连接状态清理

---

## 八、关联文档

- 详细设计：`docs/WebSocket预连接优化设计.md`
- 性能指标：`docs/级联架构性能指标设计.md`
- 打断机制：`docs/superpowers/plans/2026-05-02-interrupt-aware-cascade.md`