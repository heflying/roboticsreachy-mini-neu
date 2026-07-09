# VAD Backend Switch Design: PyTorch ↔ ONNX Runtime

> Date: 2026-05-05
> Status: Approved
> Branch: feature/interrupt-aware-cascade

## Goal

Add ONNX Runtime as an alternative VAD inference backend alongside the existing PyTorch backend, enabling A/B performance comparison through config-driven switching with inference latency logging.

## Architecture

### Class Hierarchy

```
VADBackend (Protocol)          — minimal inference interface
  ├── get_speech_prob(audio, sr) -> float
  └── reset_states() -> None

SileroVADPyTorch(VADBackend)   — existing code migrated, zero behavior change
SileroVADOnnx(VADBackend)      — new, onnxruntime inference

SileroVAD                      — facade, preserves original public API
  ├── delegates to _backend.get_speech_prob()
  ├── retains process_chunk() hysteresis logic (backend-agnostic)
  └── injects perf_counter timing log in get_speech_prob()
```

`SileroVAD` public API unchanged: `__init__`, `get_speech_prob`, `is_speech`, `process_chunk`, `reset`, `is_speaking`. Consumers only need to pass an additional `backend` parameter.

### Configuration

New `vad` section in `cascade.yaml`:

```yaml
vad:
  backend: onnx              # onnx (default) | pytorch
  threshold: 0.5
  min_speech_duration_ms: 250
  min_silence_duration_ms: 500
```

Environment variable override: `CASCADE_VAD_BACKEND=onnx|pytorch` (consistent with existing `CASCADE_ASR_PROVIDER` pattern).

`CascadeConfig` gains: `vad_backend`, `vad_threshold`, `vad_min_speech_duration_ms`, `vad_min_silence_duration_ms`.

Hardware/import checks at config load time:
- `onnx`: verify `onnxruntime` importable
- `pytorch`: verify `torch` importable

### Inference Latency Log

Single injection point in `SileroVAD.get_speech_prob()`:

```python
t0 = perf_counter()
prob = self._backend.get_speech_prob(audio, sample_rate)
elapsed_ms = (perf_counter() - t0) * 1000
logger.debug(f"VAD backend={self._backend_name} inference={elapsed_ms:.3f}ms prob={prob:.3f}")
```

Output at `DEBUG` level:
```
DEBUG VAD backend=pytorch inference=1.234ms prob=0.872
DEBUG VAD backend=onnx inference=0.387ms prob=0.869
```

A/B test procedure: run same speech scenarios with each backend, grep logs for `inference=` values.

### ONNX Model & Dependencies

Model file: `silero_vad.onnx` auto-downloaded to `models/VAD/silero` on first use of ONNX backend (with local cache check).

New optional dependency in `pyproject.toml`:

```toml
cascade_silero_vad_onnx = ["onnxruntime>=1.17.0"]
```

Existing extra `cascade_silero_vad` (`torch`, `torchaudio`) unchanged. Both extras included in `cascade_all`.

### Consumer Changes

`audio_recording.py` and `console.py` read VAD params from config:

```python
config = get_config()
self._vad = SileroVAD(backend=config.vad_backend, threshold=config.vad_threshold, ...)
```

## File Changes

| File | Type | Description |
|------|------|-------------|
| `cascade/vad.py` | Rewrite | Protocol + PyTorch backend + Facade + timing log |
| `cascade/vad_onnx.py` | New | ONNX backend implementation (with model download) |
| `cascade.yaml` | Modify | Add `vad` section with `backend: onnx` |
| `cascade/config.py` | Modify | Parse vad config + `CASCADE_VAD_BACKEND` env var + import checks |
| `cascade/ui/audio_recording.py` | Modify | Read VAD params from config |
| `cascade/console.py` | Modify | Read VAD params from config |
| `pyproject.toml` | Modify | Add `cascade_silero_vad_onnx` extra |

**Not changed**: `handler.py`, `pipeline.py`, tests — VAD public API is stable.

## Constraints

- FECMA-P C-001 (Contract): `SileroVAD` public API unchanged, drop-in replacement.
- FECMA-P F-001 (Focus): `VADBackend` Protocol has single responsibility (inference only).
- Minimal invasiveness: timing log is one `perf_counter` pair in `get_speech_prob`.
- Default backend: `onnx`.
