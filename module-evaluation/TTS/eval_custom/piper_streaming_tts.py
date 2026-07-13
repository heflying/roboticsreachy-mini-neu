# -*- coding: utf-8 -*-
"""
piper_streaming_tts.py - 基于 piper 原生库的流式 TTS 模块
"""

# Windows: 修复 espeak-ng 读取 phontab 等文件的编码问题
# 需要在任何 import 之前设置
import os
if os.name == "nt":
    os.environ["PYTHONUTF8"] = "1"
    try:
        import locale
        locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
    except Exception:
        try:
            locale.setlocale(locale.LC_ALL, "")
        except Exception:
            pass

import sys, time, wave, queue, threading
from pathlib import Path
from dataclasses import dataclass, field
from collections.abc import Callable
import numpy as np

if sys.platform == "win32":
    import builtins as _builtins
    _orig_open = _builtins.open
    def _open_utf8(file, mode="r", *args, **kwargs):
        if "encoding" not in kwargs and "b" not in str(mode):
            kwargs["encoding"] = "utf-8"
        return _orig_open(file, mode, *args, **kwargs)
    _builtins.open = _open_utf8

try:
    import tomllib
except ImportError:
    import tomli as tomllib

# 设置 Piper espeak-ng-data 路径（修复 Windows 硬编码路径错误）
# 从 .env 读取 PIPER_ESPEAK_DATA，若不存在则使用默认推导路径
import dotenv
dotenv.load_dotenv()
_espeak_data = os.environ.get("PIPER_ESPEAK_DATA")
if not _espeak_data:
    _venv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv")
    _espeak_data = os.path.join(
        _venv_path, "Lib", "site-packages", "piper", "espeak-ng-data"
    )
if os.path.isdir(_espeak_data):
    os.environ["ESPEAK_DATA_PATH"] = _espeak_data
    # Piper 内部也用这个路径
    os.environ["PIPER_ESPEAK_DATA_PATH"] = _espeak_data

from piper import PiperVoice, SynthesisConfig

SUB_CHUNK_SIZE = 4096

def _preprocess_mixed_text(text):
    char_map = {
        "b": "弼",
        "c": "s伊",
        "d": "棣",
        "e": "邑", 
        "f": "癌fu ",
        "g": "暨", 
        "h": "ah ",
        "i": "埃",
        "k": "剋",
        "l": "癌o ",
        "m": "埃mu ",
        "n": "嗯",
        "o": "讴",
        "p": "砒",
        "q": "k呦",
        "r": "錒r ",
        "s": "埃s ",
        "t": "倜",
        "u": "滺",
        "v": "u一",
        "w": "哒bbw ",
        "x": "挨ks s ",
        "y": "顡",
    }
    result = []
    for ch in text:
        if "A" <= ch <= "Z":
            ch = ch.lower()
        # 再应用字符映射
        ch = char_map.get(ch, ch)
        result.append(ch)
    return "".join(result)

def _trim_leading_silence(audio, sample_rate, silence_threshold=0.01, max_trim_duration=0.8):
    if len(audio) == 0:
        return audio
    max_trim_samples = int(max_trim_duration * sample_rate)
    end_idx = min(len(audio), max_trim_samples)
    for i in range(end_idx):
        if abs(audio[i]) > silence_threshold:
            return audio[i:]
    return audio[max_trim_samples:]

@dataclass
class PiperModelConfig:
    name: str
    model_path: str
    noise_scale: float = 0.667
    length_scale: float = 1.0
    noise_w: float = 0.8
    sample_rate: int = 22050
    prepend_buffer: str = ""
    data_dir: str = ""  # 自定义 espeak-ng-data 目录（如 huayan 模型自带）

class PiperStreamingTTS:

    @dataclass
    class StreamResult:
        audio: np.ndarray
        sample_rate: int
        ttft: float
        synthesis_time: float
        num_chunks: int = 0
        chunk_times: list = field(default_factory=list)
        audio_sample_count: int = 0

    def __init__(self, config):
        self._config = config
        self._prepend_buffer = config.prepend_buffer
        self._buffer_sample_count = None
        self._sample_rate = config.sample_rate

        model_path = Path(config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Piper model not found: {model_path}")

        # 若配置了自定义 data_dir（如 huayan 模型自带 espeak-ng-data），优先使用
        if getattr(config, "data_dir", ""):
            _data_dir = str(Path(config.data_dir).resolve())
            os.environ["ESPEAK_DATA_PATH"] = _data_dir

        download_dir = str(model_path.parent.parent)
        self._voice = PiperVoice.load(str(model_path), download_dir=download_dir)
        self._sample_rate = self._voice.config.sample_rate
        self._config.sample_rate = self._sample_rate

        self._syn_config = SynthesisConfig(
            noise_scale=config.noise_scale,
            length_scale=config.length_scale,
            noise_w_scale=config.noise_w,
        )

    @property
    def sample_rate(self):
        return self._sample_rate

    def warmup(self, test_text="今天天气不错，我们去公园散步吧。"):
        processed = _preprocess_mixed_text(test_text)
        for _ in self._voice.synthesize(processed, syn_config=self._syn_config):
            pass
        if self._prepend_buffer:
            self._get_buffer_sample_count()

    def _get_buffer_sample_count(self):
        if self._buffer_sample_count is not None:
            return self._buffer_sample_count
        if not self._prepend_buffer:
            self._buffer_sample_count = 0
            return 0
        all_audio = []
        for chunk in self._voice.synthesize(
            _preprocess_mixed_text(self._prepend_buffer),
            syn_config=self._syn_config,
        ):
            all_audio.append(chunk.audio_float_array)
        self._buffer_sample_count = sum(len(a) for a in all_audio)
        return self._buffer_sample_count

    def _save_wav(self, audio, sample_rate, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        audio_int16 = (audio * 32767).astype(np.int16)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sample_rate))
            wf.writeframes(audio_int16.tobytes())

    def generate_stream(self, text, on_chunk=None):
        synth_text = text
        if self._prepend_buffer:
            synth_text = self._prepend_buffer + synth_text

        processed = _preprocess_mixed_text(synth_text)

        q = queue.Queue()
        ttft = [0.0]
        error_holder = [None]
        t_start = time.perf_counter()

        def _producer():
            is_first_chunk = True
            try:
                for chunk in self._voice.synthesize(
                    processed, syn_config=self._syn_config,
                ):
                    audio_float = chunk.audio_float_array.astype(np.float32)
                    if is_first_chunk:
                        t_now = time.perf_counter()
                        ttft[0] = t_now - t_start
                        audio_float = _trim_leading_silence(
                            audio_float, sample_rate=self._sample_rate,
                        )
                        is_first_chunk = False
                    for i in range(0, len(audio_float), SUB_CHUNK_SIZE):
                        sub = audio_float[i: i + SUB_CHUNK_SIZE]
                        if len(sub) == 0:
                            continue
                        q.put(("chunk", sub))
                q.put(("done", None))
            except Exception as e:
                error_holder[0] = e
                q.put(("error", None))

        thread = threading.Thread(target=_producer, daemon=True)
        thread.start()

        all_chunks = []
        while True:
            msg_type, data = q.get()
            if msg_type == "error":
                thread.join(timeout=2)
                raise RuntimeError(f"Piper error: {error_holder[0]}")
            elif msg_type == "done":
                break
            elif msg_type == "chunk":
                all_chunks.append(data)
                if on_chunk is not None:
                    try:
                        on_chunk(data, self._sample_rate)
                    except Exception:
                        pass

        thread.join(timeout=5)
        t_end = time.perf_counter()

        if not all_chunks:
            full_audio = np.array([], dtype=np.float32)
        else:
            full_audio = np.concatenate(all_chunks)

        trim_samples = self._get_buffer_sample_count()
        if trim_samples > 0 and len(full_audio) > trim_samples:
            full_audio = full_audio[trim_samples:]

        synthesis_time = t_end - t_start

        real_ttft = ttft[0]
        if trim_samples > 0 and self._sample_rate:
            buffer_dur = trim_samples / self._sample_rate
            real_ttft = max(0.0, ttft[0] - buffer_dur)

        return self.StreamResult(
            audio=full_audio,
            sample_rate=self._sample_rate,
            ttft=round(real_ttft, 4),
            synthesis_time=round(synthesis_time, 4),
            num_chunks=len(all_chunks),
            audio_sample_count=len(full_audio),
        )

    def generate(self, text, output_path=None, on_chunk=None):
        result = self.generate_stream(text, on_chunk=on_chunk)
        sr = result.sample_rate
        audio = result.audio
        elapsed = result.synthesis_time
        if output_path:
            self._save_wav(audio, sr, output_path)
        return sr, audio, elapsed


def load_models_toml(toml_path=None):
    if toml_path is None:
        toml_path = os.path.join(os.getcwd(), "models.toml")
    toml_path = os.path.abspath(toml_path)
    if not os.path.exists(toml_path):
        raise FileNotFoundError(f"models.toml not found: {toml_path}")
    with open(toml_path, "rb") as f:
        return tomllib.load(f)


def create_piper_tts_from_config(model_name, toml_path=None):
    models = load_models_toml(toml_path)
    if model_name not in models:
        available = ", ".join(models.keys())
        raise ValueError(f"model '{model_name}' not found. available: {available}")
    cfg = models[model_name]
    if cfg.get("type") != "piper":
        raise ValueError(f"model '{model_name}' type is '{cfg.get('type')}', not piper")
    toml_dir = os.path.dirname(os.path.abspath(toml_path or "models.toml"))
    model_path = os.path.normpath(os.path.join(toml_dir, cfg.get("model", "")))
    # data_dir 可能是相对路径，需要基于 toml_dir 解析
    _data_dir = cfg.get("data_dir", "")
    if _data_dir:
        _data_dir = os.path.normpath(os.path.join(toml_dir, _data_dir))
    piper_config = PiperModelConfig(
        name=model_name,
        model_path=model_path,
        noise_scale=float(cfg.get("noise_scale", 0.667)),
        length_scale=float(cfg.get("length_scale", 1.0)),
        noise_w=float(cfg.get("noise_w", 0.8)),
        sample_rate=int(cfg.get("sample_rate", 22050)),
        prepend_buffer=str(cfg.get("prepend_buffer", "")),
        data_dir=_data_dir,
    )
    return PiperStreamingTTS(piper_config)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="piper-xiao_ya")
    parser.add_argument("--text", default="你好，这是一个测试句子。")
    parser.add_argument("--output", default="piper_test_output.wav")
    args = parser.parse_args()
    print(f"Loading: {args.model}")
    tts = create_piper_tts_from_config(args.model)
    print(f"SR: {tts.sample_rate} Hz")
    sr, audio, elapsed = tts.generate(args.text, args.output)
    print(f"Done! {elapsed:.3f}s, {len(audio)/sr:.2f}s audio")
