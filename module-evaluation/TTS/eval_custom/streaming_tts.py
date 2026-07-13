"""
streaming_tts.py - TTS 流式合成模块

支持模型类型: vits, piper, kokoro, melo, coqui, xtts
模型参数维护在 models.toml 中，.env 只配置使用哪个模型及运行时参数
"""

import os
import sys
import time
import wave
import queue
import threading
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field

try:
    import sherpa_onnx
except ImportError:
    print("请先安装 sherpa-onnx: pip install sherpa-onnx")
    sys.exit(1)

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # 兼容 Python < 3.11
    except ImportError:
        print("请安装 tomli: pip install tomli")
        sys.exit(1)


# ============================================
# 模型配置加载
# ============================================

def load_models_toml(toml_path: str = None) -> dict:
    """
    加载 models.toml 文件，返回所有模型配置
    toml_path 相对于工作目录，默认取 eval_custom/models.toml
    """
    if toml_path is None:
        toml_path = os.path.join(os.getcwd(), "eval_custom", "models.toml")

    toml_path = os.path.abspath(toml_path)

    if not os.path.exists(toml_path):
        raise FileNotFoundError(f"models.toml 不存在: {toml_path}")

    with open(toml_path, "rb") as f:
        config = tomllib.load(f)

    return config


def get_model_config(model_name: str, toml_path: str = None) -> dict:
    """
    从 models.toml 中获取指定模型的配置
    如果找不到模型或缺少必需参数，早失败并给出明确错误
    """
    models = load_models_toml(toml_path)

    if model_name not in models:
        available = ", ".join(models.keys())
        raise ValueError(
            f"模型 '{model_name}' 在 models.toml 中未找到。\n"
            f"可用模型: {available}"
        )

    cfg = models[model_name]

    # 检查必需字段
    if "type" not in cfg:
        raise ValueError(f"模型 '{model_name}' 缺少必需字段: type")

    model_type = cfg["type"]

    # 各模型类型必需字段检查
    required_fields = {
        "vits": ["model", "lexicon", "tokens"],
        "piper": ["model", "tokens"],
        "kokoro": ["model", "voices", "tokens", "data_dir"],
        "melo": ["model"],
        "coqui": ["model"],
        "xtts": ["model"],
        "matcha": ["acoustic_model", "vocoder", "lexicon", "tokens"],
    }

    if model_type in required_fields:
        for field in required_fields[model_type]:
            if field not in cfg:
                raise ValueError(
                    f"模型 '{model_name}' (type={model_type}) 缺少必需字段: {field}"
                )

    return cfg


def apply_env_overrides(config: dict, env: dict = None) -> dict:
    """
    用 .env 中的运行时参数覆盖 models.toml 中的配置
    .env 优先级更高
    """
    if env is None:
        env = os.environ

    overrides = {
        "TTS_SPEED": "speed",
        "TTS_SPEAKER_ID": "speaker_id",
    }

    for env_key, cfg_key in overrides.items():
        if env_key in env and env[env_key]:
            try:
                if cfg_key == "speed":
                    config[cfg_key] = float(env[env_key])
                elif cfg_key == "speaker_id":
                    config[cfg_key] = int(env[env_key])
            except ValueError:
                type_name = "数字" if cfg_key == "speed" else "整数"
                raise ValueError(
                    f"{env_key} 值无效: {env[env_key]}，期望{type_name}"
                )

    return config


# ============================================
# StreamingTTS 类
# ============================================

class StreamingTTS:
    """流式 TTS 包装类，支持多种模型后端"""

    def __init__(
        self,
        model_type: str,
        model_path: str = None,
        tokens_path: str = None,
        lexicon_path: str = None,
        rule_fsts: str = None,
        speaker_id: int = 0,
        speed: float = 1.0,
        voices_path: str = None,
        data_dir: str = None,
        acoustic_model_path: str = None,
        vocoder_path: str = None,
        sample_rate: int = None,
        prepend_buffer: str = None,
        **_kwargs,
    ):
        self.model_type = model_type
        self.speaker_id = speaker_id
        self.speed = speed
        self.prepend_buffer = prepend_buffer      # 句首缓冲字（如 "，"），用于吸收模型初始化不稳定帧
        self._buffer_sample_count = None           # 缓存：buffer 字单独合成的音频采样数
        self._cached_sample_rate = sample_rate     # 预置采样率，避免 callback 中误用 44100 fallback
        # _CHARS_PER_SEC_GEN: 模型每秒可生成的字数。
        #   默认值 50.0 为保守估算；调用 warmup() 后自动实测并乘以 1.5 安全系数覆盖。
        self._CHARS_PER_SEC_GEN = 50.0
        self._warmup_done = False                   # 是否已完成 warmup 动态测速

        # 路径处理：统一分隔符，但保持相对路径（sherpa-onnx C++ 层对中文绝对路径可能不兼容）
        def _fixpath(p):
            if p is None:
                return None
            p = os.path.expanduser(p)
            p = p.replace("\\", "/")
            return p

        model_path = _fixpath(model_path)
        tokens_path = _fixpath(tokens_path)
        lexicon_path = _fixpath(lexicon_path)
        acoustic_model_path = _fixpath(acoustic_model_path)
        vocoder_path = _fixpath(vocoder_path)
        # rule_fsts 可能是逗号分隔的多个路径
        if rule_fsts:
            rule_fsts = ",".join(_fixpath(f.strip()) for f in rule_fsts.split(",") if f.strip())
        voices_path = _fixpath(voices_path)
        data_dir = _fixpath(data_dir)

        self.tts = self._init_model(
            model_type=model_type,
            model_path=model_path,
            tokens_path=tokens_path,
            lexicon_path=lexicon_path,
            rule_fsts=rule_fsts,
            speaker_id=speaker_id,
            speed=speed,
            voices_path=voices_path,
            data_dir=data_dir,
            acoustic_model_path=acoustic_model_path,
            vocoder_path=vocoder_path,
            **_kwargs,
        )

    def warmup(self, test_text: str = "今天天气真不错，我们一起去公园散步吧"):
        """
        预热模型并动态计算每字生成耗时 (_CHARS_PER_SEC_GEN)。

        用一段中文测试文本合成一次，实测模型生成速度（字/秒），
        乘以 1.5 安全系数以应对极端环境（系统繁忙、CPU降频等），
        覆盖默认值 50.0。
        """
        t0 = time.perf_counter()
        result = self.tts.generate(text=test_text, sid=self.speaker_id,
                                   speed=self.speed)
        t1 = time.perf_counter()
        elapsed = t1 - t0

        if elapsed > 0 and result is not None and result.samples is not None:
            chars_per_sec = len(test_text) / elapsed
            self._CHARS_PER_SEC_GEN = chars_per_sec * 1.5
            self._warmup_done = True
            print(f"[warmup] 实测 {chars_per_sec:.1f} 字/秒 → "
                  f"_CHARS_PER_SEC_GEN = {self._CHARS_PER_SEC_GEN:.1f} 字/秒 (×1.5)")
        else:
            # 失败时保持默认值
            self._warmup_done = True
            print(f"[warmup] 测速失败，保持默认值 "
                  f"_CHARS_PER_SEC_GEN = {self._CHARS_PER_SEC_GEN:.1f}")

    def _init_model(self, model_type: str, **kwargs):
        """根据 model_type 初始化对应的 TTS 引擎"""
        init_map = {
            "vits": self._init_vits,
            "piper": self._init_piper,
            "kokoro": self._init_kokoro,
            "melo": self._init_melo,
            "coqui": self._init_coqui,
            "xtts": self._init_xtts,
            "matcha": self._init_matcha,
        }
        if model_type not in init_map:
            supported = ", ".join(init_map.keys())
            raise ValueError(
                f"不支持的模型类型: {model_type}。支持的类型: {supported}"
            )
        return init_map[model_type](**kwargs)

    def _init_vits(self, model_path: str, lexicon_path: str,
                   tokens_path: str, rule_fsts: str = None,
                   speaker_id: int = 0, speed: float = 1.0,
                   num_threads: int = 1, **_):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        if not os.path.exists(lexicon_path):
            raise FileNotFoundError(f"词典文件不存在: {lexicon_path}")
        if not os.path.exists(tokens_path):
            raise FileNotFoundError(f"tokens 文件不存在: {tokens_path}")

        rule_fsts_list = rule_fsts.split(",") if rule_fsts else []
        rule_fsts_list = [f.strip() for f in rule_fsts_list if f.strip()]

        model_config = sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=model_path,
                lexicon=lexicon_path,
                tokens=tokens_path,
            ),
            num_threads=num_threads,
        )

        return sherpa_onnx.OfflineTts(
            sherpa_onnx.OfflineTtsConfig(
                model=model_config,
                rule_fsts=",".join(rule_fsts_list) if rule_fsts_list else "",
                max_num_sentences=1,
            )
        )

    def _init_piper(self, model_path: str, tokens_path: str,
                     lexicon_path: str = None, rule_fsts: str = None,
                     data_dir: str = None,
                     speaker_id: int = 0, speed: float = 1.0, **_):
        """Piper 模型在 sherpa-onnx 中通过 vits 接口加载"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        if not os.path.exists(tokens_path):
            raise FileNotFoundError(f"tokens 文件不存在: {tokens_path}")

        # 构建 vits 配置（使用构造函数方式，与官方示例一致）
        vits_kwargs = {"model": model_path, "tokens": tokens_path}
        if lexicon_path and os.path.exists(lexicon_path):
            vits_kwargs["lexicon"] = lexicon_path
        if data_dir and os.path.exists(data_dir):
            vits_kwargs["data_dir"] = data_dir

        model_config = sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(**vits_kwargs),
            num_threads=1,
        )

        # 构建 rule_fsts 参数
        rule_fsts_str = ""
        if rule_fsts:
            rule_fsts_list = []
            for f in rule_fsts.split(","):
                f = f.strip()
                if f and os.path.exists(f):
                    rule_fsts_list.append(f)
            rule_fsts_str = ",".join(rule_fsts_list)

        return sherpa_onnx.OfflineTts(
            sherpa_onnx.OfflineTtsConfig(
                model=model_config,
                rule_fsts=rule_fsts_str,
                max_num_sentences=1,
            )
        )

    def _init_kokoro(self, model_path: str, voices_path: str,
                     tokens_path: str, data_dir: str,
                     lexicon_path: str = None, rule_fsts: str = None,
                     speaker_id: int = 0, speed: float = 1.0, **_):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        if not os.path.exists(voices_path):
            raise FileNotFoundError(f"voices 文件不存在: {voices_path}")
        if not os.path.exists(tokens_path):
            raise FileNotFoundError(f"tokens 文件不存在: {tokens_path}")
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"data_dir 目录不存在: {data_dir}")

        lexicon_str = ""
        if lexicon_path:
            parts = []
            for p in lexicon_path.split(","):
                p = p.strip()
                if p and os.path.exists(p):
                    parts.append(p)
            lexicon_str = ",".join(parts)

        rule_fsts_str = ""
        if rule_fsts:
            parts = []
            for p in rule_fsts.split(","):
                p = p.strip()
                if p and os.path.exists(p):
                    parts.append(p)
            rule_fsts_str = ",".join(parts)

        model_config = sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=model_path,
                voices=voices_path,
                tokens=tokens_path,
                data_dir=data_dir,
                lexicon=lexicon_str,
            ),
            num_threads=1,
        )

        return sherpa_onnx.OfflineTts(
            sherpa_onnx.OfflineTtsConfig(
                model=model_config,
                rule_fsts=rule_fsts_str,
                max_num_sentences=1,
            )
        )

    def _init_melo(self, model_path: str, tokens_path: str = None,
                   lexicon_path: str = None, data_dir: str = None,
                   speaker_id: int = 0, speed: float = 1.0,
                   num_threads: int = 1, **_):
        """Melo 模型在 sherpa-onnx 中通过 vits 接口加载"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        vits_kwargs = {"model": model_path, "tokens": tokens_path or ""}
        if lexicon_path and os.path.exists(lexicon_path):
            vits_kwargs["lexicon"] = lexicon_path
        if data_dir and os.path.exists(data_dir):
            vits_kwargs["data_dir"] = data_dir

        model_config = sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(**vits_kwargs),
            num_threads=num_threads,
        )

        return sherpa_onnx.OfflineTts(
            sherpa_onnx.OfflineTtsConfig(
                model=model_config,
                max_num_sentences=1,
            )
        )

    def _init_coqui(self, model_path: str, **_):
        raise NotImplementedError("Coqui TTS 暂未实现")

    def _init_xtts(self, model_path: str, **_):
        raise NotImplementedError("XTTS 暂未实现")

    def _init_matcha(self, acoustic_model_path: str, vocoder_path: str,
                     lexicon_path: str, tokens_path: str,
                     rule_fsts: str = None,
                     speaker_id: int = 0, speed: float = 1.0,
                     num_threads: int = 1, **_):
        """Matcha-TTS 模型（如 matcha-icefall-zh-baker），需搭配 vocos vocoder"""
        if not os.path.exists(acoustic_model_path):
            raise FileNotFoundError(f"声学模型不存在: {acoustic_model_path}")
        if not os.path.exists(vocoder_path):
            raise FileNotFoundError(f"Vocoder 不存在: {vocoder_path}")
        if not os.path.exists(lexicon_path):
            raise FileNotFoundError(f"词典文件不存在: {lexicon_path}")
        if not os.path.exists(tokens_path):
            raise FileNotFoundError(f"tokens 文件不存在: {tokens_path}")

        rule_fsts_list = rule_fsts.split(",") if rule_fsts else []
        rule_fsts_list = [f.strip() for f in rule_fsts_list if f.strip()]

        model_config = sherpa_onnx.OfflineTtsModelConfig(
            matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                acoustic_model=acoustic_model_path,
                vocoder=vocoder_path,
                lexicon=lexicon_path,
                tokens=tokens_path,
            ),
            num_threads=num_threads,
        )

        return sherpa_onnx.OfflineTts(
            sherpa_onnx.OfflineTtsConfig(
                model=model_config,
                rule_fsts=",".join(rule_fsts_list) if rule_fsts_list else "",
                max_num_sentences=1,
            )
        )

    # 流式生成统计
    @dataclass
    class StreamResult:
        """流式生成结果，包含完整音频和计时数据"""
        audio: np.ndarray
        sample_rate: int
        ttft: float            # 首音延迟 (s) — 从调用到第一个回调
        synthesis_time: float  # 总合成耗时 (s)
        num_chunks: int = 0    # 回调次数
        chunk_times: list = field(default_factory=list)  # 每个chunk的相对时间
        audio_sample_count: int = 0   # 本次生成的总音频采样数（不含 trim）

    def _get_buffer_sample_count(self) -> int:
        """返回 prepend_buffer 单独合成时的音频采样数（懒加载 + 缓存）"""
        if self._buffer_sample_count is not None:
            return self._buffer_sample_count
        if not self.prepend_buffer:
            self._buffer_sample_count = 0
            return 0
        result = self.tts.generate(self.prepend_buffer,
                                   sid=self.speaker_id, speed=self.speed)
        self._buffer_sample_count = len(result.samples)
        return self._buffer_sample_count

    def generate(self, text: str, output_path: str = None,
                 sid: int = None, speed: float = None,
                 on_chunk: callable = None):
        """全量生成语音，返回 (sample_rate, audio, elapsed)"""
        result = self.generate_stream(text, sid=sid, speed=speed,
                                       on_chunk=on_chunk)
        sr = result.sample_rate
        audio = result.audio
        elapsed = result.synthesis_time

        if output_path:
            self._save_wav(audio, sr, output_path)

        return sr, audio, elapsed

    def generate_stream(self, text: str, sid: int = None,
                        speed: float = None,
                        on_chunk: callable = None) -> StreamResult:
        """
        流式生成语音（基于 callback 分块回调），返回 StreamResult。

        参数:
            text:     要合成的中文文本
            sid:      说话人 ID
            speed:    语速
            on_chunk: 外部回调，每收到一个非空音频 chunk 时调用
                      on_chunk(chunk_data: np.ndarray, sample_rate: int)

        内部使用 threading + queue 在 callback 中实时捕获每个音频 chunk，
        从而精确测量 TTFT（首音延迟）和 chunk 时序。

        调用方可以从中获取：
        - result.audio: 完整音频数据 (numpy float32)
        - result.sample_rate: 采样率
        - result.ttft: 首音延迟（从调用到第一个音频 chunk 就绪的耗时）
        - result.synthesis_time: 总合成耗时
        - result.num_chunks: 模型实际输出的音频块数
        - result.chunk_times: 每个 chunk 的相对时间戳
        - result.audio_sample_count: 总采样数
        """
        sid = sid if sid is not None else self.speaker_id
        sp = speed if speed is not None else self.speed

        # 句首缓冲：吸收模型 decoder 初始化不稳定帧
        synth_text = text
        if self.prepend_buffer:
            synth_text = self.prepend_buffer + synth_text

        q: queue.Queue = queue.Queue()
        sr_holder = [None]
        chunk_counter = [0]
        ttft = [0.0]
        chunk_times = []
        error_holder = [None]
        t_start = time.perf_counter()

        def callback(samples: np.ndarray, progress: float) -> int:
            chunk_idx = chunk_counter[0]
            t_now = time.perf_counter()
            chunk_relative = t_now - t_start

            if chunk_idx == 0:
                ttft[0] = chunk_relative

            chunk_times.append(chunk_relative)

            # 拷贝数据避免原始数组被复用
            chunk_data = np.array(samples.copy() if samples.ndim > 0
                                  else [samples], dtype=np.float32)

            if len(chunk_data) == 0:
                chunk_counter[0] += 1
                return 1

            # 获取当前 chunk 的采样率
            chunk_sr = (sr_holder[0] or self._cached_sample_rate or 44100)

            # 外部实时回调（如实时播放）
            if on_chunk is not None:
                try:
                    on_chunk(chunk_data, chunk_sr)
                except Exception:
                    pass

            q.put(('chunk', chunk_data))
            chunk_counter[0] += 1
            return 1  # 继续生成

        def _synthesize():
            try:
                result = self.tts.generate(synth_text, sid=sid, speed=sp,
                                           callback=callback)
                if result is None or result.samples is None:
                    raise RuntimeError(f"语音生成失败，输入文本: {text}")
                sr_holder[0] = result.sample_rate
                q.put(('done', None))
            except Exception as e:
                error_holder[0] = e
                q.put(('error', None))

        thread = threading.Thread(target=_synthesize, daemon=True)
        thread.start()

        # 收集所有 chunk
        all_chunks = []
        while True:
            msg_type, data = q.get()
            if msg_type == 'error':
                thread.join(timeout=2)
                raise RuntimeError(f"流式合成错误: {error_holder[0]}")
            elif msg_type == 'done':
                break
            elif msg_type == 'chunk':
                all_chunks.append(data)
            else:
                raise RuntimeError(f"未知消息类型: {msg_type}")

        thread.join(timeout=2)
        t_end = time.perf_counter()

        # 拼接完整音频
        sr = sr_holder[0]
        if sr:
            self._cached_sample_rate = sr

        if not all_chunks:
            full_audio = np.array([], dtype=np.float32)
        else:
            full_audio = np.concatenate(all_chunks)

        # 裁剪句首缓冲部分
        trim_samples = self._get_buffer_sample_count()
        if trim_samples > 0 and len(full_audio) > trim_samples:
            full_audio = full_audio[trim_samples:]

        synthesis_time = t_end - t_start

        # 计算真实 TTFT（修正缓冲偏移）
        real_ttft = ttft[0]
        if trim_samples > 0 and sr:
            buffer_dur = trim_samples / sr
            real_ttft = max(0.0, ttft[0] - buffer_dur)

        return self.StreamResult(
            audio=full_audio,
            sample_rate=sr,
            ttft=round(real_ttft, 4),
            synthesis_time=round(synthesis_time, 4),
            num_chunks=len(all_chunks),
            chunk_times=[round(t, 4) for t in chunk_times],
            audio_sample_count=len(full_audio),
        )

    def _save_wav(self, audio: np.ndarray, sample_rate: int, path: str):
        """保存音频为 WAV 文件"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        audio_int16 = (audio * 32767).astype(np.int16)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())


# ============================================
# 工厂函数
# ============================================

def create_tts_from_config(model_name: str, toml_path: str = None,
                          env: dict = None) -> StreamingTTS:
    """
    从 models.toml 配置中创建 TTS 实例
    model_name: models.toml 中的模型条目名称
    toml_path:  models.toml 路径（相对于工作目录）
    env:        环境变量字典（默认用 os.environ）
    """
    # 1. 从 TOML 获取模型配置
    cfg = get_model_config(model_name, toml_path)

    # 2. 用 .env 覆盖运行时参数
    cfg = apply_env_overrides(cfg, env)

    # 3. 映射 TOML 字段到 StreamingTTS 构造函数参数
    param_map = {
        "model": "model_path",
        "tokens": "tokens_path",
        "lexicon": "lexicon_path",
        "rule_fsts": "rule_fsts",
        "voices": "voices_path",
        "data_dir": "data_dir",
        "acoustic_model": "acoustic_model_path",
        "vocoder": "vocoder_path",
        "type": "model_type",
        "speaker_id": "speaker_id",
        "speed": "speed",
        "sample_rate": "sample_rate",
        "prepend_buffer": "prepend_buffer",
        "num_threads": "num_threads",
    }

    kwargs = {}
    for toml_key, py_key in param_map.items():
        if toml_key in cfg:
            kwargs[py_key] = cfg[toml_key]

    return StreamingTTS(**kwargs)


def create_tts_from_env(env_path: str = None, toml_path: str = None,
                        env: dict = None) -> StreamingTTS:
    """
    从 .env 文件自动读取配置并创建 TTS 实例
    env_path:  .env 文件路径（相对于工作目录，默认 eval_custom/.env）
    toml_path: models.toml 路径（相对于工作目录，默认 eval_custom/models.toml）
    """
    # 加载 .env
    if env is None:
        env = os.environ

    # 如果指定了 env_path，用 python-dotenv 加载
    if env_path is not None:
        try:
            from dotenv import load_dotenv
            abs_path = env_path if os.path.isabs(env_path) else os.path.join(os.getcwd(), env_path)
            load_dotenv(abs_path)
            # 重新读取
            env = os.environ
        except ImportError:
            print("警告: 未安装 python-dotenv，无法加载 .env 文件")
            print("请运行: pip install python-dotenv")

    # 读取 TTS_MODEL
    model_name = env.get("TTS_MODEL")
    if not model_name:
        raise ValueError(
            "未找到 TTS_MODEL 配置。请在 .env 文件中设置 TTS_MODEL=<模型名>"
        )

    return create_tts_from_config(model_name.strip(), toml_path, env)


if __name__ == "__main__":
    # 简单测试
    import argparse

    parser = argparse.ArgumentParser(description="TTS 测试")
    parser.add_argument("--text", type=str, default="你好，这是一个测试。",
                        help="要合成的文字")
    parser.add_argument("--output", type=str, default="test_output.wav",
                        help="输出 WAV 文件路径")
    parser.add_argument("--model", type=str, default=None,
                        help="模型名称（覆盖 .env 中的 TTS_MODEL）")
    parser.add_argument("--env", type=str, default="eval_custom/.env",
                        help=".env 文件路径")
    parser.add_argument("--toml", type=str, default="eval_custom/models.toml",
                        help="models.toml 文件路径")

    args = parser.parse_args()

    # 加载 .env
    from dotenv import load_dotenv
    env_path = args.env if os.path.isabs(args.env) else os.path.join(os.getcwd(), args.env)
    if os.path.exists(env_path):
        load_dotenv(env_path)

    model_name = args.model or os.environ.get("TTS_MODEL")
    if not model_name:
        print("错误: 请通过 --model 或 .env 中的 TTS_MODEL 指定模型名称")
        sys.exit(1)

    print(f"使用模型: {model_name}")
    tts = create_tts_from_config(model_name, args.toml)

    print(f"生成中: {args.text}")
    sr, audio, elapsed = tts.generate(args.text, args.output)
    print(f"完成! 耗时: {elapsed:.3f}s, 采样率: {sr}, 时长: {len(audio)/sr:.2f}s")
    print(f"已保存到: {args.output}")
