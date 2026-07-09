"""
test_piper_tts.py - Piper 模型专用测试脚本

硬编码 piper-xiao_ya 模型，对句子列表逐条合成，实时播放 + 保存 wav。
逻辑与 test_streaming_tts.py 一致，TTFT 测量、统计汇总。

用法：
  python test_piper_tts.py
"""

import os
import sys
import json
import shutil
import threading
import queue
import numpy as np
from datetime import datetime
from pathlib import Path

# 切换到脚本所在目录，确保 models.toml 和模型路径正确
os.chdir(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from piper_streaming_tts import create_piper_tts_from_config

# ============================================
# 从 .env 读取配置（覆盖 .env 中的 TTS_MODEL 来切换模型）
# ============================================

import dotenv
dotenv.load_dotenv()

MODEL_NAME = os.environ.get("TTS_MODEL", "piper-xiao_ya")
OUTPUT_DIR = "result_piper"

# 测试句子列表（在此填入）
SENTENCES = [
    "好的，没问题。",
    "张大爷，您今天感觉怎么样呀？",
    "哎呀，这个药可不能空腹吃！",
    "社区医院明天上午九点，在二楼会议室举办糖尿病饮食管理讲座，请您准时参加。",
    "今天天气真不错，下午咱们去小花园坐坐，晒晒太阳聊聊天吧。",
    "王阿姨，您儿子刚来电话了，说这周六要带小孙子回来看您呢，高兴不？",
    "康复训练器材已经全部安装调试完毕可以正常使用了。",
    "最近气温变化大，早晚凉中午热，刘奶奶您出门记得多带件外套，别着凉了。",
    "太极拳很好。",
    "哎，对了，您昨天问的那本《三国演义》，图书馆已经帮您找到了，我待会儿给您送过去。",
    "老年大学的书法课和绘画课，特别受爷爷奶奶们欢迎，名额都快报满了。",
    "下周三下午三点，活动中心有心理健康讲座，主题是'如何保持积极乐观的心态'。",
    "您血压有点高，记得饭后半小时吃降压药，饮食少油少盐，不舒服随时按呼叫铃。",
    "智能手机培训班下周一开始报名，请各位老人家互相转告一声。",
    "嗯，明白了。",
    "下雨了！李奶奶您窗台那盆兰花我帮您搬进来了，别担心啊。",
]


# ============================================
# 实时播放辅助（sounddevice）
# ============================================

def _create_realtime_player():
    """
    创建实时音频播放器，返回 (on_chunk_callback, sync_fn, close_fn)。

    on_chunk(chunk_data: np.ndarray, sample_rate: int):
        每收到一个 TTS 音频 chunk 时调用，送入播放队列。

    sync_fn():
        等待当前句子的所有 chunk 播放完毕。

    close_fn():
        通知播放线程结束并关闭流。
    """
    try:
        import sounddevice as sd
    except ImportError:
        print("提示: 安装 sounddevice 以支持实时播放: pip install sounddevice")
        return None, None, None

    _SENTENCE_END = object()

    q: queue.Queue = queue.Queue()
    _done_flag = threading.Event()
    _sentence_done = threading.Event()

    _enqueued_samples = [0]
    _dequeued_samples = [0]

    def _player_thread():
        stream = None
        try:
            item = q.get()
            if item is None:
                _done_flag.set()
                return
            first_data, sr = item

            stream = sd.OutputStream(
                samplerate=int(sr),
                channels=1,
                dtype='float32',
            )
            stream.start()

            _dequeued_samples[0] += len(first_data)
            stream.write(first_data)

            while True:
                item = q.get()
                if item is None:
                    break
                if item is _SENTENCE_END:
                    _sentence_done.set()
                    continue
                chunk_data, _ = item
                if len(chunk_data) == 0:
                    continue
                _dequeued_samples[0] += len(chunk_data)
                stream.write(chunk_data)

        except Exception as e:
            print(f"  [播放错误] {e}")
        finally:
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
            _done_flag.set()

    t = threading.Thread(target=_player_thread, daemon=True, name="tts-player")
    t.start()

    def on_chunk(chunk_data: np.ndarray, sample_rate: int):
        if chunk_data is None or len(chunk_data) == 0:
            return
        try:
            if chunk_data.dtype != np.float32:
                chunk_data = chunk_data.astype(np.float32)
            if chunk_data.ndim > 1:
                chunk_data = chunk_data.flatten()
            _enqueued_samples[0] += len(chunk_data)
            q.put((chunk_data.copy(), sample_rate))
        except Exception as e:
            print(f"  [入队错误] {e}")

    def sync():
        q.put(_SENTENCE_END)
        _sentence_done.wait()
        _sentence_done.clear()

    def close():
        q.put(None)
        _done_flag.wait(timeout=30)

    return on_chunk, sync, close


# ============================================
# 主流程
# ============================================

def main():
    print(f"模型: {MODEL_NAME}")
    print(f"句子数: {len(SENTENCES)}")
    if not SENTENCES:
        print("警告: 句子列表为空，请在 SENTENCES 中填入测试文本后重新运行。")
        return

    # 创建 TTS 实例（使用 piper 原生库）
    try:
        tts = create_piper_tts_from_config(MODEL_NAME, "models.toml")
    except (FileNotFoundError, ValueError) as e:
        print(f"错误: {e}")
        sys.exit(1)

    # 预热 — 三步走覆盖所有代码路径
    # 1) piper 内部模型懒加载（纯中文）
    # 2) generate_stream 纯中文路径
    # 3) generate_stream 含孤立英文字母路径（匹配测试句模式）
    #    注意：不能用英文单词，_preprocess_mixed_text 会把 "good"
    #    变成 "g o o d "，8个空格让 g2pw 处理时间暴涨。
    print("预热中...")
    tts.warmup()
    tts.generate_stream("预热文本，确保所有组件初始化完成。")
    tts.generate_stream("维生素c和d很重要。")
    print("  预热完成")

    # 输出目录：result_piper/<模型名>/
    output_dir = os.path.join(os.getcwd(), OUTPUT_DIR, MODEL_NAME)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")
    print("-" * 50)

    # 统计数据
    stats = {
        "model": MODEL_NAME,
        "timestamp": datetime.now().isoformat(),
        "total_sentences": len(SENTENCES),
        "records": [],
    }

    # 实时播放器
    player_on_chunk, player_sync, player_close = _create_realtime_player()
    if player_on_chunk is None:
        print("实时播放初始化失败，请安装 sounddevice: pip install sounddevice")
        sys.exit(1)
    print("实时播放已启用")

    # 逐条合成
    start_time = datetime.now()
    for i, sentence in enumerate(SENTENCES, 1):
        safe_name = "".join(c for c in sentence[:20] if c not in r'\\/:*?"<>|')
        filename = f"{i:02d}_{safe_name}.wav"
        output_path = os.path.join(output_dir, filename)

        print(f"[{i}/{len(SENTENCES)}] {sentence[:30]}...")
        try:
            result = tts.generate_stream(sentence, on_chunk=player_on_chunk)
            sr = result.sample_rate
            duration = len(result.audio) / sr
            elapsed = result.synthesis_time
            ttft = result.ttft
            rtf = duration / elapsed if elapsed > 0 else 0.0

            print(f"    [OK] 耗时={elapsed:.3f}s TTFT={ttft:.4f}s "
                  f"时长={duration:.2f}s RTF={rtf:.1f}x chunks={result.num_chunks}")

            # 保存 wav
            tts._save_wav(result.audio, sr, output_path)

            # 等待当前句子播放完毕
            if player_sync:
                player_sync()

            stats["records"].append({
                "index": i,
                "text": sentence,
                "filename": filename,
                "synthesis_time_s": round(elapsed, 4),
                "ttft_s": ttft,
                "audio_duration_s": round(duration, 3),
                "real_time_factor": round(rtf, 1),
                "sample_rate": sr,
                "num_chunks": result.num_chunks,
                "success": True,
            })
        except Exception as e:
            print(f"    [FAIL] 失败: {e}")
            stats["records"].append({
                "index": i,
                "text": sentence,
                "filename": filename,
                "success": False,
                "error": str(e),
            })

    # 关闭播放器
    if player_close:
        player_close()

    elapsed_total = (datetime.now() - start_time).total_seconds()

    # 汇总统计
    success_records = [r for r in stats["records"] if r.get("success")]
    fail_count = len(stats["records"]) - len(success_records)
    if success_records:
        total_synth = sum(r.get("synthesis_time_s", 0) for r in success_records)
        total_audio = sum(r.get("audio_duration_s", 0) for r in success_records)
        avg_rtf = total_audio / total_synth if total_synth > 0 else 0
        avg_speed = total_synth / len(success_records)
        avg_ttft = sum(r.get("ttft_s", 0) for r in success_records) / len(success_records)
        max_ttft = max(r.get("ttft_s", 0) for r in success_records)
        min_ttft = min(r.get("ttft_s", 0) for r in success_records)
    else:
        total_synth = total_audio = avg_rtf = avg_speed = avg_ttft = max_ttft = min_ttft = 0

    stats["summary"] = {
        "success_count": len(success_records),
        "fail_count": fail_count,
        "total_wall_time_s": round(elapsed_total, 1),
        "total_synthesis_time_s": round(total_synth, 3),
        "total_audio_duration_s": round(total_audio, 3),
        "average_rtf": round(avg_rtf, 1),
        "average_synthesis_time_s": round(avg_speed, 4),
        "average_ttft_s": round(avg_ttft, 4),
        "min_ttft_s": round(min_ttft, 4),
        "max_ttft_s": round(max_ttft, 4),
    }

    # 保存统计
    stats_path = os.path.join(output_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"统计数据已保存: {stats_path}")

    print("-" * 50)
    summary = stats["summary"]
    print(f"全部完成！成功 {summary['success_count']}/{len(SENTENCES)}")
    print(f"  平均耗时: {summary['average_synthesis_time_s']:.4f}s")
    print(f"  平均TTFT: {summary['average_ttft_s']:.4f}s "
          f"({summary['min_ttft_s']:.4f}s~{summary['max_ttft_s']:.4f}s)")
    print(f"  平均RTF:  {summary['average_rtf']:.1f}x")
    print(f"  结果目录: {output_dir}")


if __name__ == "__main__":
    main()
