"""
test_streaming_tts.py - TTS 模型测试脚本

功能：
  1. 从 .env 读取 TTS_MODEL 配置
  2. 从 models.toml 读取对应模型的参数
  3. 对句子列表逐条合成，输出 wav 文件到 result/ 目录

用法：
  python test_streaming_tts.py
  python test_streaming_tts.py --model piper-chaowen
  python test_streaming_tts.py --sentences-file my_sentences.txt
"""

import os
import sys
import json
import shutil
import argparse
import threading
import queue
import numpy as np
from datetime import datetime
from pathlib import Path

# 添加当前目录到 sys.path，以便导入 streaming_tts
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from streaming_tts import create_tts_from_env, create_tts_from_config


# ============================================
# 实时播放辅助（sounddevice）
# ============================================

def _create_realtime_player():
    """
    创建实时音频播放器，返回 (on_chunk_callback, sync_fn, close_fn, get_remaining_fn)。

    采用专用播放线程 + stream.write() 阻塞写入，避免 callback 流竞态问题。
    stream.write() 在硬件消费完毕前阻塞，天然保证实时播放速率。

    on_chunk(chunk_data: np.ndarray, sample_rate: int):
        每收到一个 TTS 音频 chunk 时调用，送入播放队列。

    sync_fn():
        等待当前句子的所有 chunk 播放完毕（在每句合成完成后调用）。

    close_fn():
        通知播放线程结束，等待所有音频播完并关闭流。

    get_remaining_fn() -> int:
        返回播放队列中尚未被播放线程取走消费的音频采样数。
    """
    try:
        import sounddevice as sd
    except ImportError:
        print("提示: 安装 sounddevice 以支持实时播放: pip install sounddevice")
        return None, None, None, None

    _SENTENCE_END = object()  # 每句结束的哨兵

    q: queue.Queue = queue.Queue()
    _done_flag = threading.Event()
    _sentence_done = threading.Event()

    # 跨句衔接：跟踪已入队和已出队（已开始播放）的采样数
    _enqueued_samples = [0]   # list 实现闭包可变计数
    _dequeued_samples = [0]

    def _player_thread():
        """专用播放线程：从队列取 chunk，通过 stream.write 阻塞播放"""
        stream = None
        try:
            # 阻塞等待第一个 chunk（同时确定采样率）
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

            # 标记已取出并播放第一个 chunk
            _dequeued_samples[0] += len(first_data)
            stream.write(first_data)

            # 逐个播放后续 chunk
            while True:
                item = q.get()
                if item is None:
                    # 终止信号
                    break
                if item is _SENTENCE_END:
                    # 当前句子播放完毕，通知主线程
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
        """将 TTS chunk 送入播放队列"""
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
        """等待当前句子的所有 chunk 播放完毕"""
        q.put(_SENTENCE_END)
        _sentence_done.wait()
        _sentence_done.clear()

    def close():
        """通知播放线程结束，等待所有音频播完"""
        q.put(None)
        _done_flag.wait(timeout=30)

    def get_remaining():
        """返回尚未被播放线程取走消费的采样数（含当前正在播放的）"""
        return max(0, _enqueued_samples[0] - _dequeued_samples[0])

    return on_chunk, sync, close, get_remaining


def get_default_sentences():
    """获取默认测试句子列表"""
    return [
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


def load_sentences_from_file(file_path: str) -> list:
    """从文件加载测试句子，每行一句"""
    path = Path(file_path)
    if not path.exists():
        print(f"句子文件不存在: {file_path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    if not sentences:
        print(f"句子文件为空: {file_path}")
        sys.exit(1)

    return sentences


def main():
    parser = argparse.ArgumentParser(description="TTS 模型测试脚本")
    parser.add_argument(
        "--model", type=str, default=None,
        help="模型名称（覆盖 .env 中的 TTS_MODEL）"
    )
    parser.add_argument(
        "--env", type=str, default=".env",
        help=".env 文件路径（相对于工作目录）"
    )
    parser.add_argument(
        "--toml", type=str, default="models.toml",
        help="models.toml 文件路径（相对于工作目录）"
    )
    parser.add_argument(
        "--sentences-file", type=str, default=None,
        help="句子列表文件（每行一句，覆盖默认句子）"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="输出目录（覆盖 .env 中的 TTS_OUTPUT_DIR）"
    )
    parser.add_argument(
        "--listen", action="store_true", default=None,
        help="实时播放 TTS 输出（需要安装 sounddevice: pip install sounddevice）"
    )
    parser.add_argument(
        "--no-listen", dest="listen", action="store_false",
        help="禁用实时播放"
    )
    args = parser.parse_args()

    # 加载 .env
    try:
        from dotenv import load_dotenv
        env_path = args.env if os.path.isabs(args.env) else os.path.join(os.getcwd(), args.env)
        if os.path.exists(env_path):
            load_dotenv(env_path)
            print(f"已加载 .env: {env_path}")
        else:
            print(f"警告: .env 文件不存在: {env_path}")
    except ImportError:
        print("警告: 未安装 python-dotenv")
        print("请运行: pip install python-dotenv")

    # 确定模型名称
    model_name = args.model or os.environ.get("TTS_MODEL")
    if not model_name:
        print("错误: 请通过 --model 或 .env 中的 TTS_MODEL 指定模型名称")
        print("可用模型请参见 eval_custom/models.toml")
        sys.exit(1)

    print(f"使用模型: {model_name}")

    # 创建 TTS 实例
    try:
        tts = create_tts_from_config(model_name, args.toml)
    except (FileNotFoundError, ValueError) as e:
        print(f"错误: {e}")
        sys.exit(1)

    # 预热：动态计算模型每字生成速度
    print("预热测速中...")
    tts.warmup()

    # 确定输出目录（result/<模型名>/）
    base_dir = args.output_dir or os.environ.get("TTS_OUTPUT_DIR", "result")
    output_dir = os.path.join(base_dir, model_name)
    # 相对于工作目录
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.getcwd(), output_dir)
    # 清空已有结果，确保本次输出干净
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    print(f"输出目录: {output_dir}")

    # 加载测试句子
    if args.sentences_file:
        sentences = load_sentences_from_file(args.sentences_file)
    else:
        sentences = get_default_sentences()

    print(f"共 {len(sentences)} 条测试句子")
    print(f"_CHARS_PER_SEC_GEN = {tts._CHARS_PER_SEC_GEN:.1f} 字/秒")
    print("-" * 50)

    # 统计数据收集
    stats = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "total_sentences": len(sentences),
        "records": [],
    }

    # 实时播放器
    listen_mode = args.listen
    player_on_chunk = None
    player_sync = None
    player_close = None
    if listen_mode:
        player_on_chunk, player_sync, player_close, _ = _create_realtime_player()
        if player_on_chunk is None:
            print("实时播放初始化失败，已跳过")
            listen_mode = False
        else:
            print("实时播放已启用")

    # 逐条合成
    start_time = datetime.now()
    for i, sentence in enumerate(sentences, 1):
        # 文件名：序号_前20个字（清理非法字符）
        safe_name = "".join(c for c in sentence[:20] if c not in r'\\/:*?"<>|')
        filename = f"{i:02d}_{safe_name}.wav"
        output_path = os.path.join(output_dir, filename)

        print(f"[{i}/{len(sentences)}] {sentence[:30]}...")
        try:
            result = tts.generate_stream(sentence, on_chunk=player_on_chunk)
            sr = result.sample_rate
            duration = len(result.audio) / sr
            elapsed = result.synthesis_time
            ttft = result.ttft
            rtf = duration / elapsed if elapsed > 0 else 0.0

            play_tag = " [play]" if listen_mode else ""
            print(f"    [OK] 耗时={elapsed:.3f}s TTFT={ttft:.4f}s "
                  f"时长={duration:.2f}s RTF={rtf:.1f}x chunks={result.num_chunks}"
                  f"{play_tag}")

            # 保存 wav
            tts._save_wav(result.audio, sr, output_path)

            # 等待当前句子播放完毕，再处理下一句
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

    # 关闭实时播放器
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

    # 保存统计数据
    stats_path = os.path.join(output_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"统计数据已保存: {stats_path}")

    print("-" * 50)
    summary = stats["summary"]
    print(f"全部完成！成功 {summary['success_count']}/{len(sentences)}")
    print(f"  平均耗时: {summary['average_synthesis_time_s']:.4f}s")
    print(f"  平均TTFT: {summary['average_ttft_s']:.4f}s "
          f"({summary['min_ttft_s']:.4f}s~{summary['max_ttft_s']:.4f}s)")
    print(f"  平均RTF:  {summary['average_rtf']:.1f}x")
    print(f"  结果目录: {output_dir}")


if __name__ == "__main__":
    main()
