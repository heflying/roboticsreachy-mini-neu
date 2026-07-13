#!/usr/bin/env python3
"""
基于 EvalScope 的评测脚本
遍历 dataset/ 下所有 JSONL 文件，逐个评测每个子维度
用法: python run_eval.py
配置: 复制 .env.example 为 .env 并填入你的配置
"""

import json
import shutil
from pathlib import Path
from dotenv import load_dotenv
import os
from evalscope import run_task
from evalscope.config import TaskConfig

# ==== 导入 Ollama 原生 API（禁用 thinking 模式） ================
# 必须在创建 TaskConfig 之前导入，以触发注册
import ollama_native_model
# ================================================================

# ==== 自定义提示词 + 选项格式 ====================================
def _apply_eval_customizations():
    """统一应用所有评测自定义：提示词模板 + 选项空格分隔"""
    import evalscope.utils.multi_choices as _mc
    from evalscope.utils.multi_choices import MultipleChoiceTemplate

    # 1. 自定义中文单选模板（普通模式）
    MultipleChoiceTemplate.CHINESE_SINGLE_ANSWER_TEMPLATE = """[提示] 你是老人陪伴机器人，请尊重用户，对用户的问题保持耐心。回答下面的单项选择题，请选出其中的正确答案。你的回答的全部内容应该是这样的格式："答案：[LETTER]"（不带引号），其中 [LETTER] 是 {letters} 中的一个。  [问题]：{question}  [选项]：{choices} """

    # 2. 自定义中文单选 CoT 模板（链式思考模式，开启 use_cot 时使用）
    MultipleChoiceTemplate.CHINESE_SINGLE_ANSWER_TEMPLATE_COT = """[提示] 你是老人陪伴机器人，请尊重用户，对用户的问题保持耐心。回答下面的单项选择题，请选出其中的正确答案。你的回答的全部内容应该是这样的格式："[思考]\n答案：[LETTER]"（不带引号），其中 [LETTER] 是 {letters} 中的一个。  [问题]：{question}  [选项]：{choices}"""

    # 3. 选项之间用两个空格分隔（替代默认换行分隔）
    _original = _mc.answer_options

    def _answer_options_space(choices):
        from evalscope.api.evaluator import Choices
        if isinstance(choices, list):
            choices = Choices(choices)
        indexes = list(range(len(choices)))
        return '  '.join([f'[{_mc.answer_character(i)}] {choices[j].value}' for i, j in enumerate(indexes)])

    _mc.answer_options = _answer_options_space

_apply_eval_customizations()
# ================================================================

# 加载 .env 文件
load_dotenv()

# 项目根目录
ROOT = Path(__file__).parent.parent
DATASET_ROOT = ROOT / "dataset"
DATASETS_DIR = Path(__file__).parent / "datasets"

# 从 .env 读取模型配置
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5:1.5b-instruct")
# Ollama 原生 API 地址（不是 OpenAI 兼容接口）
API_URL = os.getenv("API_URL", "http://localhost:11434")
API_KEY = os.getenv("API_KEY", "ollama")
MODEL_ID = MODEL_NAME.replace(":", "_").replace("/", "_")  # EvalScope 输出的目录名

def prepare_datasets() -> list[str]:
    """扫描 dataset/ 目录，准备 EvalScope 数据集，返回 subset_list"""
    if DATASETS_DIR.exists():
        shutil.rmtree(DATASETS_DIR)
    DATASETS_DIR.mkdir(parents=True)

    subset_list = []

    for category_dir in sorted(DATASET_ROOT.iterdir()):
        if not category_dir.is_dir() or category_dir.name.startswith("."):
            continue
        # 只处理数字开头的文件夹（如 "03_智力"）
        if not category_dir.name[0].isdigit():
            print(f"  跳过非数字开头文件夹: {category_dir.name}")
            continue
        category_name = category_dir.name  # 如 "03_智力"
        for jsonl_file in sorted(category_dir.glob("*.jsonl")):
            subset_name = f"{category_name}_{jsonl_file.stem}"  # 如 "03_智力_指令遵循"
            subset_list.append(subset_name)
            # 复制到 datasets/ 目录，重命名为 {subset}_val.jsonl
            target = DATASETS_DIR / f"{subset_name}_val.jsonl"
            shutil.copy(jsonl_file, target)
            print(f"  已准备: {subset_name}")

    print(f"\n共准备 {len(subset_list)} 个评测子集\n")
    return subset_list


def run_evaluation(subsets: list[str]) -> Path:
    """运行 EvalScope 评测，返回输出目录路径"""
    print("=" * 60)
    print("开始评测...")
    print("=" * 60)

    script_dir = Path(__file__).parent
    outputs_dir = script_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # 1. 评测前清空 latest/ 目录
    latest_dir = outputs_dir / "latest"
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    latest_dir.mkdir(parents=True, exist_ok=True)

    task_cfg = TaskConfig(
        model=MODEL_NAME,
        api_url=API_URL,
        api_key=API_KEY,
        eval_type='ollama_native',  # 使用 Ollama 原生 API（支持 think=false）
        datasets=['general_mcq'],
        dataset_args={
            'general_mcq': {
                'local_path': str(DATASETS_DIR),
                'subset_list': subsets,
                'few_shot_num': 0,
                'extra_params': {
                    'use_cot': True,  # 开启链式思考模式
                }
            }
        },
        work_dir=str(latest_dir),
    )

    result = run_task(task_cfg)

    # 2. 找出新生成的时间戳子目录（latest/ 中唯一的子目录）
    subdirs = [p for p in latest_dir.iterdir() if p.is_dir()]
    if not subdirs:
        raise RuntimeError(f"评测后未找到输出目录 in {latest_dir}")
    
    # 取最新修改的子目录
    eval_output_dir = max(subdirs, key=lambda p: p.stat().st_mtime)

    # 3. 复制到 outputs/ 根目录，按规则重命名
    # 规则：{MODEL_ID}_{EvalScope生成的时间戳}
    evalcope_timestamp = eval_output_dir.name
    final_dir_name = f"{MODEL_ID}_{evalcope_timestamp}"
    final_dir = outputs_dir / final_dir_name

    # 如果目标目录已存在，先删除
    if final_dir.exists():
        shutil.rmtree(final_dir)

    # 复制前等待文件句柄释放
    import gc, time
    gc.collect()
    time.sleep(2)

    # 重试复制，解决 Windows 文件句柄未释放问题
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"复制结果到: {final_dir}")
            shutil.copytree(str(eval_output_dir), str(final_dir))
            break
        except (OSError, PermissionError) as e:
            if attempt < max_retries - 1:
                wait = 2 * (attempt + 1)
                print(f"  复制失败（第{attempt+1}次），{wait}秒后重试... 错误: {e}")
                gc.collect()
                time.sleep(wait)
            else:
                raise

    print(f"输出目录: {final_dir}")
    return final_dir


def parse_results(output_dir: Path, subsets: list[str]) -> dict:
    """解析评测结果，返回 {subset: {accuracy, correct, total, errors}}"""
    results = {}
    # 使用统一配置的模型ID（EvalScope 输出的目录名）
    reviews_dir = output_dir / "reviews" / MODEL_ID

    if not reviews_dir.exists():
        print(f"警告: 结果目录不存在 {reviews_dir}")
        return results

    for jsonl_file in reviews_dir.glob("general_mcq_*.jsonl"):
        subset = jsonl_file.stem[len("general_mcq_"):]
        if subset not in subsets:
            continue
        scores = []
        errors = []  # 收集答错的题目详情
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                acc = item.get("sample_score", {}).get("score", {}).get("value", {}).get("acc", 0)
                scores.append(acc)

                # 收集答错的题目
                if acc == 0:
                    target = item.get("target", "")
                    pred = item.get("sample_score", {}).get("score", {}).get("extracted_prediction", "")
                    sample_id = item.get("sample_metadata", {}).get("id", item.get("index", ""))
                    # 从 messages[0].content 提取问题（去掉 prompt 模板前缀）
                    question_raw = item.get("messages", [{}])[0].get("content", "")
                    # 保存完整问题文字
                    errors.append({
                        "id": sample_id,
                        "question": question_raw,
                        "model_answer": pred,
                        "correct_answer": target
                    })
        total = len(scores)
        correct = int(sum(scores))
        accuracy = sum(scores) / total if total > 0 else 0
        results[subset] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "errors": errors
        }

    return results


def print_summary(results: dict):
    """打印每个子维度的正确率汇总表，并输出答错题目详情"""
    print("\n" + "=" * 70)
    print("评测结果汇总")
    print("=" * 70)

    # 按大类分组
    grouped = {}
    all_errors = []  # 收集所有错误题目
    for subset, data in results.items():
        parts = subset.split("_", 1)
        if len(parts) == 2:
            category = parts[0]
            name = parts[1]
        else:
            category = "其他"
            name = subset

        if category not in grouped:
            grouped[category] = []
        grouped[category].append((name, data))

        # 收集错误题目
        for err in data.get("errors", []):
            all_errors.append((name, err))

    # 打印分组正确率
    for category in sorted(grouped.keys()):
        print(f"\n[{category}]")
        for name, data in grouped[category]:
            acc = data["accuracy"]
            cor = data["correct"]
            tot = data["total"]
            mark = " [错误]" if acc < 1.0 else ""
            print(f"  {name:<30} {cor}/{tot} = {acc:.1%}{mark}")

    # 打印总计
    total_correct = sum(d["correct"] for d in results.values())
    total_questions = sum(d["total"] for d in results.values())
    overall_acc = total_correct / total_questions if total_questions > 0 else 0

    print("\n" + "-" * 70)
    print(f"总计: {total_correct}/{total_questions} = {overall_acc:.1%}")
    print("=" * 70)

    # 打印答错题目详情
    if all_errors:
        print("\n答错题目详情:")
        print("-" * 70)
        for i, (name, err) in enumerate(all_errors, 1):
            print(f"\n[{i}] 维度: {name}")
            print(f"    题目ID: {err['id']}")
            # 截取问题前200字符，避免太长
            q = err['question'][:200].replace('\n', ' ')
            print(f"    题目: {q}...")
            print(f"    模型回答: {err['model_answer']}")
            print(f"    正确答案: {err['correct_answer']}")
        print("\n" + "=" * 70)

    print()  # 末尾空行


def save_errors(results: dict, output_dir: Path):
    """将答错题目详情保存到文件"""
    all_errors = []
    for subset, data in results.items():
        for err in data.get("errors", []):
            all_errors.append({
                "dimension": subset,
                "id": err["id"],
                "question": err["question"],
                "model_answer": err["model_answer"],
                "correct_answer": err["correct_answer"]
            })

    if not all_errors:
        print("没有答错题目，无需保存错误详情")
        return

    # 保存为 JSON
    json_path = output_dir / "error_details.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_errors, f, ensure_ascii=False, indent=2)
    print(f"\n答错题目已保存到: {json_path}")

    # 保存为可读文本
    txt_path = output_dir / "error_details.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("答错题目详情\n")
        f.write("=" * 70 + "\n\n")
        for i, err in enumerate(all_errors, 1):
            f.write(f"[{i}] 维度: {err['dimension']}\n")
            f.write(f"    题目ID: {err['id']}\n")
            q = err["question"].replace("\n", " ")
            f.write(f"    题目: {q}\n")
            f.write(f"    模型回答: {err['model_answer']}\n")
            f.write(f"    正确答案: {err['correct_answer']}\n\n")
    print(f"答错题目已保存到: {txt_path}")


def main():
    print("\n" + "=" * 60)
    print("EvalScope 批量评测")
    print("=" * 60)
    print(f"模型: {MODEL_NAME}")
    print(f"后端: Ollama (http://localhost:11434/v1)")
    print(f"数据集目录: {DATASET_ROOT}")
    print("=" * 60 + "\n")

    # 1. 准备数据集
    print("【步骤1】准备数据集...")
    subsets = prepare_datasets()

    if not subsets:
        print("错误: 未找到任何 JSONL 文件")
        return

    # 2. 运行评测
    output_dir = run_evaluation(subsets)

    # 3. 解析结果
    print("\n【步骤2】解析结果...")
    results = parse_results(output_dir, subsets)

    # 4. 解析结果并打印汇总
    if results:
        print_summary(results)
        # 保存答错题目详情到文件
        save_errors(results, output_dir)
    else:
        print("警告: 未解析到结果，请检查输出目录")

    print(f"详细结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()
