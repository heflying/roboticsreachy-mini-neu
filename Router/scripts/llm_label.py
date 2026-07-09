"""
scripts/llm_label.py

Re-label existing training CSV files using LLM judgment.

Reads a CSV with columns (text, label), sends each text to an LLM with
a labeling prompt, and outputs a new CSV where label is replaced by
the LLM's judgment. Labels are strictly matched to privacy/no_privacy;
unmatched responses are labeled as unknown.

Supports:
  - Multiple input files via --input (comma-separated or multiple flags)
  - Resume: skips rows already present in output file
  - --llm_backend selection

Usage:
    python scripts/llm_label.py --input data/privacy/datasets/privacy-care-interactions/train.csv
    python scripts/llm_label.py --input file1.csv,file2.csv --llm_backend spark
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys

# Ensure project root is on sys.path for llm_client
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.llm_client import create_client

logger = logging.getLogger(__name__)

VALID_LABELS = {"privacy", "no_privacy"}

# Default labeling prompt template
DEFAULT_LABEL_PROMPT = """
# Role
现在你是一个专为高性能网关设计的"隐私与敏感信息"分类器。你的唯一任务是判断用户的输入是否包含需要拦截的隐私或机密信息。

# Definitions (敏感信息定义)
- 宗教信仰信息：
    --信仰的宗教派别、归属的宗教组织
    --宗教组织中的职位或头衔
    --参加的宗教仪式、特殊宗教习俗
    --宗教相关饮食禁忌等敏感特征
- 特定身份信息：
    --残障人士身份（残疾证信息、残疾类别）
    --不适宜公开的职业身份（如未成年人犯罪记录中的特定身份）
    --因政策法规需要特殊保护的身份类别
    --贫困救助对象等易受歧视身份
- 医疗健康信息：
    --身体状况类：病症描述、既往病史、家族病史、传染病史、生育信息
    --诊疗就诊记录：住院志、医嘱单、手术及麻醉记录、护理记录、病程记录
    --检验检查数据：检验报告、影像报告（CT/X光等）、病理报告、体检结论
    --用药与康复：用药记录、过敏信息、输血信息、康复计划
    --心理健康评估、精神类疾病相关信息
- 金融账户信息：
    --银行/证券/基金/保险/公积金账号及密码
    --支付账户、银行卡磁道数据或芯片等效信息
    --收入明细、账户余额、交易流水
    --信用卡安全码（CVV）、有效期等验证信息
    --理财持仓、投资偏好等风险敏感数据
- 行踪轨迹信息：
    --连续精准定位轨迹、实时GPS位置序列
    --车辆行驶轨迹（行车记录仪联网数据）
    --人员活动路线、出入敏感地点记录
    --基于基站/WiFi的连续性定位追踪数据
    --固定时间段内居住地、办公地频繁往返路线
- 其他敏感个人信息：
    --精准定位信息（实时单点精准位置）
    --居民身份证照片（含照片的身份证影像）
    --性取向、性生活及私密生活信息
    --征信信息、信用报告详细记录
    --犯罪记录（含案底、刑事处罚记录）
    --显示身体私密部位的照片、视频或生物特征图像
    --工会会员特定歧视性信息等
-- PII类型隐私信息
    --身份证号、护照号、驾驶证号等政府颁发的身份证明号码
    --姓名、昵称、用户名等直接或间接识别个人身份的信息
    --电话号码、电子邮件地址等联系方式
    --家庭住址、工作单位地址等地理位置信息
    --社交媒体账号、在线身份标识等网络身份信息

# Rules
1. 你的输出只能是 [隐私] 或 [非隐私] 这两个固定标签之一，然后输出原因。
2. 严禁输出任何多余的标点符号、解释、分析或空格。
3. 只要发现【任何一项】上面提到的用户相关的敏感信息，就判定为 [隐私]。
4. 如果完全安全，判定为 [非隐私]。
5. 隐私只针对用户个人的信息，他人的信息不属于隐私，除非涉及用户与他人关系的敏感信息（如家庭成员的姓名和联系方式）。

# Examples
输入：我的手机号是13800000000
输出：隐私#个人手机号属于个人隐私内容

输入：请问明天的天气怎么样？
输出：非隐私#天气是公开信息，没有敏感信息。

输入：老板，今天下午的会议几点开始？
输出：非隐私#会议时间不属于敏感信息。

输入：我叫林家栋，很高兴认识你
输出：隐私#包含用户姓名，输入隐私信息。

输入：英语很重要，我应该如何开始进行学习啊
输出：非隐私#英语是一种语言，没有隐私信息，学习英语也不涉及隐私信息。

输入：我心脏有问题，运动多了就会不舒服
输出：隐私#涉及到用户的身体健康信息，属于个人隐私。

输入：今天天气很好，我一会儿打算去运动运动
输出：非隐私#天气不属于隐私，出去运动是日常生活内容，未涉及隐私信息。

基于以下内容判断用户最后一句话是否涉及隐私，并给出合理原因。

{text}

"""


def parse_label(response: str) -> str:
    """Parse LLM response into a label.

    Strips whitespace, lowercases, and matches against valid labels.
    Returns 'unknown' if no valid label is found.
    """
    cleaned = response.strip().lower()
    # Remove common formatting artifacts
    if "非隐私" in cleaned:
        return "no_privacy"
    elif "隐私" in cleaned:
        return "privacy"
    else:
        return "unknown"

def process_file(input_path: str, client, label_prompt: str):
    """Process a single CSV file through LLM labeling."""
    if not os.path.exists(input_path):
        logger.error("Input file not found: %s", input_path)
        return

    # Determine output path: same dir, train_llm_labeled.csv
    input_dir = os.path.dirname(os.path.abspath(input_path))
    output_path = os.path.join(input_dir, "train_llm_labeled.csv")

    # Read input CSV
    rows = []
    with open(input_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        logger.warning("No rows in %s", input_path)
        return

    logger.info("Loaded %d rows from %s", len(rows), input_path)

    # Collect all rows, then sort by label order
    label_order = {"privacy": 0, "no_privacy": 1, "unknown": 2}
    results = []

    for i, row in enumerate(rows):
        text = row.get("text", "")

        if not text:
            results.append((text, "unknown"))
            continue

        prompt = label_prompt.format(text=text)
        try:
            response = client.generate(prompt)
            llm_label = parse_label(response)
        except Exception as e:
            logger.error("Row %d: LLM labeling failed: %s", i + 1, e)
            llm_label = "unknown"

        results.append((text, llm_label))

        if (i + 1) % 10 == 0 or i + 1 == len(rows):
            logger.info("Progress: %d/%d (last label: %s)", i + 1, len(rows), llm_label)

    # Sort by label order
    results.sort(key=lambda x: label_order.get(x[1], 99))

    # Write sorted output
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        for text, llm_label in results:
            writer.writerow([text, llm_label])

    logger.info("Output: %s (sorted by label)", output_path)


def main():
    parser = argparse.ArgumentParser(description="Re-label training CSV using LLM")
    parser.add_argument("--input", required=True, nargs="+",
                        help="Input CSV file(s). Multiple files can be specified.")
    parser.add_argument("--llm_backend", default="ollama", choices=["ollama", "qwen", "spark"],
                        help="LLM backend (default: ollama)")
    parser.add_argument("--label_prompt", default=None,
                        help="Custom labeling prompt template (use {text} as placeholder)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    label_prompt = args.label_prompt or DEFAULT_LABEL_PROMPT
    client = create_client(args.llm_backend)
    logger.info("Using backend=%s, model=%s", args.llm_backend, client.model)

    # Expand comma-separated inputs
    input_files = []
    for inp in args.input:
        if "," in inp:
            input_files.extend(inp.split(","))
        else:
            input_files.append(inp)

    for input_path in input_files:
        input_path = input_path.strip()
        logger.info("Processing: %s", input_path)
        process_file(input_path, client, label_prompt)

    return 0


if __name__ == "__main__":
    sys.exit(main())
