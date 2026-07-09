"""
隐私判断训练数据生成脚本
根据 GB/T 45574-2025《数据安全技术 敏感个人信息处理安全要求》生成训练数据
"""
import os
import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

# 导入 LLM 客户端
from llm_client import create_client

# 加载环境变量
load_dotenv()

# GB/T 45574-2025 定义的敏感个人信息类别
PRIVACY_CATEGORIES = {
    "biometric": "生物识别信息（指纹、人脸、虹膜、声纹等）",
    "religious": "宗教信仰信息",
    "specific_identity": "特定身份信息（身份证、护照、驾照、社保卡等）",
    "medical": "医疗健康信息（病历、诊断、用药记录等）",
    "financial": "金融账户信息（银行卡号、支付密码、交易记录等）",
    "location": "精确位置信息（家庭地址、实时位置、行踪轨迹等）",
    "communication": "通信内容（邮件、短信、聊天记录等）",
    "personal_identifiable": "个人身份信息（姓名、电话、邮箱等）",
    "minor": "未成年人个人信息",
    "health_tracking": "健康监测信息（运动数据、睡眠数据、心率等）"
}

# 非隐私类别（用于生成负样本）
NO_PRIVACY_CATEGORIES = {
    "public_info": "公开信息（新闻、天气、常识等）",
    "others_privacy": "他人的隐私信息（不涉及用户本人）",
    "general_chat": "普通聊天（问候、闲聊等）",
    "technical_question": "技术问题（编程、软件使用等）",
    "entertainment": "娱乐内容（电影、音乐、游戏等）"
}


def generate_privacy_samples(
    llm_client,
    category: str,
    category_desc: str,
    num_samples: int = 20,
    batch_size: int = 5
) -> List[str]:
    """生成包含特定隐私类别的样本"""
    prompt = f"""
请生成 {num_samples} 条中文聊天语句，每条语句都包含"{category_desc}"。

要求：
1. 语句要自然，像是真实用户在聊天中说的话
2. 每条语句单独一行，不要编号
3. 语句长度在 10-50 字之间
4. 只输出语句，不要解释

例如（仅作参考，不要照搬）：
我的人脸识别信息可能被泄露了
我的身份证号是 XXX
"""

    samples = []
    try:
        response = llm_client.generate(
            prompt=prompt,
            system_prompt="你是一个数据生成助手，负责生成包含特定隐私信息的聊天语句。",
            temperature=0.9,
            max_tokens=2000
        )

        # 解析响应
        lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
        samples = [line for line in lines if not line[0].isdigit() or '. ' not in line[:3]][:num_samples]

    except Exception as e:
        print(f"生成隐私样本时出错 ({category}): {e}")

    return samples


def generate_no_privacy_samples(
    llm_client,
    category: str,
    category_desc: str,
    num_samples: int = 20
) -> List[str]:
    """生成不包含隐私的样本（负样本）"""
    prompt = f"""
请生成 {num_samples} 条中文聊天语句，属于"{category_desc}"类别，且不包含任何用户本人的隐私信息。

要求：
1. 语句要自然，像是真实用户在聊天中说的话
2. 每条语句单独一行，不要编号
3. 语句长度在 10-50 字之间
4. 只输出语句，不要解释

例如（仅作参考，不要照搬）：
今天天气真好
你觉得这部电影怎么样
"""

    samples = []
    try:
        response = llm_client.generate(
            prompt=prompt,
            system_prompt="你是一个数据生成助手，负责生成不包含隐私信息的聊天语句。",
            temperature=0.9,
            max_tokens=2000
        )

        # 解析响应
        lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
        samples = [line for line in lines if not line[0].isdigit() or '. ' not in line[:3]][:num_samples]

    except Exception as e:
        print(f"生成非隐私样本时出错 ({category}): {e}")

    return samples


def generate_few_shot_examples() -> List[Dict]:
    """生成 few-shot 示例，用于指导 LLM 理解任务"""
    examples = [
        {"text": "我的身份证号是 110101199001011234", "label": "privacy"},
        {"text": "今天天气真好", "label": "no_privacy"},
        {"text": "我的银行卡密码是 123456", "label": "privacy"},
        {"text": "帮我查一下明天的天气", "label": "no_privacy"},
        {"text": "我刚才测的心率是 85 次/分", "label": "privacy"},
        {"text": "这本书讲的是什么内容", "label": "no_privacy"},
        {"text": "这是我的护照号码 E12345678", "label": "privacy"},
        {"text": "别人的身份证号码是 XXX", "label": "no_privacy"},  # 重要：他人的隐私不算
    ]
    return examples


def augment_data(texts: List[str], llm_client, augment_times: int = 2) -> List[str]:
    """数据增强：使用 LLM 进行 paraphrase"""
    augmented = []

    for text in texts:
        prompt = f"""
请将以下句子用不同的方式表达出来，生成 {augment_times} 个变体。

原句：{text}

要求：
1. 保持原意不变
2. 每条单独一行，不要编号
3. 只输出变体句子
"""

        try:
            response = llm_client.generate(
                prompt=prompt,
                system_prompt="你是一个文本改写助手。",
                temperature=0.8,
                max_tokens=1000
            )

            lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
            augmented.extend(lines[:augment_times])

        except Exception as e:
            print(f"数据增强时出错: {e}")

    return augmented


def save_to_csv(data: List[Dict], output_path: str, mode: str = 'w'):
    """保存数据到 CSV 文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, mode, encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'label'])
        if mode == 'w':
            writer.writeheader()
        writer.writerows(data)

    print(f"已保存 {len(data)} 条数据到 {output_path}")


def main():
    parser = argparse.ArgumentParser(description='生成隐私判断训练数据')
    parser.add_argument('--llm_backend', type=str, default='ollama',
                        choices=['ollama', 'qwen', 'spark'],
                        help='LLM 后端')
    parser.add_argument('--num_per_category', type=int, default=50,
                        help='每个类别生成的样本数')
    parser.add_argument('--output_path', type=str, default='data/privacy/generated_train.csv',
                        help='输出文件路径')
    parser.add_argument('--augment', action='store_true',
                        help='是否进行数据增强')
    parser.add_argument('--augment_times', type=int, default=2,
                        help='每个样本增强的次数')

    args = parser.parse_args()

    # 创建 LLM 客户端
    llm_client = create_client(args.llm_backend)
    print(f"使用 LLM 后端: {args.lm_backend}")

    all_data = []

    # 生成隐私样本（正样本）
    print("\n=== 生成隐私样本（正样本）===")
    for category, desc in PRIVACY_CATEGORIES.items():
        print(f"生成类别: {category} - {desc}")
        samples = generate_privacy_samples(llm_client, category, desc, args.num_per_category)

        for sample in samples:
            all_data.append({
                'text': sample,
                'label': 'privacy'
            })

        print(f"  生成 {len(samples)} 条样本")

    # 生成非隐私样本（负样本）
    print("\n=== 生成非隐私样本（负样本）===")
    for category, desc in NO_PRIVACY_CATEGORIES.items():
        print(f"生成类别: {category} - {desc}")
        samples = generate_no_privacy_samples(llm_client, category, desc, args.num_per_category)

        for sample in samples:
            all_data.append({
                'text': sample,
                'label': 'no_privacy'
            })

        print(f"  生成 {len(samples)} 条样本")

    # 数据增强（可选）
    if args.augment:
        print("\n=== 数据增强 ===")
        privacy_texts = [d['text'] for d in all_data if d['label'] == 'privacy']
        no_privacy_texts = [d['text'] for d in all_data if d['label'] == 'no_privacy']

        augmented_privacy = augment_data(privacy_texts, llm_client, args.augment_times)
        augmented_no_privacy = augment_data(no_privacy_texts, llm_client, args.augment_times)

        for text in augmented_privacy:
            all_data.append({'text': text, 'label': 'privacy'})
        for text in augmented_no_privacy:
            all_data.append({'text': text, 'label': 'no_privacy'})

        print(f"增强后总计: {len(all_data)} 条样本")

    # 保存数据
    save_to_csv(all_data, args.output_path)

    # 打印统计信息
    privacy_count = sum(1 for d in all_data if d['label'] == 'privacy')
    no_privacy_count = sum(1 for d in all_data if d['label'] == 'no_privacy')
    print(f"\n=== 数据统计 ===")
    print(f"总样本数: {len(all_data)}")
    print(f"隐私样本: {privacy_count} ({privacy_count/len(all_data)*100:.1f}%)")
    print(f"非隐私样本: {no_privacy_count} ({no_privacy_count/len(all_data)*100:.1f}%)")

    print(f"\n✅ 数据生成完成！文件保存在: {args.output_path}")
    print("下一步：")
    print(f"1. 检查生成的数据质量")
    print(f"2. 合并到训练集: python scripts/merge_datasets.py")
    print(f"3. 开始训练: python scripts/train_lora.py --train_csv {args.output_path} --label_list 'privacy,no_privacy'")


if __name__ == "__main__":
    main()

