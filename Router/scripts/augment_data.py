"""
数据增强脚本 - 用于扩充隐私判断训练数据
支持多种数据增强技术：EDA、回译、同义词替换等
"""
import argparse
import csv
import random
import os
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np

# 设置随机种子
random.seed(42)
np.random.seed(42)


class EDAAugmenter:
    """EDA (Easy Data Augmentation) 增强器"""

    def __init__(self, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, alpha_rd=0.1):
        self.alpha_sr = alpha_sr  # 同义词替换比例
        self.alpha_ri = alpha_ri  # 随机插入比例
        self.alpha_rs = alpha_rs  # 随机交换比例
        self.alpha_rd = alpha_rd  # 随机删除比例

        # 常用同义词词典（简化版，实际应使用 WordNet 或同义词库）
        self.synonyms = {
            '我': ['本人', '我自己'],
            '你': ['您'],
            '是': ['等于', '为'],
            '有': ['拥有', '持有'],
            '没有': ['无', '不具备'],
            '不知道': ['不清楚', '不了解'],
            '想': ['希望', '打算'],
            '可以': ['能够', '可行'],
            '需要': ['要求', '必须'],
        }

    def synonym_replacement(self, words: List[str], n: int) -> List[str]:
        """同义词替换"""
        new_words = words.copy()
        random_indices = random.sample(range(len(words)), min(n, len(words)))

        for idx in random_indices:
            word = words[idx]
            if word in self.synonyms:
                new_words[idx] = random.choice(self.synonyms[word])

        return new_words

    def random_insertion(self, words: List[str], n: int) -> List[str]:
        """随机插入"""
        new_words = words.copy()

        for _ in range(n):
            add_word = random.choice(list(self.synonyms.keys()))
            random_idx = random.randint(0, len(new_words))
            new_words.insert(random_idx, add_word)

        return new_words

    def random_swap(self, words: List[str], n: int) -> List[str]:
        """随机交换"""
        new_words = words.copy()

        for _ in range(n):
            idx1, idx2 = random.sample(range(len(new_words)), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]

        return new_words

    def random_deletion(self, words: List[str], p: float) -> List[str]:
        """随机删除"""
        if len(words) == 1:
            return words

        new_words = []
        for word in words:
            if random.uniform(0, 1) > p:
                new_words.append(word)

        if len(new_words) == 0:
            return [random.choice(words)]

        return new_words

    def augment(self, sentence: str) -> str:
        """对句子进行 EDA 增强"""
        words = sentence.split()
        n = max(1, int(len(words) * self.alpha_sr))

        # 随机选择一种增强方法
        method = random.choice(['sr', 'ri', 'rs', 'rd'])

        if method == 'sr':
            words = self.synonym_replacement(words, n)
        elif method == 'ri':
            words = self.random_insertion(words, n)
        elif method == 'rs':
            words = self.random_swap(words, n)
        else:  # rd
            words = self.random_deletion(words, self.alpha_rd)

        return ' '.join(words)


class BackTranslationAugmenter:
    """回译增强器（需要翻译 API）"""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def augment(self, sentence: str) -> str:
        """通过回译进行增强"""
        if not self.llm_client:
            return sentence

        try:
            # 中文 -> 英文
            en_text = self.llm_client.generate(
                prompt=f"将以下中文翻译成英文：{sentence}",
                system_prompt="你是一个翻译助手。",
                temperature=0.3,
                max_tokens=500
            )

            # 英文 -> 中文（不同的表达）
            zh_text = self.llm_client.generate(
                prompt=f"将以下英文翻译回中文，使用不同的表达方式：{en_text}",
                system_prompt="你是一个翻译助手。",
                temperature=0.7,
                max_tokens=500
            )

            return zh_text.strip()

        except Exception as e:
            print(f"回译增强失败: {e}")
            return sentence


class ParaphraseAugmenter:
    """释义增强器 - 使用 LLM 进行句子改写"""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def augment(self, sentence: str, num_paraphrases: int = 1) -> List[str]:
        """生成句子的释义"""
        if not self.llm_client:
            return [sentence]

        try:
            prompt = f"""
请将以下句子用不同的方式表达出来，生成 {num_paraphrases} 个变体。

原句：{sentence}

要求：
1. 保持原意不变
2. 每条单独一行，不要编号
3. 只输出变体句子
"""

            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt="你是一个文本改写助手。",
                temperature=0.8,
                max_tokens=1000
            )

            paraphrases = [line.strip() for line in response.strip().split('\n') if line.strip()]
            return paraphrases[:num_paraphrases]

        except Exception as e:
            print(f"释义增强失败: {e}")
            return [sentence]


def augment_dataset(
    input_path: str,
    output_path: str,
    augment_times: int = 2,
    use_eda: bool = True,
    use_back_translation: bool = False,
    use_paraphrase: bool = False,
    llm_backend: str = None
):
    """
    增强数据集

    Args:
        input_path: 输入 CSV 文件路径
        output_path: 输出 CSV 文件路径
        augment_times: 每个样本增强的次数
        use_eda: 是否使用 EDA 增强
        use_back_translation: 是否使用回译增强
        use_paraphrase: 是否使用释义增强
        llm_backend: LLM 后端（用于回译和释义）
    """
    # 读取数据
    df = pd.read_csv(input_path)
    print(f"原始数据: {len(df)} 条")

    # 初始化增强器
    eda_augmenter = EDAAugmenter() if use_eda else None
    back_translation_augmenter = None
    paraphrase_augmenter = None

    if use_back_translation or use_paraphrase:
        if llm_backend:
            from llm_client import create_client
            llm_client = create_client(llm_backend)
            if use_back_translation:
                back_translation_augmenter = BackTranslationAugmenter(llm_client)
            if use_paraphrase:
                paraphrase_augmenter = ParaphraseAugmenter(llm_client)
        else:
            print("警告: 未提供 LLM 后端，跳过回译和释义增强")

    # 增强数据
    augmented_data = []

    for idx, row in df.iterrows():
        text = row['text']
        label = row['label']

        # 保留原始样本
        augmented_data.append({'text': text, 'label': label})

        # EDA 增强
        if eda_augmenter and use_eda:
            for _ in range(augment_times):
                augmented_text = eda_augmenter.augment(text)
                if augmented_text != text:  # 避免重复
                    augmented_data.append({'text': augmented_text, 'label': label})

        # 回译增强
        if back_translation_augmenter:
            for _ in range(augment_times):
                augmented_text = back_translation_augmenter.augment(text)
                if augmented_text != text:
                    augmented_data.append({'text': augmented_text, 'label': label})

        # 释义增强
        if paraphrase_augmenter:
            paraphrases = paraphrase_augmenter.augment(text, num_paraphrases=augment_times)
            for para in paraphrases:
                if para != text:
                    augmented_data.append({'text': para, 'label': label})

        if (idx + 1) % 10 == 0:
            print(f"处理进度: {idx + 1}/{len(df)}")

    # 去除重复样本
    seen = set()
    unique_data = []
    for item in augmented_data:
        key = (item['text'], item['label'])
        if key not in seen:
            seen.add(key)
            unique_data.append(item)

    # 保存增强后的数据
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    augmented_df = pd.DataFrame(unique_data)
    augmented_df.to_csv(output_path, index=False, encoding='utf-8')

    print(f"\n增强完成:")
    print(f"  原始样本数: {len(df)}")
    print(f"  增强后样本数: {len(augmented_df)}")
    print(f"  增强倍数: {len(augmented_df) / len(df):.2f}x")
    print(f"  保存路径: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='数据增强脚本')
    parser.add_argument('--input', type=str, required=True,
                        help='输入 CSV 文件路径')
    parser.add_argument('--output', type=str, required=True,
                        help='输出 CSV 文件路径')
    parser.add_argument('--augment_times', type=int, default=2,
                        help='每个样本增强的次数')
    parser.add_argument('--use_eda', action='store_true', default=True,
                        help='是否使用 EDA 增强')
    parser.add_argument('--use_back_translation', action='store_true',
                        help='是否使用回译增强（需要 LLM）')
    parser.add_argument('--use_paraphrase', action='store_true',
                        help='是否使用释义增强（需要 LLM）')
    parser.add_argument('--llm_backend', type=str,
                        choices=['ollama', 'qwen', 'spark'],
                        help='LLM 后端（用于回译和释义）')

    args = parser.parse_args()

    augment_dataset(
        input_path=args.input,
        output_path=args.output,
        augment_times=args.augment_times,
        use_eda=args.use_eda,
        use_back_translation=args.use_back_translation,
        use_paraphrase=args.use_paraphrase,
        llm_backend=args.llm_backend
    )


if __name__ == "__main__":
    main()
