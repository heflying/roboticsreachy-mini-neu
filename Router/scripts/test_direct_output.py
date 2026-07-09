#!/usr/bin/env python3
"""Simple test runner for a saved classification model.
Prints per-sentence probabilities to the command line (no evaluation/HTML).

Usage examples:
  python scripts\test_direct_output.py --model_dir output_lora --sentence "这是隐私信息" --sentence "这是普通句子"
  python scripts\test_direct_output.py --model_dir output_lora --input_file sentences.txt
  python scripts\test_direct_output.py --model_dir output_lora --input_json sentences.json
"""
import argparse
import json
import os
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_model_dir", default="output_lora", help="Path to merged model directory")
    p.add_argument("--sentence", action="append", help="A single sentence to classify (can be repeated)")
    p.add_argument("--input_file", help="Path to a text/CSV file (CSV: auto-reads text column, skips header)")
    p.add_argument("--input_json", help="Path to a JSON file containing a list of sentences")
    p.add_argument("--text_column", default="text", help="Column name for CSV input (default: text)")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    return p.parse_args()


def load_sentences(args) -> List[str]:
    if args.sentence:
        return args.sentence
    if args.input_file:
        # Auto-detect CSV by file extension
        if args.input_file.lower().endswith('.csv'):
            import csv
            sentences = []
            with open(args.input_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                text_col = args.text_column
                for row in reader:
                    if text_col in row and row[text_col].strip():
                        sentences.append(row[text_col].strip())
            return sentences
        else:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
    if args.input_json:
        with open(args.input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return [str(x) for x in data]
            raise ValueError('input_json must contain a list of strings')
    # default examples
    return [
        "这是包含个人身份证号码的隐私信息：身份证 123456789012345678。",
        "今天天气很好，我们去散步吧。",
    ]


def get_label_name(config, idx: int) -> str:
    id2label = getattr(config, 'id2label', None)
    if id2label:
        if idx in id2label:
            return id2label[idx]
        if str(idx) in id2label:
            return id2label[str(idx)]
    return str(idx)


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.isdir(args.eval_model_dir):
        raise SystemExit(f"Model directory not found: {args.model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.eval_model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.eval_model_dir, local_files_only=True
    )
    model.to(device)
    model.eval()

    sentences = load_sentences(args)
    if len(sentences) == 0:
        print('No sentences provided.')
        return

    # Process in batches
    all_probs = []
    all_preds = []
    for start in range(0, len(sentences), args.batch_size):
        batch_texts = sentences[start:start + args.batch_size]
        enc = tokenizer(batch_texts, padding=True, truncation=True,
                        max_length=args.max_length, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1).tolist()
            all_probs.extend(probs.tolist())
            all_preds.extend(preds)

    # Print results
    for text, pred, prob_vec in zip(sentences, all_preds, all_probs):
        label_name = get_label_name(model.config, pred)
        probs_str = ', '.join(
            f"{get_label_name(model.config, i)}:{p:.4f}" for i, p in enumerate(prob_vec)
        )
        print('---')
        print(text)
        print(f'Pred: {pred} ({label_name})')
        print(f'Probs: {probs_str}')


if __name__ == '__main__':
    main()
