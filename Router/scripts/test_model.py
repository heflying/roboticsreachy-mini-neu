#!/usr/bin/env python3
"""Evaluate a merged classification model on multiple CSV test files and generate an HTML report.

Usage examples:
  # Evaluate multiple test CSVs, generate HTML report
  python scripts/test_model.py --eval_model_dir output_lora --test_csv data/test1.csv data/test2.csv --recall_label privacy

  # Custom column names
  python scripts/test_model.py --eval_model_dir output_lora --test_csv data/test.csv --text_column sentence --label_column category --recall_label privacy

  # Custom HTML output path
  python scripts/test_model.py --eval_model_dir output_lora --test_csv data/test.csv --recall_label privacy --output_html report.html
"""
import argparse
import csv
import os
from typing import Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_model_dir", default="output_lora",
                   help="Path to merged model directory")
    # Evaluation mode (CSV with labels)
    p.add_argument("--test_csv", nargs="+", required=True,
                   help="Path(s) to test CSV file(s) with text and label columns")
    p.add_argument("--text_column", default="text",
                   help="Name of text column in CSV (default: text)")
    p.add_argument("--label_column", default="label",
                   help="Name of label column in CSV (default: label)")
    p.add_argument("--recall_label", default=None,
                   help="Label name to highlight recall for (e.g. 'privacy')")
    p.add_argument("--output_html", default="test_report.html",
                   help="Output HTML report path (default: test_report.html)")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def load_test_csv(path: str, text_column: str, label_column: str):
    """Load text and labels from a CSV file."""
    texts, labels = [], []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row[text_column])
            labels.append(row[label_column])
    return texts, labels


def compute_metrics(y_true: List[int], y_pred: List[int],
                    label2id: Dict[str, int]):
    """Compute accuracy, macro F1, and per-class precision/recall/F1."""
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    error_rate = 1.0 - acc

    precisions, recalls, f1s, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(label2id.values()), zero_division=0
    )
    per_class = {}
    for name, idx in label2id.items():
        per_class[name] = {
            'precision': precisions[idx],
            'recall': recalls[idx],
            'f1': f1s[idx],
        }
    
    # Calculate F1 for the second label (index 1)
    f1_second_label = None
    second_label_name = None
    if len(label2id) >= 2:
        # Get the second label (sorted by index)
        sorted_labels = sorted(label2id.items(), key=lambda x: x[1])
        if len(sorted_labels) >= 2:
            second_label_name = sorted_labels[1][0]
            second_label_idx = sorted_labels[1][1]
            f1_second_label = f1s[second_label_idx]
    
    return {
        'accuracy': acc,
        'error_rate': error_rate,
        'f1_macro': f1_macro,
        'f1_second_label': f1_second_label,
        'second_label_name': second_label_name,
        'per_class': per_class,
    }


def evaluate_file(model, tokenizer, device: str, path: str,
                 text_column: str, label_column: str,
                 label2id: Dict[str, int], id2label: Dict,
                 max_length: int, batch_size: int):
    """Evaluate model on a single CSV file. Returns metrics + error samples."""
    texts, true_labels_str = load_test_csv(path, text_column, label_column)

    # Build int-key label lookup
    id2str = {}
    for k, v in id2label.items():
        try:
            id2str[int(k)] = v
        except (ValueError, TypeError):
            id2str[k] = v
    for k, v in list(id2str.items()):
        id2str[str(k)] = v

    def _label_name(idx):
        return id2str.get(idx, id2str.get(str(idx), str(idx)))

    # Map string labels to ids
    true_ids = []
    for lab in true_labels_str:
        if lab in label2id:
            true_ids.append(label2id[lab])
        else:
            true_ids.append(int(lab))

    # Run inference in batches
    all_preds = []
    all_probs = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        enc = tokenizer(batch_texts, padding=True, truncation=True,
                        max_length=max_length, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1).tolist()
            all_preds.extend(preds)
            all_probs.extend(probs.tolist())

    # Compute metrics
    metrics = compute_metrics(true_ids, all_preds, label2id)

    # Collect error samples
    error_samples = []
    for i, (t, p, prob_vec) in enumerate(zip(true_ids, all_preds, all_probs)):
        if t != p:
            prob_dict = {}
            for j in range(len(prob_vec)):
                prob_dict[_label_name(j)] = float(prob_vec[j])
            error_samples.append({
                'text': texts[i],
                'true_label': _label_name(t),
                'pred_label': _label_name(p),
                'probs': prob_dict,
            })

    return metrics, error_samples, len(texts)


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

def _sorted_label_names(id2label: dict) -> list:
    """Return label names sorted by integer key."""
    items = []
    for k, v in id2label.items():
        try:
            items.append((int(k), v))
        except (ValueError, TypeError):
            items.append((k, v))
    items.sort(key=lambda x: x[0])
    return [v for _, v in items]


def generate_html_report(results: List[dict], output_path: str,
                        recall_label: str = None):
    """Generate a single HTML report with per-file sections and a summary table."""
    html = []
    html.append('<!DOCTYPE html>')
    html.append('<html lang="zh-CN">')
    html.append('<head>')
    html.append('  <meta charset="UTF-8">')
    html.append('  <title>Model Evaluation Report</title>')
    html.append('  <style>')
    html.append('    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 40px; background: #f5f5f5; }')
    html.append('    h1 { color: #333; }')
    html.append('    h2 { color: #555; margin-top: 40px; border-bottom: 2px solid #4CAF50; padding-bottom: 8px; }')
    html.append('    h3 { color: #666; margin-top: 20px; }')
    html.append('    table { border-collapse: collapse; width: 100%; margin: 16px 0; background: white; }')
    html.append('    th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; }')
    html.append('    th { background-color: #4CAF50; color: white; }')
    html.append('    tr:nth-child(even) { background-color: #f9f9f9; }')
    html.append('    tr:hover { background-color: #f1f1f1; }')
    html.append('    .metric-good { color: #2e7d32; font-weight: bold; }')
    html.append('    .metric-bad { color: #c62828; font-weight: bold; }')
    html.append('    .error-table td { max-width: 400px; word-break: break-all; }')
    html.append('    .summary { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }')
    html.append('    .file-section { background: white; padding: 20px; border-radius: 8px; margin-bottom: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }')
    html.append('  </style>')
    html.append('</head>')
    html.append('<body>')
    html.append('  <h1>Model Evaluation Report</h1>')

    # Per-file sections
    for r in results:
        path = r['path']
        fname = os.path.basename(path)
        m = r['metrics']
        errors = r['errors']
        label_names = _sorted_label_names(r['id2label'])

        html.append(f'  <div class="file-section">')
        html.append(f'    <h2>{fname}</h2>')
        html.append(f'    <p style="color:#888;">File: {path}</p>')

        # Metrics table
        html.append('    <h3>Metrics</h3>')
        html.append('    <table>')
        html.append('      <tr><th>Metric</th><th>Value</th></tr>')
        acc_class = "metric-good" if m["accuracy"] >= 0.8 else "metric-bad"
        err_class = "metric-bad" if m["error_rate"] >= 0.2 else "metric-good"
        html.append(f'      <tr><td>Accuracy</td><td class="{acc_class}">{m["accuracy"]:.4f}</td></tr>')
        html.append(f'      <tr><td>Error Rate</td><td class="{err_class}">{m["error_rate"]:.4f}</td></tr>')
        html.append(f'      <tr><td>F1 (Macro)</td><td>{m["f1_macro"]:.4f}</td></tr>')
        # Add F1 for second label if available
        if m.get('f1_second_label') is not None and m.get('second_label_name'):
            html.append(f'      <tr><td>F1 ({m["second_label_name"]})</td><td>{m["f1_second_label"]:.4f}</td></tr>')
        html.append('    </table>')

        # Per-class table (highlight recall_label)
        html.append('    <h3>Per-Class Metrics</h3>')
        html.append('    <table>')
        html.append('      <tr><th>Label</th><th>Precision</th><th>Recall</th><th>F1</th></tr>')
        for cls, vals in m['per_class'].items():
            highlight = ' style="background-color: #fff3cd;"' if recall_label and cls == recall_label else ''
            html.append(f'      <tr{highlight}>')
            html.append(f'        <td>{cls}</td>')
            html.append(f'        <td>{vals["precision"]:.4f}</td>')
            html.append(f'        <td>{vals["recall"]:.4f}</td>')
            html.append(f'        <td>{vals["f1"]:.4f}</td>')
            html.append('      </tr>')
        html.append('    </table>')

        # Error samples
        if errors:
            html.append(f'    <h3>Error Samples ({len(errors)} misclassified)</h3>')
            html.append('    <table class="error-table">')
            header = '      <tr><th>#</th><th>Text</th><th>True Label</th><th>Pred Label</th>'
            for lbl in label_names:
                header += f'<th>P({lbl})</th>'
            header += '</tr>'
            html.append(header)
            for i, e in enumerate(errors):
                html.append('      <tr>')
                html.append(f'        <td>{i+1}</td>')
                html.append(f'        <td>{e["text"]}</td>')
                html.append(f'        <td>{e["true_label"]}</td>')
                html.append(f'        <td>{e["pred_label"]}</td>')
                for lbl in label_names:
                    p = e['probs'].get(lbl, 0.0)
                    cell_class = ' class="metric-bad"' if lbl == e['pred_label'] and lbl != e['true_label'] else ''
                    html.append(f'        <td{cell_class}>{p:.4f}</td>')
                html.append('      </tr>')
            html.append('    </table>')
        else:
            html.append('    <p style="color: #2e7d32;">No misclassified samples!</p>')
        html.append('  </div>')

    # Summary table
    html.append('  <div class="summary">')
    html.append('    <h2>Summary (All Files)</h2>')
    html.append('    <table>')
    html.append('      <tr><th>File</th><th>Samples</th><th>Accuracy</th><th>Error Rate</th><th>F1 (Macro)</th>')
    if recall_label:
        html.append(f'      <th>Recall ({recall_label})</th>')
    # Add F1 for second label column if available
    has_second_label_f1 = any(r['metrics'].get('f1_second_label') is not None for r in results)
    if has_second_label_f1:
        html.append(f'      <th>F1 (Second Label)</th>')
    html.append('      </tr>')
    for r in results:
        m = r['metrics']
        fname = os.path.basename(r['path'])
        ns = r['num_samples']
        html.append('      <tr>')
        html.append(f'        <td>{fname}</td>')
        html.append(f'        <td>{ns}</td>')
        html.append(f'        <td>{m["accuracy"]:.4f}</td>')
        html.append(f'        <td>{m["error_rate"]:.4f}</td>')
        html.append(f'        <td>{m["f1_macro"]:.4f}</td>')
        if recall_label:
            rec = m['per_class'].get(recall_label, {}).get('recall', '-')
            if isinstance(rec, float):
                html.append(f'        <td>{rec:.4f}</td>')
            else:
                html.append(f'        <td>{rec}</td>')
        if has_second_label_f1:
            f1_val = m.get('f1_second_label')
            if f1_val is not None:
                html.append(f'        <td>{f1_val:.4f}</td>')
            else:
                html.append(f'        <td>-</td>')
        html.append('      </tr>')
    
    # Weighted average row (by sample count)
    total_samples = sum(r['num_samples'] for r in results)
    if total_samples > 0:
        w_acc = sum(r['metrics']['accuracy'] * r['num_samples'] for r in results) / total_samples
        w_err = 1.0 - w_acc
        w_f1 = sum(r['metrics']['f1_macro'] * r['num_samples'] for r in results) / total_samples
        
        html.append('      <tr style="font-weight: bold; background-color: #e8f5e9;">')
        html.append(f'        <td>Weighted Avg</td>')
        html.append(f'        <td>{total_samples}</td>')
        html.append(f'        <td>{w_acc:.4f}</td>')
        html.append(f'        <td>{w_err:.4f}</td>')
        html.append(f'        <td>{w_f1:.4f}</td>')
        
        if recall_label:
            w_rec = 0.0
            for r in results:
                rec_val = r['metrics']['per_class'].get(recall_label, {}).get('recall', 0.0)
                if isinstance(rec_val, float):
                    w_rec += rec_val * r['num_samples']
            w_rec /= total_samples
            html.append(f'        <td>{w_rec:.4f}</td>')
        
        if has_second_label_f1:
            w_f1_s = 0.0
            for r in results:
                f1_val = r['metrics'].get('f1_second_label')
                if f1_val is not None:
                    w_f1_s += f1_val * r['num_samples']
            w_f1_s /= total_samples
            html.append(f'        <td>{w_f1_s:.4f}</td>')
        
        html.append('      </tr>')
    html.append('    </table>')
    html.append('  </div>')
    html.append('</body>')
    html.append('</html>')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))
    print(f"HTML report saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.isdir(args.eval_model_dir):
        raise SystemExit(f"Model directory not found: {args.eval_model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.eval_model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.eval_model_dir, local_files_only=True
    )
    model.to(device)
    model.eval()

    config = model.config
    label2id = config.label2id
    id2label = config.id2label

    if args.recall_label and args.recall_label not in label2id:
        raise SystemExit(
            f"--recall_label '{args.recall_label}' not found in model labels: {list(label2id.keys())}"
        )

    results = []
    for csv_path in args.test_csv:
        if not os.path.isfile(csv_path):
            print(f"Warning: file not found, skipping: {csv_path}")
            continue
        print(f"Evaluating: {csv_path}")
        metrics, errors, num_samples = evaluate_file(
            model, tokenizer, device, csv_path,
            args.text_column, args.label_column,
            label2id, id2label,
            args.max_length, args.batch_size
        )
        results.append({
            'path': csv_path,
            'metrics': metrics,
            'errors': errors,
            'id2label': id2label,
            'num_samples': num_samples,
        })
        rl = args.recall_label
        rec = metrics['per_class'].get(rl, {}).get('recall', 0.0) if rl else metrics['f1_macro']
        rec_label = rl if rl else 'macro'
        
        # Add F1 for second label in output
        f1_second = ""
        if metrics.get('f1_second_label') is not None and metrics.get('second_label_name'):
            f1_second = f", F1({metrics['second_label_name']})={metrics['f1_second_label']:.4f}"
        
        print(f"  Accuracy={metrics['accuracy']:.4f}, Recall({rec_label})={rec:.4f}, "
              f"F1(macro)={metrics['f1_macro']:.4f}{f1_second}, Errors={len(errors)}")

    if results:
        generate_html_report(results, args.output_html, args.recall_label)
    else:
        print("No valid test CSV files found.")


if __name__ == '__main__':
    main()
