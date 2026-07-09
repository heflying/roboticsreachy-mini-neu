#!/usr/bin/env python3
r"""
train_lora.py

Usage examples:
  .venv\Scripts\activate
  pip install -U torch transformers datasets peft accelerate evaluate

  python scripts\train_lora.py --train_csv data\sample.csv --output_dir output_lora \
    --model_dir chinese-bert-wwm-ext --epochs 3 --batch_size 8

Single-file, CPU-friendly LoRA fine-tuning for text classification.
Assumes CSVs have header columns: text,label (adjust via --text_column/--label_column).
"""
import argparse
import os
from typing import Dict, Any, List

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
# Prefer the installed Hugging Face `evaluate` package even if a local scripts/evaluate.py exists.
# This helper temporarily removes repo and scripts paths from sys.path to avoid shadowing.
import importlib, sys, os

def _import_hf_evaluate():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    scripts_dir = os.path.join(repo_root, 'scripts')
    orig_sys_path = list(sys.path)
    try:
        sys.path = [p for p in sys.path if p and os.path.abspath(p) not in (repo_root, scripts_dir)]
        pkg = importlib.import_module('evaluate')
    finally:
        sys.path[:] = orig_sys_path
    return pkg

evaluate = _import_hf_evaluate()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=False, nargs="+", help="Path(s) to training CSV file(s)")
    p.add_argument("--val_csv", required=False, nargs="*", default=[], help="Path(s) to validation CSV file(s)")
    p.add_argument("--from_model_dir", default="chinese-bert-wwm-ext", help="Local pretrained model dir")
    p.add_argument("--output_model_dir", default="", help="Where to save LoRA adapter + config")
    p.add_argument("--eval_model_dir", default="", help="Where is the evaluation model")
    p.add_argument("--text_column", default="text")
    p.add_argument("--label_column", default="label")
    p.add_argument("--label_list", required=True, default="", help="Comma-separated fixed label list (e.g. 'no_privacy,privacy'). If provided, labels are not inferred from data.")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_every_epochs", type=int, default=10, help="Save merged checkpoint every N epochs (default: 10)")
    p.add_argument("--eval_every_epochs", type=int, default=1, help="Run validation every N epochs and save best model (default: 1)")
    p.add_argument("--no_train_head", action="store_false", dest="train_head", help="Disable training classification head (default: enabled)")
    p.add_argument("--no_class_weight", action="store_true", default=False,
                   help="Disable inverse-frequency class weights in CrossEntropyLoss (default: enabled)")
    p.add_argument("--do_train", action="store_true", default=False)
    p.add_argument("--do_eval", action="store_true", default=False)
    p.add_argument("--logging_dir", default=None,
                   help="TensorBoard logging directory (default: output_model_dir/runs)")
    p.add_argument("--logging_steps", type=int, default=50,
                   help="Log metrics every N steps (default: 50)")
    return p.parse_args()


def make_label_mapping(train_ds, label_col: str, label_list_str: str = ""):
    # If a fixed label list is provided (comma-separated), use it in the given order.
    if label_list_str:
        labels = [l.strip() for l in label_list_str.split(",") if l.strip()]
    else:
        labels = train_ds.unique(label_col)
        labels = sorted(list(set(labels)), key=lambda x: str(x))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label


def compute_class_weights(dataset, label_col: str, num_labels: int):
    """Compute inverse-frequency class weights for imbalanced datasets.
    Weight_i = N / (C * count_i) where N=total samples, C=num classes, count_i=samples in class i.
    Returns a torch.Tensor of shape (num_labels,).
    """
    from collections import Counter
    # Ensure labels are integers (Dataset may keep string type after map)
    labels = [int(l) for l in dataset[label_col]]
    counts = Counter(labels)
    total = len(labels)
    weights = []
    for i in range(num_labels):
        c = counts.get(i, 0)
        if c == 0:
            weights.append(0.0)
        else:
            weights.append(total / (num_labels * c))
    w = torch.tensor(weights, dtype=torch.float32)
    print(f"Class weights (inverse frequency): {w.tolist()}")
    return w


class WeightedLossTrainer(Trainer):
    """Trainer subclass that uses class-weighted CrossEntropyLoss."""

    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        if class_weights is not None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.class_weights = class_weights.to(device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


class SaveEveryNEpochsCallback(TrainerCallback):
    """Custom callback to save merged model every N epochs."""
    
    def __init__(self, save_every_epochs=10, output_dir=".", config=None,
                 tokenizer=None, model=None):
        self.save_every_epochs = save_every_epochs
        self.output_dir = output_dir
        self.config = config
        self.tokenizer = tokenizer
        self.model = model  # PEFT model (will be updated via self.trainer.model)
    

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_idx = int(state.epoch)
        if self.save_every_epochs > 0 and (epoch_idx % self.save_every_epochs == 0 or epoch_idx == args.num_train_epochs):
            # Get the LIVE model that Trainer is actually training
            live_model = kwargs.get('model', None)
            if live_model is None and hasattr(self, 'trainer') and self.trainer is not None:
                live_model = self.trainer.model
            if live_model is None:
                live_model = self.model  # fallback
            else:
                self.model = live_model  # update reference for next epoch

            print(f"\n[SAVE] Saving merged model at epoch {epoch_idx}")

            import os, shutil

            ckpt_dir = os.path.join(self.output_dir, f"epoch_{epoch_idx}")
            os.makedirs(ckpt_dir, exist_ok=True)

            try:
                # Merge and save
                import copy
                try:
                    live_model_clone = copy.deepcopy(live_model)
                    merged_model = live_model_clone.merge_and_unload()
                except Exception:
                    # Fallback: merge directly (modifies live_model)
                    merged_model = live_model.merge_and_unload()
                
                merged_model.save_pretrained(ckpt_dir)
                self.config.save_pretrained(ckpt_dir)
                if self.tokenizer:
                    self.tokenizer.save_pretrained(ckpt_dir)

                print(f"  Saved merged model to: {ckpt_dir}")

                # Update latest pointer
                latest_dir = os.path.join(self.output_dir, 'latest')
                if os.path.exists(latest_dir):
                    shutil.rmtree(latest_dir)
                shutil.copytree(ckpt_dir, latest_dir)
                print(f"  Updated latest: {latest_dir}")

            except Exception as e:
                print(f"  [WARN] Failed to save merged model: {e}")
                import traceback; traceback.print_exc()

        return control


class EvalEveryNEpochsCallback(TrainerCallback):
    """After each epoch eval (triggered by evaluation_strategy="epoch"),
    process results every N epochs and save best model info to best/ directory.
    """
    def __init__(self, eval_every_epochs=1, output_dir=".", label2id=None):
        self.eval_every_epochs = eval_every_epochs
        self.output_dir = output_dir
        self.label2id = label2id or {}
        self.best_f1 = -1.0
        self.best_epoch = -1

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_idx = int(state.epoch)
        # Only process eval every N epochs
        if self.eval_every_epochs <= 0 or (epoch_idx % self.eval_every_epochs != 0):
            return control

        # Read latest eval results from state.log_history
        if not state.log_history:
            return control
        latest = None
        for entry in reversed(state.log_history):
            if "eval_f1_macro" in entry:
                latest = entry
                break
        if latest is None:
            return control

        f1 = latest.get("eval_f1_macro", -1.0)
        acc = latest.get("eval_accuracy", 0.0)
        print(f"\n{'='*60}")
        print(f"📊 Epoch {epoch_idx} eval: accuracy={acc:.4f}, f1_macro={f1:.4f}")
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.best_epoch = epoch_idx
            print(f"  ✅ New best F1: {f1:.4f} (epoch {epoch_idx})")
            self._save_best(latest)
        else:
            print(f"  F1: {f1:.4f} (best: {self.best_f1:.4f} at epoch {self.best_epoch})")
        print(f"{'='*60}\n")
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.best_epoch >= 0:
            print(f"\n🏆 Training finished. Best F1: {self.best_f1:.4f} at epoch {self.best_epoch}")
            print(f"   Best score saved in: {os.path.join(self.output_dir, 'best')}")
        return control

    def _save_best(self, metrics):
        import json
        best_dir = os.path.join(self.output_dir, "best")
        os.makedirs(best_dir, exist_ok=True)
        score_info = {
            "best_epoch": int(self.best_epoch),
            "best_f1_macro": float(self.best_f1),
            "metrics": {k.replace("eval_", ""): float(v) for k, v in metrics.items()
                        if isinstance(v, (int, float))}
        }
        with open(os.path.join(best_dir, "score.txt"), "w", encoding="utf-8") as f:
            for k, v in score_info.items():
                f.write(f"{k}: {v}\n")
        with open(os.path.join(best_dir, "score.json"), "w", encoding="utf-8") as f:
            json.dump(score_info, f, indent=2, ensure_ascii=False)
        print(f"  💾 Best model info saved to: {best_dir}")


def tokenize_function(examples, tokenizer, text_col, max_length):
    return tokenizer(examples[text_col], truncation=True, max_length=max_length)


import numpy as _np

def _macro_f1(preds, labels, labels_list=None):
    # simple macro-F1 implementation
    preds = _np.array(preds, dtype=int)
    labels = _np.array(labels, dtype=int)
    if labels_list is None:
        classes = _np.unique(_np.concatenate([preds, labels]))
    else:
        classes = _np.array(list(range(len(labels_list))))
    f1s = []
    for c in classes:
        tp = int(_np.sum((preds == c) & (labels == c)))
        fp = int(_np.sum((preds == c) & (labels != c)))
        fn = int(_np.sum((preds != c) & (labels == c)))
        if tp + fp == 0 or tp + fn == 0:
            f1 = 0.0
        else:
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(_np.mean(f1s))


def compute_metrics_fn(label_map):
    def compute(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        labels = _np.array(labels)
        preds = _np.array(preds)
        acc = float((_np.array(preds) == _np.array(labels)).mean())
        try:
            f1_macro = _macro_f1(preds, labels, labels_list=list(label_map.keys()))
        except Exception:
            f1_macro = _macro_f1(preds, labels)
        return {"accuracy": acc, "f1_macro": f1_macro}
    return compute


def unfreeze_classification_head(model):
    # Target common classification-head modules by name, avoiding broad matches.
    matched = []
    for name, param in model.named_parameters():
        lname = name.lower()
        # Positive matches for heads
        if ("classifier" in lname or ".pooler." in lname or ".cls." in lname or "classification" in lname or "out_proj" in lname):
            # Exclude MLM/prediction heads and unrelated modules
            if any(x in lname for x in ("predictions", "seq_relationship", "predictions.transform")):
                continue
            param.requires_grad = True
            matched.append(name)
    if not matched:
        # Fallback: unfreeze classifier if exists as attribute
        try:
            if hasattr(model, 'classifier'):
                for name, param in model.classifier.named_parameters():
                    param.requires_grad = True
                    matched.append(f"classifier.{name}")
        except Exception:
            pass
    if not matched:
        # Final fallback: unfreeze the last parameter
        all_params = list(model.named_parameters())
        if all_params:
            name, param = all_params[-1]
            param.requires_grad = True
            matched.append(name)
    print("Enabled training for classification head params:", matched)


def merge_and_save_base_final(args, config, tokenizer, peft_model=None):
    """Merge LoRA adapter into args.output_model_dir/latest.
    The merged model will be saved directly to args.output_model_dir/latest.
    If peft_model is provided, perform in-memory merge (preferred)."""
    try:
        merged_dir = os.path.join(args.output_model_dir, 'latest')
        os.makedirs(merged_dir, exist_ok=True)
        # In-memory merge when a PeftModel instance is provided
        if peft_model is not None:
            if hasattr(peft_model, 'merge_and_unload'):
                merged_res = peft_model.merge_and_unload()
                merged_model = merged_res if merged_res is not None else peft_model
            else:
                raise RuntimeError('In-memory merge not supported by this PEFT version')
            merged_model.save_pretrained(merged_dir)
            config.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)
            print(f"Merged LoRA adapter and base model saved to: {merged_dir}")
        else:
            # Fallback: disk-based merge (older PEFT behavior)
            from peft import PeftModel
            base_out = os.path.join(args.output_model_dir, "base_with_head")
            base_model = AutoModelForSequenceClassification.from_pretrained(base_out, local_files_only=True, config=config)
            peft_loaded = PeftModel.from_pretrained(base_model, args.output_model_dir, local_files_only=True)
            if hasattr(peft_loaded, 'merge_and_unload'):
                merged_model = peft_loaded.merge_and_unload()
            else:
                peft_loaded.merge_and_unload()
                merged_model = base_model
            merged_model.save_pretrained(merged_dir)
            config.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)
            print(f"Merged LoRA adapter and base model saved to: {merged_dir}")
    except Exception as e:
        print(f"Warning: failed to merge adapter into base model: {e}")


def copy_model_to_origin(model_dir, output_dir):
    """Copy initial model files into output_dir/origin and return the path to load from.
    Ensures output/origin is empty before copying. Returns model_dir on failure."""
    import shutil
    origin_dir = os.path.join(output_dir, 'origin')
    try:
        if os.path.exists(origin_dir):
            shutil.rmtree(origin_dir)
        os.makedirs(origin_dir, exist_ok=True)
        if os.path.isdir(model_dir):
            for name in os.listdir(model_dir):
                # avoid copying the origin directory into itself
                if name == os.path.basename(origin_dir):
                    continue
                src = os.path.join(model_dir, name)
                dst = os.path.join(origin_dir, name)
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
            print(f"Copied initial model from {model_dir} to {origin_dir}; loading from origin.")
            return origin_dir
        else:
            print(f"Warning: model_dir {model_dir} is not a directory; skipping copy and loading directly from model_dir")
            return model_dir
    except Exception as e:
        print(f"Warning: failed to copy initial model to origin: {e}. Falling back to loading from {model_dir}")
        return model_dir


def train(args):
    """Train the model with LoRA fine-tuning."""
    torch.manual_seed(args.seed)
    load_model_dir = copy_model_to_origin(args.from_model_dir, args.output_model_dir)

    # Load CSVs (support multiple files per split)
    data_files = {}
    has_train = args.train_csv is not None and len(args.train_csv) > 0
    if has_train:
        data_files["train"] = args.train_csv
    if args.val_csv:
        data_files["validation"] = args.val_csv
    
    if not data_files:
        raise ValueError("No data files provided. Please provide --train_csv and/or --val_csv")
    
    raw = load_dataset("csv", data_files=data_files)

    # Create label mapping (can be fixed via --label_list)
    if has_train:
        label2id, id2label = make_label_mapping(raw["train"], args.label_column, args.label_list)
    else:
        # No training data: use label_list to create mapping
        if not args.label_list:
            raise ValueError("Cannot determine label mapping without --train_csv or --label_list")
        labels = [l.strip() for l in args.label_list.split(",") if l.strip()]
        label2id = {l: i for i, l in enumerate(labels)}
        id2label = {i: l for l, i in enumerate(labels)}
    
    num_labels = len(label2id)
    print("Label mapping (label -> id):", label2id)

    # Load tokenizer and config
    tokenizer = AutoTokenizer.from_pretrained(load_model_dir, use_fast=True, local_files_only=True)
    config = AutoConfig.from_pretrained(load_model_dir, local_files_only=True, num_labels=num_labels,
                                        id2label=id2label, label2id=label2id)
    model = AutoModelForSequenceClassification.from_pretrained(load_model_dir, config=config, local_files_only=True)

    # Map string labels to ints if necessary
    def map_labels(example):
        lab = example[args.label_column]
        if isinstance(lab, str):
            if lab in label2id:
                example[args.label_column] = label2id[lab]
            else:
                raise ValueError(f"Label '{lab}' not found in fixed label list: {list(label2id.keys())}")
        else:
            example[args.label_column] = int(lab)
        return example

    raw = raw.map(map_labels)

    # Tokenize
    remove_cols = [c for c in raw["train"].column_names if c not in [args.text_column, args.label_column]]
    tokenized = raw.map(lambda ex: tokenize_function(ex, tokenizer, args.text_column, args.max_length),
                        batched=True, remove_columns=remove_cols)

    # Rename label column to 'labels' expected by Trainer/DataCollator
    for split in tokenized.keys():
        if args.label_column in tokenized[split].column_names:
            tokenized[split] = tokenized[split].rename_column(args.label_column, "labels")

    # Convert labels to ints explicitly (some CSV loads keep them as strings)
    def _to_int_labels(batch):
        if 'labels' in batch:
            batch['labels'] = [int(x) for x in batch['labels']]
        return batch
    tokenized = tokenized.map(_to_int_labels, batched=True)

    # Debug: inspect tokenized features
    print("Tokenized columns:")
    for split in tokenized.keys():
        print(f"  {split}: {tokenized[split].column_names}")
        if len(tokenized[split]) > 0:
            sample = tokenized[split][0]
            for k, v in sample.items():
                print(f"    {k}: type={type(v)}, sample={str(v)[:80]}")

    # Custom collator to ensure labels are tensors of dtype long and to control padding/truncation
    def data_collator(features):
        labels = [int(f.pop('labels')) for f in features]
        batch = tokenizer.pad(features, padding='max_length', max_length=args.max_length, return_tensors='pt')
        batch['labels'] = torch.tensor(labels, dtype=torch.long)
        return batch

    # LoRA config (standard defaults)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS",
    )

    model = get_peft_model(model, lora_config)

    # Optionally enable classification head training (keep adapters trainable)
    if getattr(args, "train_head", True):
        unfreeze_classification_head(model)

    # Determine if validation data exists (needed before TrainingArguments)
    has_val = tokenized is not None and "validation" in tokenized

    # Training args
    import shutil
    # evaluation_strategy: enable epoch-level eval if user wants periodic evaluation
    eval_strategy = "epoch" if (args.eval_every_epochs > 0 and has_val) else "no"
    # TensorBoard logging directory (default: output_model_dir/runs/)
    logging_dir = args.logging_dir or os.path.join(args.output_model_dir, "runs")
    ta_kwargs = dict(
        output_dir=args.output_model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy=eval_strategy,
        save_strategy="no",
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        seed=args.seed,
        load_best_model_at_end=False,
        fp16=False,
        dataloader_drop_last=False,
        report_to=["tensorboard"],
        logging_dir=logging_dir,
    )
    # Filter kwargs to only those supported by this transformers version
    try:
        from inspect import signature
        sig = signature(TrainingArguments.__init__)
        valid_keys = set(sig.parameters.keys())
    except Exception:
        valid_keys = set(ta_kwargs.keys())
    filtered = {k: v for k, v in ta_kwargs.items() if k in valid_keys}
    if len(filtered) != len(ta_kwargs):
        missing = set(ta_kwargs.keys()) - set(filtered.keys())
        print(f"Note: TrainingArguments does not support parameters: {missing}; they will be skipped for compatibility.")
    training_args = TrainingArguments(**filtered)

    # Compute class weights if requested
    class_weights = None
    if not args.no_class_weight:
        class_weights = compute_class_weights(raw["train"], args.label_column, num_labels)


    # Setup Trainer
    # compute_metrics is needed whenever we have validation data (for eval callback)
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"] if "train" in tokenized else None,
        eval_dataset=tokenized.get("validation", None) if has_val else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn(label2id) if has_val else None,
        remove_unused_columns=False,
    )
    try:
        from inspect import signature
        sig = signature(Trainer.__init__)
        valid = set(sig.parameters.keys())
    except Exception:
        valid = set(trainer_kwargs.keys())
    filtered_trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if k in valid}
    removed = set(trainer_kwargs.keys()) - set(filtered_trainer_kwargs.keys())
    if removed:
        print(f"Note: Trainer does not support parameters: {removed}; they will be skipped for compatibility.")
    trainer = WeightedLossTrainer(class_weights=class_weights, **filtered_trainer_kwargs)

    # Add custom callback for per-N-epoch merged checkpoint saving
    trainer.add_callback(SaveEveryNEpochsCallback(
        save_every_epochs=args.save_every_epochs,
        output_dir=args.output_model_dir,
        config=config,
        tokenizer=tokenizer,
        model=model,
    ))

    # Add eval callback (reads eval results every epoch, saves best score every N epochs)
    if args.eval_every_epochs > 0 and has_val:
        eval_callback = EvalEveryNEpochsCallback(
            eval_every_epochs=args.eval_every_epochs,
            output_dir=args.output_model_dir,
            label2id=label2id,
        )
        trainer.add_callback(eval_callback)

    # Train
    trainer.train()

    # After training, merge and save final model
    try:
        merge_and_save_base_final(args, config, tokenizer, peft_model=model)
    except Exception as e:
        print(f"Warning: final merge failed: {e}")


def evaluate(args):
    """Evaluate the merged model on validation data."""
    merged_dir = args.eval_model_dir
    if not os.path.isdir(merged_dir):
        raise ValueError(f"Merged model not found at: {merged_dir}")

    # Load label mapping from saved model config
    config = AutoConfig.from_pretrained(merged_dir, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(merged_dir, use_fast=True, local_files_only=True)
    label2id = config.label2id
    id2label = config.id2label
    print("Loaded label mapping from merged model:", label2id)
    print(f"  num_labels={config.num_labels}, id2label={id2label}")

    # Load and process validation data
    if not args.val_csv:
        print("Warning: No validation CSV provided for evaluation.")
        return

    val_files = {"validation": args.val_csv}
    val_raw = load_dataset("csv", data_files=val_files)

    # Map labels
    def map_labels(example):
        lab = example[args.label_column]
        if isinstance(lab, str):
            if lab in label2id:
                example[args.label_column] = label2id[lab]
            else:
                raise ValueError(f"Label '{lab}' not found in label list: {list(label2id.keys())}")
        else:
            example[args.label_column] = int(lab)
        return example
    val_raw = val_raw.map(map_labels)

    # Tokenize
    remove_cols = [c for c in val_raw["validation"].column_names if c not in [args.text_column, args.label_column]]
    tokenized = val_raw.map(lambda ex: tokenize_function(ex, tokenizer, args.text_column, args.max_length),
                            batched=True, remove_columns=remove_cols)

    if args.label_column in tokenized["validation"].column_names:
        tokenized["validation"] = tokenized["validation"].rename_column(args.label_column, "labels")

    def _to_int_labels(batch):
        if 'labels' in batch:
            batch['labels'] = [int(x) for x in batch['labels']]
        return batch
    tokenized = tokenized.map(_to_int_labels, batched=True)

    print("Eval tokenized columns:", tokenized["validation"].column_names)

    # Evaluate using merged model
    val_ds = tokenized["validation"]

    # Use same data collator as training (pad to max_length for consistency)
    def eval_data_collator(features):
        labels = [int(f.pop('labels')) for f in features]
        batch = tokenizer.pad(features, padding='max_length', max_length=args.max_length, return_tensors='pt')
        batch['labels'] = torch.tensor(labels, dtype=torch.long)
        return batch

    try:
        from transformers import Trainer as EvalTrainer
        temp_model = AutoModelForSequenceClassification.from_pretrained(
            merged_dir, config=config, local_files_only=True
        )
        # output_dir 是 TrainingArguments 的必填参数，此处仅做评估（do_train=False），
        # 不会写入任何文件，直接用 merged_dir 即可。
        temp_args = TrainingArguments(
            output_dir=merged_dir,
            per_device_eval_batch_size=args.batch_size,
            do_train=False,
            report_to=[],
        )
        temp_trainer = EvalTrainer(
            model=temp_model,
            args=temp_args,
            data_collator=eval_data_collator,
            compute_metrics=compute_metrics_fn(label2id),
        )
        metrics = temp_trainer.evaluate(val_ds)
        print("Eval metrics (merged model):", metrics)
    except Exception as e:
        print(f"Warning: merged-model evaluation failed: {e}")


def main():
    args = parse_args()
    
    if args.do_train:
        train(args)
    
    if args.do_eval:
        evaluate(args)

if __name__ == "__main__":
    main()

