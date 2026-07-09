"""
scripts/convert_privacy_care.py

Convert privacy-care-interactions dataset to training CSV format.

Pipeline:
  1. Read unsplit-train-en.jsonl
  2. Split by CW:/CR: speaker turns
  3. Deduplicate turns
  4. Label: category==2 → privacy, else → no_privacy
  5. Translate to Chinese via LLM (skip if no Chinese chars)
  6. Output train.csv

Usage:
    python scripts/convert_privacy_care.py [--llm_backend ollama] [--output PATH]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys

# Ensure project root is on sys.path for llm_client
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.llm_client import create_client

logger = logging.getLogger(__name__)

# Regex to split by CW: or CR: prefixes
SPEAKER_PREFIX_RE = re.compile(r"(?:CW|CR)\s*[:：]")

# Default translation prompt template
DEFAULT_TRANSLATE_PROMPT = (
    "请将以下英文翻译为自然流畅的中文，只输出翻译结果，不要添加解释或标注：\n\n{text}"
)


def split_speaker_turns(text: str) -> list[str]:
    """Split dialog text by CW:/CR: prefixes and return individual turns.

    Each occurrence of CW: or CR: starts a new turn. The text before the
    first prefix (if any) is discarded.
    """
    # Find all positions of speaker prefixes
    positions = []
    for m in SPEAKER_PREFIX_RE.finditer(text):
        positions.append((m.start(), m.end()))

    if not positions:
        return []

    turns = []
    for i, (content_start_char, prefix_end) in enumerate(positions):
        # The turn content starts after the prefix
        turn_start = prefix_end
        # The turn content ends where the next prefix starts, or end of text
        if i + 1 < len(positions):
            turn_end = positions[i + 1][0]
        else:
            turn_end = len(text)
        turn_text = text[turn_start:turn_end].strip()
        if turn_text:
            turns.append(turn_text)

    return turns


def has_chinese(text: str) -> bool:
    """Check if text contains at least one Chinese character."""
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL file and return list of parsed dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON line %d", line_no)
    return records


def main():
    parser = argparse.ArgumentParser(description="Convert privacy-care-interactions to train.csv")
    parser.add_argument("--input", default=None,
                        help="Path to unsplit-train-en.jsonl (auto-detected if omitted)")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (auto-detected if omitted)")
    parser.add_argument("--llm_backend", default="ollama", choices=["ollama", "qwen", "spark"],
                        help="LLM backend for translation (default: ollama)")
    parser.add_argument("--translate_prompt", default=None,
                        help="Custom translation prompt template (use {text} as placeholder)")
    parser.add_argument("--skip_translate", action="store_true",
                        help="Skip translation step (output English text)")
    parser.add_argument("--min_length", type=int, default=5,
                        help="Minimum character length for a sample (default: 5)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Resolve paths
    dataset_dir = os.path.join(_project_root, "data", "privacy", "datasets", "privacy-care-interactions")
    input_path = args.input or os.path.join(dataset_dir, "unsplit-train-en.jsonl")
    output_path = args.output or os.path.join(dataset_dir, "train.csv")

    if not os.path.exists(input_path):
        logger.error("Input not found: %s", input_path)
        return 1

    # Load and process
    records = load_jsonl(input_path)
    logger.info("Loaded %d records from %s", len(records), input_path)

    # Step 1-3: Split turns, label, dedupe
    seen = set()
    samples = []  # list of (text, label)
    for rec in records:
        text = rec.get("text", "")
        category = rec.get("category", -1)
        label = "privacy" if category == 2 else "no_privacy"

        turns = split_speaker_turns(text)
        for turn in turns:
            # Normalize whitespace for dedup
            normalized = re.sub(r"\s+", " ", turn).strip()
            if normalized in seen:
                continue
            seen.add(normalized)
            samples.append((normalized, label))

    logger.info("Extracted %d unique speaker turns", len(samples))

    # Step 4: Translate (if not skipped)
    translate_prompt = args.translate_prompt or DEFAULT_TRANSLATE_PROMPT

    if not args.skip_translate:
        client = create_client(args.llm_backend)
        logger.info("Translating with backend=%s, model=%s", args.llm_backend, client.model)

    # Always overwrite output
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])

        for i, (text, label) in enumerate(samples):
            if args.skip_translate:
                if len(text) < args.min_length:
                    continue
                final_text = text
            else:
                prompt = translate_prompt.format(text=text)
                try:
                    translated = client.generate(prompt)
                    if has_chinese(translated):
                        if len(translated) < args.min_length:
                            logger.warning("Row %d: Translated text too short (%d chars), skipping: %s", i + 1, len(translated), translated[:80])
                            continue
                        final_text = translated
                    else:
                        logger.warning("Row %d: Translation has no Chinese chars, skipping: %s", i + 1, translated[:80])
                        continue
                except Exception as e:
                    logger.error("Row %d: Translation failed: %s", i + 1, e)
                    continue

            writer.writerow([final_text, label])
            f.flush()

            if (i + 1) % 10 == 0 or i + 1 == len(samples):
                logger.info("Progress: %d/%d", i + 1, len(samples))

    logger.info("Done. Output: %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
