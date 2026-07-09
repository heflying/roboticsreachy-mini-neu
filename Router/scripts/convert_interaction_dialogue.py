"""
scripts/convert_interaction_dialogue.py

Convert Interaction_Dialogue_with_Privacy dataset to training CSV format.

Pipeline:
  1. Read privacy_annotation_train_zh.json and privacy_annotation_test_zh.json
  2. Extract each user/assistant utterance as a separate row
  3. Label: check if any privacy phrase falls within the speaker's text → privacy;
     fallback: if phrase matching fails but privacy array is non-empty, label both as privacy
  4. Deduplicate by text
  5. Output train.csv (Chinese, no translation needed)

Usage:
    python scripts/convert_interaction_dialogue.py [--input PATH ...] [--output PATH]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)


def load_json(path: str) -> list[dict]:
    """Load JSON file and return list of conversation records."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data)}")
    return data


def extract_samples(records: list[dict], include_assistant: bool = False) -> list[tuple[str, str]]:
    """Extract (text, label) samples from conversation records.

    Each user/assistant utterance becomes a separate row.
    Privacy labels are determined by checking if any privacy phrase
    appears in the speaker's text. Fallback: if phrase matching fails
    but privacy array is non-empty, label both speakers as privacy.
    """
    samples = []

    for rec in records:
        conv = rec.get("conversation", [])
        for turn in conv:
            user_text = (turn.get("user") or "").strip()
            asst_text = (turn.get("assistant") or "").strip()
            privacy_list = turn.get("privacy", [])

            if not user_text and not asst_text:
                continue

            if not privacy_list:
                # No privacy annotations — both no_privacy
                if user_text:
                    samples.append((user_text, "no_privacy"))
                if asst_text and include_assistant:
                    samples.append((asst_text, "no_privacy"))
                continue

            # Check which phrases belong to which speaker
            user_has_privacy = False
            asst_has_privacy = False
            matched_any = False

            for p in privacy_list:
                phrase = (p.get("phrase") or "").strip()
                if not phrase:
                    continue
                in_user = phrase in user_text
                in_asst = phrase in asst_text
                if in_user:
                    user_has_privacy = True
                    matched_any = True
                if in_asst:
                    asst_has_privacy = True
                    matched_any = True

            # Fallback: if no phrase matched either speaker but privacy list is non-empty,
            # conservatively label both as privacy
            if not matched_any:
                user_has_privacy = True
                asst_has_privacy = True

            if user_text:
                label = "privacy" if user_has_privacy else "no_privacy"
                samples.append((user_text, label))
            if asst_text and include_assistant:
                label = "privacy" if asst_has_privacy else "no_privacy"
                samples.append((asst_text, label))

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Convert Interaction_Dialogue_with_Privacy to train.csv"
    )
    parser.add_argument(
        "--input", nargs="+", default=None,
        help="Input JSON file(s). Defaults to train+test in dataset directory."
    )
    parser.add_argument(
        "--output", default=None,
        help="Output CSV path (auto-detected if omitted)"
    )
    parser.add_argument("--min_length", type=int, default=5,
                        help="Minimum character length for an utterance (default: 5)")
    parser.add_argument("--include_assistant", action="store_true", default=False,
                        help="Include assistant utterances in output (default: only user)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    dataset_dir = os.path.join(
        _project_root, "data", "privacy", "datasets", "Interaction_Dialogue_with_Privacy"
    )

    # Default inputs: merge train and test
    if args.input:
        input_paths = args.input
    else:
        input_paths = [
            os.path.join(dataset_dir, "privacy_annotation_train_zh.json"),
            os.path.join(dataset_dir, "privacy_annotation_test_zh.json"),
        ]

    output_path = args.output or os.path.join(dataset_dir, "train.csv")

    # Load all records
    all_records = []
    for path in input_paths:
        if not os.path.exists(path):
            logger.warning("Input not found, skipping: %s", path)
            continue
        records = load_json(path)
        logger.info("Loaded %d records from %s", len(records), os.path.basename(path))
        all_records.extend(records)

    if not all_records:
        logger.error("No records loaded")
        return 1

    # Extract samples
    samples = extract_samples(all_records, include_assistant=args.include_assistant)
    logger.info("Extracted %d utterances before dedup", len(samples))

    # Deduplicate by text and filter by min length
    seen = set()
    deduped = []
    for text, label in samples:
        if len(text) < args.min_length:
            continue
        if text in seen:
            continue
        seen.add(text)
        deduped.append((text, label))

    priv = sum(1 for _, l in deduped if l == "privacy")
    nop = sum(1 for _, l in deduped if l == "no_privacy")
    logger.info("After dedup: %d samples (privacy=%d, no_privacy=%d)", len(deduped), priv, nop)

    # Sort by label order: privacy first, then no_privacy
    label_order = {"privacy": 0, "no_privacy": 1}
    deduped.sort(key=lambda x: label_order.get(x[1], 99))

    # Write CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        for text, label in deduped:
            writer.writerow([text, label])

    logger.info("Done. Output: %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
