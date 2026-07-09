"""
scripts/review_labels.py

Interactive CLI tool for reviewing and correcting labels in a CSV file.

Reads a CSV with columns (text, label), presents entries grouped by label
(two passes: first label found in file, then the other), and allows the
user to flip incorrect labels. Edits the file in-place with a .bak backup.

Key bindings (single-keystroke, no Enter needed):
    ↑ / p     Previous entry
    ↓ / n     Flip label (privacy <-> no_privacy)
    Space/Enter  Accept label, go next
    q         Quit and save

Usage:
    python scripts/review_labels.py --input data/privacy/train.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import shutil
import sys

import msvcrt

logger = logging.getLogger(__name__)


def read_csv(path: str) -> list[dict]:
    """Read CSV file, return list of {text, label} dicts."""
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({"text": row.get("text", ""), "label": row.get("label", "")})
    return rows


def write_csv(path: str, rows: list[dict]) -> None:
    """Write rows back to CSV."""
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        for row in rows:
            writer.writerow([row["text"], row["label"]])


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def read_key() -> str:
    """Read a single keystroke using msvcrt."""
    ch = msvcrt.getwch()
    # Arrow keys send two bytes on Windows: 0x00 or 0xE0 prefix
    if ch in ("\x00", "\xe0"):
        second = msvcrt.getwch()
        if second == "H":
            return "UP"
        elif second == "P":
            return "DOWN"
        return f"SPECIAL_{second}"
    return ch


def review_group(rows: list[dict], indices: list[int], first_label: str, second_label: str) -> int:
    """Review a group of entries sharing the same label.

    Returns the number of labels flipped.
    """
    if not indices:
        return 0

    flipped = 0
    pos = 0  # current position within indices list

    while True:
        idx = indices[pos]
        row = rows[idx]

        clear_screen()
        total = len(indices)
        print(f"[{pos + 1}/{total}]  label: {row['label']}")
        print(f"\n{row['text']}")
        print(f"\n↑=prev  ↓=next  n=flip  Space/Enter=next  q=quit")

        key = read_key()

        if key in ("UP",):
            if pos > 0:
                pos -= 1
        elif key in ("DOWN",):
            if pos < total - 1:
                pos += 1
        elif key == "n":
            # Toggle label
            row["label"] = second_label if row["label"] == first_label else first_label
            # Stay on current entry so user can verify
        elif key in (" ", "\r", "\n"):
            # Accept, go next
            if pos < total - 1:
                pos += 1
            else:
                # Reached end of group
                break
        elif key == "q":
            break

    return flipped


def main():
    parser = argparse.ArgumentParser(description="Interactive label review tool")
    parser.add_argument("--input", required=True, help="Input CSV file to review (edited in-place)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    input_path = args.input
    if not os.path.exists(input_path):
        logger.error("File not found: %s", input_path)
        return 1

    # Read rows
    rows = read_csv(input_path)
    if not rows:
        logger.warning("No rows in %s", input_path)
        return 0

    # Detect labels
    labels = sorted(set(r["label"] for r in rows))
    if len(labels) != 2:
        logger.error("Expected exactly 2 unique labels, found %d: %s", len(labels), labels)
        return 1

    # Determine first label: whichever appears first in the file
    first_label = rows[0]["label"]
    if first_label not in labels:
        logger.error("First row label '%s' not in detected labels %s", first_label, labels)
        return 1
    second_label = [l for l in labels if l != first_label][0]

    # Group indices by label (based on first-label order)
    first_indices = [i for i, r in enumerate(rows) if r["label"] == first_label]
    second_indices = [i for i, r in enumerate(rows) if r["label"] == second_label]

    logger.info("Loaded %d rows. Labels: %s (%d), %s (%d)",
                len(rows), first_label, len(first_indices), second_label, len(second_indices))

    # Warning
    clear_screen()
    print("WARNING: This tool will modify the input file in-place!")
    print(f"  File: {input_path}")
    print(f"  A backup will be saved to: {input_path}.bak")
    print("\nPress Enter to continue, any other key to abort...")
    key = read_key()
    if key not in ("\r", "\n"):
        print("Aborted.")
        return 0

    # Backup
    bak_path = input_path + ".bak"
    shutil.copy2(input_path, bak_path)
    logger.info("Backup saved to %s", bak_path)

    # Pass 1: first label group
    clear_screen()
    print(f"=== Pass 1: Reviewing '{first_label}' entries ({len(first_indices)} total) ===")
    print("Press any key to start...")
    read_key()

    flipped1 = review_group(rows, first_indices, first_label, second_label)

    # Summary between passes
    clear_screen()
    print(f"=== Pass 1 complete: '{first_label}' entries ===")
    print(f"Reviewed: {len(first_indices)}  Flipped: {flipped1}")
    print(f"\nNext: reviewing '{second_label}' entries ({len(second_indices)} total)")
    print("Press any key to continue...")
    read_key()

    # Pass 2: second label group
    # Recalculate indices for second label (some may have been flipped to first_label)
    second_indices = [i for i, r in enumerate(rows) if r["label"] == second_label]

    clear_screen()
    print(f"=== Pass 2: Reviewing '{second_label}' entries ({len(second_indices)} total) ===")
    print("Press any key to start...")
    read_key()

    flipped2 = review_group(rows, second_indices, second_label, first_label)

    # Final summary
    clear_screen()
    print(f"=== Review complete ===")
    print(f"Pass 1 ('{first_label}'): {len(first_indices)} reviewed, {flipped1} flipped")
    print(f"Pass 2 ('{second_label}'): {len(second_indices)} reviewed, {flipped2} flipped")
    print(f"Total flips: {flipped1 + flipped2}")

    # Write back
    write_csv(input_path, rows)
    print(f"\nChanges saved to {input_path}")
    print(f"Original backup at {bak_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
