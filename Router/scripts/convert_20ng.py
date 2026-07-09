"""
scripts/convert_20ng.py

Convert 20 Newsgroups PII-Augmented dataset to training CSV format.

Pipeline:
  1. Read 20NG_5topics_PII_anotated.jsonl
  2. Split original text into sentences using nltk.sent_tokenize
  3. Generate sliding-window samples (configurable window sizes)
  4. Map entity offsets to determine privacy/no_privacy labels per sample
  5. Clean each sample (remove quote headers, signatures, merge newlines, strip > prefix)
  6. Deduplicate by final text
  7. Translate to Chinese via LLM (skip if no Chinese chars)
  8. Output train.csv

Usage:
    python scripts/convert_20ng.py [--window_sizes 1,2,3] [--llm_backend ollama]
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

# Default translation prompt template
DEFAULT_TRANSLATE_PROMPT = (
    "请将以下英文翻译为自然流畅的中文，只输出翻译结果，不要添加解释或标注：\n\n{text}"
)


# ------------------------------------------------------------------
# NLTK setup
# ------------------------------------------------------------------

def _ensure_nltk():
    """Ensure nltk punkt_tab tokenizer data is available.

    Tries standard download first; on SSL errors falls back to manual
    download via urllib.
    """
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
        return
    except LookupError:
        pass

    # Try standard download
    try:
        nltk.download("punkt_tab", quiet=True)
        return
    except Exception:
        pass

    # Fallback: manual download via urllib (avoids SSL issues on some systems)
    import ssl
    import urllib.request
    import zipfile

    nltk_dir = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "nltk_data", "tokenizers")
    if not os.path.exists(nltk_dir):
        # Try Linux/Mac path
        nltk_dir = os.path.join(os.path.expanduser("~"), "nltk_data", "tokenizers")
    os.makedirs(nltk_dir, exist_ok=True)

    url = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip"
    dest = os.path.join(nltk_dir, "punkt_tab.zip")

    logger.info("Manual download of punkt_tab (SSL workaround)...")
    try:
        ctx = ssl._create_unverified_context()
        urllib.request.urlretrieve(url, dest, context=ctx)
    except Exception:
        urllib.request.urlretrieve(url, dest)

    with zipfile.ZipFile(dest, "r") as z:
        z.extractall(nltk_dir)
    logger.info("punkt_tab downloaded successfully")


def sent_tokenize(text: str) -> list[tuple[str, int, int]]:
    """Split text into sentences using nltk, returning (sentence, start, end).

    The start/end offsets are relative to the input text.
    """
    import nltk
    try:
        # Try span_bounds parameter (nltk >= 3.9.2)
        spans = list(nltk.sent_tokenize(text, span_bounds=True))
        result = []
        for item in spans:
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], tuple):
                sent_text, (start, end) = item
                result.append((sent_text, start, end))
            else:
                result.append((str(item), 0, len(str(item))))
        return result
    except TypeError:
        # Fallback: tokenize and compute offsets manually
        pass

    sentences = nltk.sent_tokenize(text)
    result = []
    search_start = 0
    for sent in sentences:
        # Find the sentence in the original text starting from search_start
        idx = text.find(sent, search_start)
        if idx == -1:
            # Fallback: try with stripped/normalized matching
            result.append((sent, search_start, search_start + len(sent)))
            search_start += len(sent)
        else:
            result.append((sent, idx, idx + len(sent)))
            search_start = idx + len(sent)
    return result


# ------------------------------------------------------------------
# Text cleaning
# ------------------------------------------------------------------

# Email quote header patterns
QUOTE_HEADER_RE = re.compile(
    r"^In\s+(article|message)\s+.*?(?:wrote|said)\s*:\s*$",
    re.MULTILINE | re.IGNORECASE,
)
QUOTE_HEADER_RE2 = re.compile(
    r"^On\s+.*?(?:wrote|said)\s*:\s*$",
    re.MULTILINE | re.IGNORECASE,
)
# Signature separator (standalone line)
SIGNATURE_RE = re.compile(r"^--\s*$", re.MULTILINE)
# Inline signature (e.g. "\n-- Name, Title, City" or starts with "-- Name")
INLINE_SIGNATURE_RE = re.compile(r"(?:^|\n)--\s+\S.*$", re.DOTALL)
# Lines starting with > (quote prefix)
QUOTE_PREFIX_RE = re.compile(r"^>\s*", re.MULTILINE)
# Email header fields
EMAIL_HEADER_RE = re.compile(
    r"^(?:From|Subject|Organization|Lines|NNTP-Posting-Host|Message-ID|Date|To|Cc|Reply-To|Followup-To|Distribution|Keywords|Summary|Approved|Supersedes|Xref)\s*:",
    re.MULTILINE | re.IGNORECASE,
)
# Multiple blank lines
MULTI_BLANK_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Clean a text sample for training.

    Operations:
      - Remove email quote headers (In article ..., On ... wrote:)
      - Remove everything after signature separator (--)
      - Strip > prefix from quoted lines (keep content)
      - Remove email header fields (From:, Subject:, etc.)
      - Collapse multiple blank lines to single
      - Merge newlines to spaces, normalize whitespace
    """
    # Remove signature and everything after (standalone -- line)
    text = SIGNATURE_RE.split(text, maxsplit=1)[0]

    # Remove inline signature (e.g. "\n-- Mike Harrison, ...")
    text = INLINE_SIGNATURE_RE.sub("", text)

    # Remove standalone signature-like lines at start/end (e.g. "-- Name, Title, City")
    # Only match short lines (<80 chars after --) to avoid removing horizontal rules (---)
    text = re.sub(r"(?:^|\n)\s*--\s+(?!--)\S.{0,80}(?:\n|$)", "", text)

    # Remove quote headers
    text = QUOTE_HEADER_RE.sub("", text)
    text = QUOTE_HEADER_RE2.sub("", text)

    # Remove email headers
    text = EMAIL_HEADER_RE.sub("", text)

    # Strip > prefix from quoted lines (keep content)
    text = QUOTE_PREFIX_RE.sub("", text)

    # Collapse multiple blank lines
    text = MULTI_BLANK_RE.sub("\n\n", text)

    # Merge newlines to spaces, normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ------------------------------------------------------------------
# Entity offset mapping
# ------------------------------------------------------------------

def entities_in_range(entities: list[dict], start: int, end: int) -> list[dict]:
    """Return entities whose span overlaps with [start, end)."""
    result = []
    for ent in entities:
        ent_start = ent.get("start", -1)
        ent_end = ent.get("end", -1)
        # Check overlap
        if ent_start < end and ent_end > start:
            result.append(ent)
    return result


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------

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


def count_existing_rows(path: str) -> int:
    """Count rows in existing CSV (excluding header) for resume support."""
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return 0
        return sum(1 for _ in reader)


def main():
    parser = argparse.ArgumentParser(description="Convert 20 Newsgroups PII-Aug to train.csv")
    parser.add_argument("--input", default=None,
                        help="Path to 20NG_5topics_PII_anotated.jsonl (auto-detected if omitted)")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (auto-detected if omitted)")
    parser.add_argument("--window_sizes", default="1,2,3",
                        help="Comma-separated window sizes for sliding window (default: 1,2,3)")
    parser.add_argument("--llm_backend", default="ollama", choices=["ollama", "qwen", "spark"],
                        help="LLM backend for translation (default: ollama)")
    parser.add_argument("--translate_prompt", default=None,
                        help="Custom translation prompt template (use {text} as placeholder)")
    parser.add_argument("--skip_translate", action="store_true",
                        help="Skip translation step (output English text)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Parse window sizes
    window_sizes = [int(x.strip()) for x in args.window_sizes.split(",")]
    logger.info("Window sizes: %s", window_sizes)

    # Resolve paths
    dataset_dir = os.path.join(_project_root, "data", "privacy", "datasets", "20 Newsgroups PII-Aug")
    input_path = args.input or os.path.join(dataset_dir, "20NG_5topics_PII_anotated.jsonl")
    output_path = args.output or os.path.join(dataset_dir, "train.csv")

    if not os.path.exists(input_path):
        logger.error("Input not found: %s", input_path)
        return 1

    # Ensure nltk data
    _ensure_nltk()

    # Load records
    records = load_jsonl(input_path)
    logger.info("Loaded %d records from %s", len(records), input_path)

    # Step 1-4: Split sentences, sliding window, map entities, clean, dedupe
    seen = set()
    samples = []  # list of (cleaned_text, label)

    for rec_idx, rec in enumerate(records):
        text = rec.get("text", "")
        entities = rec.get("entities", [])

        # Step 1: Split into sentences on original text (preserving offsets)
        sentences = sent_tokenize(text)
        if not sentences:
            continue

        # Step 2-3: Sliding window samples
        for ws in window_sizes:
            for i in range(len(sentences)):
                if i + ws > len(sentences):
                    break

                # Window covers sentences[i : i+ws]
                win_start = sentences[i][1]   # start offset of first sentence
                win_end = sentences[i + ws - 1][2]  # end offset of last sentence

                # Raw text for this window (from original text)
                raw_window = text[win_start:win_end]

                # Step 4: Check if any entity falls within this window
                overlapping = entities_in_range(entities, win_start, win_end)
                label = "privacy" if overlapping else "no_privacy"

                # Step 5: Clean the text
                cleaned = clean_text(raw_window)

                if not cleaned:
                    continue

                # Step 6: Dedupe by final text
                if cleaned in seen:
                    continue
                seen.add(cleaned)

                samples.append((cleaned, label))

        if (rec_idx + 1) % 100 == 0:
            logger.info("Processed %d/%d records, %d unique samples so far", rec_idx + 1, len(records), len(samples))

    logger.info("Total unique samples: %d", len(samples))

    # Count label distribution
    privacy_count = sum(1 for _, l in samples if l == "privacy")
    no_privacy_count = len(samples) - privacy_count
    logger.info("Label distribution: privacy=%d, no_privacy=%d", privacy_count, no_privacy_count)

    # Step 7: Translate
    translate_prompt = args.translate_prompt or DEFAULT_TRANSLATE_PROMPT

    if not args.skip_translate:
        client = create_client(args.llm_backend)
        logger.info("Translating with backend=%s, model=%s", args.llm_backend, client.model)

    # Resume support
    existing_count = count_existing_rows(output_path)
    if existing_count > 0:
        logger.info("Resuming: %d rows already in output, starting from row %d", existing_count, existing_count + 1)

    write_header = existing_count == 0
    mode = "a" if existing_count > 0 else "w"

    with open(output_path, mode, encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["text", "label"])

        for i, (text, label) in enumerate(samples):
            # Skip already processed rows
            if i < existing_count:
                continue

            if args.skip_translate:
                final_text = text
            else:
                prompt = translate_prompt.format(text=text)
                try:
                    translated = client.generate(prompt)
                    if has_chinese(translated):
                        final_text = translated
                    else:
                        logger.warning("Row %d: Translation has no Chinese chars, skipping: %s", i + 1, translated[:80])
                        continue
                except Exception as e:
                    logger.error("Row %d: Translation failed: %s", i + 1, e)
                    continue

            writer.writerow([final_text, label])
            f.flush()

            if (i + 1) % 50 == 0 or i + 1 == len(samples):
                logger.info("Translation progress: %d/%d", i + 1, len(samples))

    logger.info("Done. Output: %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
