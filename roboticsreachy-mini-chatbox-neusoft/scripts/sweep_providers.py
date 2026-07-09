#!/usr/bin/env python3
"""Sweep all cascade providers one at a time via --autotest.

For each provider type (ASR, LLM, TTS), iterates over every provider defined
in cascade.yaml and runs `reachy-mini-conversation-app --autotest --no-camera`
with that single override, keeping defaults for the other two types.

Usage:
    uv run python scripts/sweep_providers.py
    uv run python scripts/sweep_providers.py --test-file scripts/my_utterances.txt
"""

import sys
import argparse
import subprocess
from pathlib import Path

import yaml


def load_providers(yaml_path: Path) -> dict[str, list[str]]:
    """Load provider names from cascade.yaml, keyed by type."""
    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {
        "asr": list(cfg["asr"]["providers"].keys()),
        "llm": list(cfg["llm"]["providers"].keys()),
        "tts": list(cfg["tts"]["providers"].keys()),
    }


def run_one(provider_type: str, provider_name: str, test_file: str | None) -> tuple[bool, str]:
    """Run autotest with a single provider override. Returns (passed, stderr_tail)."""
    cmd = [
        sys.executable, "-m", "reachy_mini_conversation_app.main",
        "--autotest",
        "--no-camera",
        f"--{provider_type}-provider", provider_name,
    ]
    if test_file:
        cmd[cmd.index("--autotest")] = "--autotest"
        # Replace --autotest with --autotest <file>
        idx = cmd.index("--autotest")
        cmd.insert(idx + 1, test_file)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    stderr_tail = "\n".join(result.stderr.strip().splitlines()[-20:])
    return result.returncode == 0, stderr_tail


def main() -> None:
    """Run the sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-file", default=None, help="Custom utterances file for --autotest")
    args = parser.parse_args()

    yaml_path = Path("cascade.yaml")
    if not yaml_path.exists():
        print("ERROR: cascade.yaml not found. Run from project root.", file=sys.stderr)
        sys.exit(1)

    providers = load_providers(yaml_path)
    results: list[tuple[str, str, bool, str]] = []

    for ptype, names in providers.items():
        for name in names:
            label = f"{ptype.upper()}: {name}"
            print(f"--- {label} ---")
            try:
                passed, stderr_tail = run_one(ptype, name, args.test_file)
            except subprocess.TimeoutExpired:
                passed, stderr_tail = False, "TIMEOUT (120s)"
            results.append((ptype, name, passed, stderr_tail))
            status = "PASS" if passed else "FAIL"
            print(f"  -> {status}\n")

    # Summary table
    print("\n=== SWEEP SUMMARY ===")
    print(f"{'Type':<6} {'Provider':<30} {'Status':<6}")
    print("-" * 44)
    n_pass = 0
    for ptype, name, passed, stderr_tail in results:
        status = "PASS" if passed else "FAIL"
        if passed:
            n_pass += 1
        print(f"{ptype.upper():<6} {name:<30} {status:<6}")
        if not passed and stderr_tail:
            for line in stderr_tail.splitlines()[-5:]:
                print(f"       {line}")

    total = len(results)
    print(f"\n{n_pass}/{total} passed")
    sys.exit(0 if n_pass == total else 1)


if __name__ == "__main__":
    main()
