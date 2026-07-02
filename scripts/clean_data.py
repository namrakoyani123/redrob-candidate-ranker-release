#!/usr/bin/env python3
"""Clean candidate JSONL — normalize text, fix signals, remove skill noise."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cleaner import clean_jsonl  # noqa: E402
from src.ranker import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean candidates.jsonl before ranking")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config.yaml",
    )
    parser.add_argument("--input", type=Path, default=None, help="Raw candidates JSONL")
    parser.add_argument("--output", type=Path, default=None, help="Cleaned JSONL output")
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "data" / "clean_report.json",
        help="Cleaning statistics JSON",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    input_path = (args.input or ROOT / data_cfg.get("raw_candidates_path", data_cfg["candidates_path"])).resolve()
    output_path = (args.output or ROOT / data_cfg.get("cleaned_candidates_path", "data/candidates_cleaned.jsonl")).resolve()

    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    print(f"Cleaning: {input_path}")
    print(f"Output:   {output_path}")
    report = clean_jsonl(input_path, output_path, args.report.resolve())
    stats = report.to_dict()
    print(f"\nDone — {stats['candidates_with_fixes']}/{stats['total_candidates']} profiles fixed")
    print("Top fixes:")
    for flag, count in list(stats["flag_counts"].items())[:8]:
        print(f"  {flag}: {count}")
    print(f"\nReport: {args.report.resolve()}")


if __name__ == "__main__":
    main()
