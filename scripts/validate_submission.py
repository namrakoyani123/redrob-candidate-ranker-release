#!/usr/bin/env python3
"""Validate submission CSV against hackathon spec (submission_spec.txt)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["candidate_id", "rank", "score", "reasoning"]
CAND_ID_RE = re.compile(r"^CAND_\d{7}$")
TRAP_TITLE_TERMS = (
    "hr manager",
    "marketing manager",
    "accountant",
    "content writer",
    "graphic designer",
    "mechanical engineer",
    "civil engineer",
    "sales executive",
    "customer support",
    "operations manager",
)


def _load_candidate_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.add(json.loads(line)["candidate_id"])
    return ids


def validate(
    csv_path: Path,
    candidates_path: Path | None = None,
    *,
    expect_rows: int = 100,
) -> list[str]:
    errors: list[str] = []
    warnings: list[str] = []

    if not csv_path.exists():
        return [f"File not found: {csv_path}"]

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        return [f"Cannot read CSV: {exc}"]

    if list(df.columns) != REQUIRED_COLUMNS:
        errors.append(f"Columns must be exactly {REQUIRED_COLUMNS}; got {list(df.columns)}")

    if len(df) != expect_rows:
        errors.append(f"Expected {expect_rows} data rows; got {len(df)}")

    ranks = df["rank"].tolist()
    if sorted(ranks) != list(range(1, expect_rows + 1)):
        errors.append("Ranks must be unique integers 1..100")

    if df["candidate_id"].duplicated().any():
        errors.append("Duplicate candidate_id values found")

    for cid in df["candidate_id"].astype(str):
        if not CAND_ID_RE.match(cid):
            errors.append(f"Invalid candidate_id format: {cid}")
            break

    scores = df["score"].astype(float).tolist()
    if any(s < 0 or s > 1 for s in scores):
        warnings.append("Some scores outside [0, 1]")
    if len(set(scores)) < 10:
        errors.append("Too few unique scores — model may not be differentiating")
    if len(set(scores)) == 1:
        errors.append("All scores identical")
    for i in range(len(scores) - 1):
        if scores[i] < scores[i + 1]:
            errors.append("Scores must be non-increasing with rank")
            break

    reasoning = df["reasoning"].fillna("").astype(str)
    if (reasoning.str.strip() == "").any():
        warnings.append("Empty reasoning on some rows (allowed but penalized at Stage 4)")
    dup_reason = reasoning.duplicated().sum()
    if dup_reason > 5:
        warnings.append(f"{dup_reason} duplicate reasoning strings (Stage 4 risk)")

    trap_hits = 0
    for text in reasoning.head(20).str.lower():
        if any(term in text for term in TRAP_TITLE_TERMS):
            trap_hits += 1
    if trap_hits > 2:
        warnings.append(f"{trap_hits}/20 top rows mention trap-like role titles in reasoning")

    if candidates_path and candidates_path.exists():
        official = _load_candidate_ids(candidates_path)
        missing = [c for c in df["candidate_id"].astype(str) if c not in official]
        if missing:
            errors.append(f"{len(missing)} candidate_ids not in {candidates_path.name}")
    elif candidates_path:
        warnings.append(f"Candidates file not found: {candidates_path}")

    if errors:
        return errors
    print(f"PASS: {len(df)} rows, format OK, scores monotonic, {len(set(scores))} unique scores")
    for w in warnings:
        print(f"WARN: {w}")
    print(df.head(3).to_string(index=False))
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate hackathon submission CSV")
    parser.add_argument("csv", type=Path, nargs="?", default=Path("outputs/submission.csv"))
    parser.add_argument(
        "--candidates",
        type=Path,
        default=None,
        help="Official candidates.jsonl for ID validation",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    candidates = args.candidates
    if candidates is None:
        default = root / "data" / "candidates_cleaned.jsonl"
        raw = root.parent / "[PUB] India_runs_data_and_ai_challenge" / "India_runs_data_and_ai_challenge" / "candidates.jsonl"
        candidates = raw if raw.exists() else default
    errors = validate(args.csv.resolve(), candidates.resolve() if candidates else None)
    if errors:
        for err in errors:
            print(f"FAIL: {err}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
