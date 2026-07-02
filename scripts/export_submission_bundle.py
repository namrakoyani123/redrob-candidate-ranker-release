#!/usr/bin/env python3
"""Export portal CSV + top-100 profiles for sandbox/API parity (no 100K re-rank)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    csv_path = ROOT / "team_redrob_candidate_ranker.csv"
    jsonl_path = ROOT / "data" / "candidates_cleaned.jsonl"
    out_path = ROOT / "sandbox" / "submission_bundle.json"

    if not csv_path.is_file():
        raise SystemExit(f"Missing {csv_path}")
    if not jsonl_path.is_file():
        raise SystemExit(f"Missing {jsonl_path}")

    rows: list[dict[str, str]] = []
    want: set[str] = set()
    with csv_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "candidate_id": row["candidate_id"],
                    "rank": row["rank"],
                    "score": row["score"],
                    "reasoning": row["reasoning"],
                }
            )
            want.add(row["candidate_id"])

    candidates: dict[str, dict] = {}
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cand = json.loads(line)
            cid = cand["candidate_id"]
            if cid in want:
                candidates[cid] = cand
            if len(candidates) == len(want):
                break

    missing = want - set(candidates)
    if missing:
        raise SystemExit(f"Profiles missing for {len(missing)} submission IDs: {sorted(missing)[:5]}...")

    bundle = {
        "source_csv": csv_path.name,
        "pipeline_note": "Produced by run.py on full 100K + job_description.md",
        "submission": rows,
        "candidates": candidates,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
    mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {out_path} ({mb:.1f} MB, {len(rows)} rows, {len(candidates)} profiles)")


if __name__ == "__main__":
    main()
