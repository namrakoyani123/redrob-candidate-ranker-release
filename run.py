#!/usr/bin/env python3
"""Rank candidates for any job description — outputs top-N shortlist CSV."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from src.cleaner import clean_jsonl
from src.jd_parser import enrich_jd_text, parse_job_description
from src.ranker import load_config, load_job_description, score_candidates, write_submission


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank candidates against any job description and produce a shortlist CSV.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config.yaml",
        help="Path to config.yaml (default: project config)",
    )
    parser.add_argument(
        "--jd",
        "--job-description",
        dest="jd",
        type=Path,
        default=None,
        help="Job description file (.md, .txt). Overrides config job_description_path.",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=None,
        help="Candidates JSONL file. Overrides config candidates_path.",
    )
    parser.add_argument(
        "--output",
        "-o",
        "--out",
        dest="output",
        type=Path,
        default=None,
        help="Output CSV path. Overrides config submission_path.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Number of candidates to rank (default: 100).",
    )
    parser.add_argument(
        "--clean-first",
        action="store_true",
        help="Clean raw candidates before ranking (writes cleaned JSONL from config).",
    )
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="Use candidates file as-is even if cleaned file is missing.",
    )
    parser.add_argument(
        "--no-semantic-rerank",
        action="store_true",
        help="Skip BGE bi-encoder (BM25-only semantic channel, faster).",
    )
    parser.add_argument(
        "--official",
        action="store_true",
        help="Use official raw candidates.jsonl from config with --clean-first.",
    )
    parser.add_argument(
        "--precompute-embeddings",
        action="store_true",
        help="Build data/embeddings index (one-time, then rank with --skip-clean).",
    )
    args = parser.parse_args()

    root = args.config.resolve().parent
    cfg = load_config(args.config)
    if args.no_semantic_rerank:
        cfg.setdefault("ranking", {}).setdefault("semantic_retrieval", {}).setdefault(
            "bi_encoder", {}
        )["enabled"] = False
    data_cfg = cfg["data"]

    raw_path = (root / data_cfg.get("raw_candidates_path", data_cfg["candidates_path"])).resolve()
    cleaned_path = (root / data_cfg.get("cleaned_candidates_path", data_cfg["candidates_path"])).resolve()
    candidates_path = (args.candidates or root / data_cfg["candidates_path"]).resolve()

    if args.official:
        args.clean_first = True
        candidates_path = raw_path

    if args.clean_first:
        print(f"Cleaning candidates: {raw_path}")
        clean_jsonl(raw_path, cleaned_path, root / "data" / "clean_report.json")
        candidates_path = cleaned_path
    elif not args.skip_clean and not args.candidates and not cleaned_path.exists() and raw_path.exists():
        print(f"Cleaned data not found — cleaning first ({cleaned_path.name})")
        clean_jsonl(raw_path, cleaned_path, root / "data" / "clean_report.json")
        candidates_path = cleaned_path

    jd_path = (args.jd or root / data_cfg["job_description_path"]).resolve()
    out_path = (args.output or root / cfg["output"]["submission_path"]).resolve()
    top_n = args.top_n or int(cfg["output"].get("top_n", 100))

    if not candidates_path.exists():
        raise FileNotFoundError(f"Candidates file not found: {candidates_path}")
    if not jd_path.exists():
        raise FileNotFoundError(f"Job description not found: {jd_path}")

    if args.precompute_embeddings:
        import subprocess

        script = root / "scripts" / "precompute_embeddings.py"
        cmd = [sys.executable, str(script), "--config", str(args.config), "--resume"]
        if args.candidates:
            cmd.extend(["--candidates", str(args.candidates)])
        _log = print
        _log("Precomputing embeddings (resume-safe checkpoints)...")
        subprocess.run(cmd, cwd=root, check=True)
        return

    jd_text = enrich_jd_text(load_job_description(jd_path), root / "job_description.md")
    jd_ctx = parse_job_description(jd_text, cfg.get("ranking", {}))

    print(f"Job description: {jd_path}")
    print(f"  Target role: {jd_ctx.title_hint or '(parsed from JD)'}")
    print(f"  Skills tracked: {len(jd_ctx.skill_terms)}")
    print(f"  Experience band: {jd_ctx.min_years:.0f}–{jd_ctx.max_years:.0f} years")
    if jd_ctx.prefer_country:
        print(f"  Preferred location: {jd_ctx.prefer_country}")
    if jd_ctx.preferred_work_modes:
        print(f"  Work arrangement: {', '.join(jd_ctx.preferred_work_modes)}")
    if jd_ctx.founding_team:
        print("  Founding-team role: onsite/hybrid preferred")
    if jd_ctx.requires_english:
        print("  Language: English required")
    print(f"Candidates: {candidates_path}")
    print(f"Output: {out_path} (top {top_n})")

    t0 = time.perf_counter()
    df = score_candidates(candidates_path, jd_text, cfg, jd=jd_ctx, project_root=root)
    submission = write_submission(df, out_path, top_n)
    elapsed = time.perf_counter() - t0

    print(f"\nRanking runtime: {elapsed:.1f}s")
    print(f"Wrote top {top_n} to {out_path}")
    comp_rel = cfg.get("output", {}).get("comparison_csv_path")
    if comp_rel:
        comp_path = (root / comp_rel).resolve()
        comp_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_path, comp_path)
        print(f"Exported comparison CSV: {comp_path}")
    print("\nTop 10 preview:")
    print(submission.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
