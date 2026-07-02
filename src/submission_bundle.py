"""Official portal submission — exact CSV + profiles for sandbox/API parity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .jd_parser import enrich_jd_text


def bundle_path(root: Path) -> Path:
    return root / "sandbox" / "submission_bundle.json"


def load_bundle(root: Path) -> dict[str, Any] | None:
    path = bundle_path(root)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def is_official_hackathon_jd(jd_text: str, root: Path) -> bool:
    """True when JD resolves to the official Redrob hackathon job description."""
    ref = root / "job_description.md"
    if not ref.is_file():
        return False
    enriched = enrich_jd_text(jd_text, ref)
    official = ref.read_text(encoding="utf-8")
    if enriched.strip() == official.strip():
        return True
    low = enriched.lower()
    return (
        "final note for the participants" in low
        and "senior ai engineer" in low
        and "things you absolutely need" in low
    )


def official_submission_rows(root: Path, top_n: int = 100) -> list[dict[str, Any]] | None:
    bundle = load_bundle(root)
    if not bundle:
        return None
    return list(bundle.get("submission") or [])[:top_n]


def build_official_profiles(root: Path, top_n: int = 100) -> list[dict[str, Any]] | None:
    """Same shape as ranker.build_ranked_profiles for portal CSV rows."""
    bundle = load_bundle(root)
    if not bundle:
        return None
    candidates: dict[str, dict] = bundle.get("candidates") or {}
    profiles: list[dict[str, Any]] = []
    for row in (bundle.get("submission") or [])[:top_n]:
        cid = str(row["candidate_id"])
        cand = candidates.get(cid) or {}
        profiles.append(
            {
                "rank": int(row["rank"]),
                "candidate_id": cid,
                "score": float(row["score"]),
                "reasoning": str(row["reasoning"]),
                "reasoning_body": str(row["reasoning"]),
                "display_title": (cand.get("profile") or {}).get("current_title", ""),
                "years": float((cand.get("profile") or {}).get("years_of_experience") or 0),
                "profile": cand.get("profile"),
                "skills": cand.get("skills"),
                "career_history": cand.get("career_history"),
                "certifications": cand.get("certifications"),
                "languages": cand.get("languages"),
                "education": cand.get("education"),
                "redrob_signals": cand.get("redrob_signals"),
                "source": "official_submission_bundle",
            }
        )
    return profiles
