"""Clean and normalize candidate profiles before ranking."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from tqdm import tqdm

from .constants import PROFICIENCY_WEIGHT

_VALID_PROFICIENCY = frozenset(PROFICIENCY_WEIGHT.keys())

# Experience/tenure phrases commonly appended to profile headlines
_HEADLINE_EXP_PATTERNS = (
    r"\d+(?:\.\d+)?\+?\s*(?:years?|yrs?)\s+of\s+experience",
    r"\d+(?:\.\d+)?\+?\s*(?:years?|yrs?)\s+experience",
    r"\d+(?:\.\d+)?\+?\s*(?:years?|yrs?)\s+in\s+[^|]+",
    r"\d+(?:\.\d+)?\+?\s*(?:years?|yrs?)\b",
    r"\b\d+(?:\.\d+)?\+?\s*years?\b",
)


def _norm_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _prof_rank(prof: str) -> float:
    return PROFICIENCY_WEIGHT.get(_norm_text(prof), 0.35)


def _clean_headline(text: str, flags: list[str]) -> str:
    """Remove experience tenure and numeric noise from profile headline."""
    h = _norm_text(text)
    if not h:
        return h
    fallback = h.split("|")[0].strip()

    for pat in _HEADLINE_EXP_PATTERNS:
        updated = re.sub(pat, "", h, flags=re.IGNORECASE)
        if updated != h:
            flags.append("headline_experience_removed")
            h = updated

    if re.search(r"\d", h):
        flags.append("headline_numbers_removed")
        h = re.sub(r"\d+(?:\.\d+)?", "", h)

    h = re.sub(r"\++", "", h)
    parts = [_norm_text(p) for p in h.split("|")]
    parts = [
        p
        for p in parts
        if p and p.lower() not in {"experience", "exp", "yrs", "years", "yrs experience"}
    ]
    h = " | ".join(parts) if parts else fallback
    h = re.sub(r"(?:\|\s*){2,}", "| ", h).strip(" |")
    return _norm_text(h) or fallback


# Years-of-experience phrases in profile summaries (parsed before opener removal)
_SUMMARY_YOE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^Professional\s+with\s+(\d+(?:\.\d+)?)\+?\s*years?\s+of\s+experience",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:Senior|Staff|Lead)?\s*(?:AI|ML|Machine Learning|Applied ML|Recommendation Systems)?\s*"
        r"(?:Engineer|Scientist|Developer|Manager|Analyst|Professional)\s+with\s+"
        r"(\d+(?:\.\d+)?)\+?\s*years?\s+of\s+(?:hands-on\s+)?experience",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(.+?)\s+with\s+(\d+(?:\.\d+)?)\+?\s*years?\s+of\s+experience\s+",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:with|having|over|more than|around|approximately|~)\s+"
        r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:hands-on\s+)?(?:experience|exp)?",
        re.IGNORECASE,
    ),
    re.compile(r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s+of\s+(?:hands-on\s+)?experience", re.IGNORECASE),
)


def _parse_summary_yoe(text: str) -> float | None:
    """Extract the first explicit years claim from a raw profile summary."""
    s = _norm_text(text)
    if not s:
        return None
    for pat in _SUMMARY_YOE_PATTERNS:
        m = pat.search(s)
        if not m:
            continue
        raw = m.group(2) if m.lastindex and m.lastindex >= 2 else m.group(1)
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if 0 < val <= 50:
            return val
    return None


def _career_yoe(career: list[dict[str, Any]]) -> float:
    months = sum(max(0, int(j.get("duration_months") or 0)) for j in (career or []))
    return round(months / 12.0, 1) if months else 0.0


def _resolve_years_of_experience(
    profile_yoe: float,
    summary_yoe: float | None,
    career_yoe: float,
) -> tuple[float, str | None]:
    """Pick trustworthy YOE when profile, summary opener, and career history disagree."""
    p = round(max(0.0, profile_yoe), 1)
    s = round(summary_yoe, 1) if summary_yoe is not None else None
    c = round(max(0.0, career_yoe), 1)

    def _close(a: float, b: float, tol: float = 1.0) -> bool:
        return abs(a - b) < tol

    # Summary and career agree — profile is often the honeypot field.
    if s is not None and c > 0 and _close(s, c):
        consensus = round((s + c) / 2, 1)
        if p > 0 and not _close(p, consensus):
            return consensus, "yoe_resolved_summary_career"

    # Profile and summary agree — career total may be inflated by overlapping jobs.
    if s is not None and p > 0 and _close(p, s) and c > 0 and c - max(p, s) >= 2.0:
        return round((p + s) / 2, 1), "yoe_resolved_profile_summary"

    # Profile and career agree — keep as-is.
    if p > 0 and c > 0 and _close(p, c):
        return round((p + c) / 2, 1), None

    # Profile and summary agree without inflated career.
    if s is not None and p > 0 and _close(p, s):
        return round((p + s) / 2, 1), None

    # Remaining disagreements: trust career when present.
    if c > 0 and p > 0 and not _close(p, c):
        if s is None or _close(s, c):
            return c, "yoe_resolved_career"
        if s is not None and _close(s, p):
            return c, "yoe_resolved_career"

    vals = sorted(v for v in (p, s, c) if v is not None and v > 0)
    if not vals:
        return 0.0, None
    median = vals[len(vals) // 2]
    if p > 0 and not _close(median, p):
        return median, "yoe_resolved_median"
    return median, None


_SUMMARY_OPENER_PATTERNS: tuple[tuple[str, str], ...] = (
    # "Professional with 12.5+ years of experience. I've spent..."
    (r"^Professional\s+with\s+\d+(?:\.\d+)?\+?\s*years?\s+of\s+experience\.?\s*", ""),
    # "Software engineer with 8.4 years of experience across..."
    (
        r"^(.+?)\s+with\s+\d+(?:\.\d+)?\+?\s*years?\s+of\s+experience\s+",
        r"\1 ",
    ),
)


def _clean_summary(text: str, flags: list[str]) -> str:
    """Remove templated experience opener; keep contextual experience mentions later."""
    s = _norm_text(text)
    if not s:
        return s
    original = s
    for pattern, repl in _SUMMARY_OPENER_PATTERNS:
        updated = re.sub(pattern, repl, s, count=1, flags=re.IGNORECASE)
        if updated != s:
            flags.append("summary_experience_opener_removed")
            s = _norm_text(updated)
    return s or original


def _clean_skills(skills: list[dict[str, Any]], flags: list[str]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for sk in skills or []:
        name = _norm_text(sk.get("name", ""))
        if not name:
            flags.append("empty_skill_name")
            continue
        prof = _norm_text(sk.get("proficiency", "beginner"))
        if prof not in _VALID_PROFICIENCY:
            flags.append("invalid_proficiency")
            prof = "beginner"
        months = max(0, int(sk.get("duration_months") or 0))
        endorsements = max(0, int(sk.get("endorsements") or 0))
        if prof in ("expert", "advanced") and months == 0:
            flags.append("expert_skill_zero_months")
            prof = "intermediate"
        cleaned = {
            "name": name.title() if name.islower() else sk.get("name", name).strip(),
            "proficiency": prof,
            "duration_months": months,
            "endorsements": endorsements,
        }
        prev = best.get(name.lower())
        if prev is None:
            best[name.lower()] = cleaned
        else:
            flags.append("duplicate_skill")
            prev_score = _prof_rank(prev["proficiency"]) * (1 + prev["duration_months"])
            new_score = _prof_rank(cleaned["proficiency"]) * (1 + cleaned["duration_months"])
            if new_score >= prev_score:
                best[name.lower()] = cleaned
    return list(best.values())


def _clean_career(career: list[dict[str, Any]], years: float, flags: list[str]) -> list[dict[str, Any]]:
    cleaned_jobs = []
    max_months = int(years * 12) if years > 0 else 600
    for job in career or []:
        j = dict(job)
        j["company"] = _norm_text(j.get("company", ""))
        j["title"] = _norm_text(j.get("title", ""))
        j["description"] = _norm_text(j.get("description", ""))
        j["industry"] = _norm_text(j.get("industry", ""))
        dm = max(0, int(j.get("duration_months") or 0))
        if dm > 0 and years > 0 and dm > max_months + 24:
            flags.append("impossible_job_duration")
            j["duration_months"] = max_months
        else:
            j["duration_months"] = dm
        cleaned_jobs.append(j)
    return cleaned_jobs


def _clean_signals(signals: dict[str, Any], flags: list[str]) -> dict[str, Any]:
    sig = dict(signals or {})
    for key in ("recruiter_response_rate", "interview_completion_rate", "offer_acceptance_rate"):
        if key in sig and sig[key] is not None and float(sig[key]) >= 0:
            val = float(sig[key])
            if val > 1:
                flags.append(f"clamped_{key}")
            sig[key] = min(max(val, 0.0), 1.0)
    if "profile_completeness_score" in sig:
        sig["profile_completeness_score"] = min(max(float(sig["profile_completeness_score"] or 0), 0), 100)
    if "notice_period_days" in sig:
        sig["notice_period_days"] = min(max(int(sig["notice_period_days"] or 0), 0), 180)
    sal = dict(sig.get("expected_salary_range_inr_lpa") or {})
    if sal:
        smin = max(0.0, float(sal.get("min", 0)))
        smax = max(0.0, float(sal.get("max", 0)))
        if smin > smax and smax > 0:
            flags.append("salary_min_max_swapped")
            smin, smax = smax, smin
        elif smax == 0 and smin > 0:
            smax = smin
        sig["expected_salary_range_inr_lpa"] = {"min": smin, "max": smax}
    gh = float(sig.get("github_activity_score", -1))
    if gh < 0:
        flags.append("github_activity_missing")
    else:
        sig["github_activity_score"] = min(max(gh, 0), 100)
    return sig


def _clean_location(text: str, flags: list[str]) -> str:
    """Collapse redundant city/state pairs like 'Delhi, Delhi' -> 'Delhi'."""
    loc = _norm_text(text)
    if not loc:
        return loc
    parts = [part.strip() for part in loc.split(",") if part.strip()]
    if len(parts) == 2 and parts[0].lower() == parts[1].lower():
        flags.append("location_redundant_state_removed")
        return parts[0]
    return loc


def _clean_profile(profile: dict[str, Any], flags: list[str], years: float | None = None) -> dict[str, Any]:
    p = dict(profile or {})
    if isinstance(p.get("headline"), str):
        p["headline"] = _clean_headline(p["headline"], flags)
    # Support alternate field name if present in exports
    if isinstance(p.get("profile_headline"), str):
        p["profile_headline"] = _clean_headline(p["profile_headline"], flags)
    if isinstance(p.get("summary"), str):
        p["summary"] = _clean_summary(p["summary"], flags)
    if isinstance(p.get("profile_summary"), str):
        p["profile_summary"] = _clean_summary(p["profile_summary"], flags)
    if isinstance(p.get("location"), str):
        p["location"] = _clean_location(p["location"], flags)
    for key in ("current_title", "current_company", "current_industry"):
        if key in p and isinstance(p[key], str):
            p[key] = _norm_text(p[key])
    yrs = float(years if years is not None else p.get("years_of_experience") or 0)
    if yrs < 0 or yrs > 50:
        flags.append("years_clamped")
        yrs = min(max(yrs, 0), 50)
    p["years_of_experience"] = round(yrs, 1)
    return p


def _clean_education(education: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned = []
    for edu in education or []:
        e = dict(edu)
        for key in ("institution", "degree", "field_of_study"):
            if key in e and isinstance(e[key], str):
                e[key] = _norm_text(e[key])
        cleaned.append(e)
    return cleaned


def clean_candidate(candidate: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Return a cleaned candidate record and list of fix flags."""
    flags: list[str] = []
    out = dict(candidate)
    raw_profile = out.get("profile") or {}
    raw_summary = raw_profile.get("summary") or raw_profile.get("profile_summary") or ""
    raw_career = out.get("career_history") or []

    profile_yoe = float(raw_profile.get("years_of_experience") or 0)
    summary_yoe = _parse_summary_yoe(raw_summary)
    career_yoe = _career_yoe(raw_career)
    resolved_yoe, yoe_reason = _resolve_years_of_experience(profile_yoe, summary_yoe, career_yoe)
    if yoe_reason:
        flags.append(yoe_reason)
    if abs(resolved_yoe - profile_yoe) >= 1.0:
        flags.append("yoe_mismatch_corrected")

    profile = _clean_profile(raw_profile, flags, years=resolved_yoe)
    years = float(profile.get("years_of_experience") or 0)
    out["profile"] = profile
    out["skills"] = _clean_skills(out.get("skills") or [], flags)
    out["career_history"] = _clean_career(raw_career, years, flags)
    out["education"] = _clean_education(out.get("education") or [])
    out["redrob_signals"] = _clean_signals(out.get("redrob_signals") or {}, flags)
    return out, flags


@dataclass
class CleanReport:
    total: int = 0
    candidates_with_fixes: int = 0
    flag_counts: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_candidates": self.total,
            "candidates_with_fixes": self.candidates_with_fixes,
            "fix_rate": round(self.candidates_with_fixes / max(self.total, 1), 4),
            "flag_counts": dict(self.flag_counts.most_common()),
        }


def iter_raw_candidates(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def clean_jsonl(
    input_path: Path,
    output_path: Path,
    report_path: Path | None = None,
) -> CleanReport:
    """Clean all candidates and write normalized JSONL plus optional report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = CleanReport()
    total = count_lines(input_path)

    with output_path.open("w", encoding="utf-8") as out_f:
        for raw in tqdm(iter_raw_candidates(input_path), total=total, desc="clean"):
            report.total += 1
            cleaned, flags = clean_candidate(raw)
            if flags:
                report.candidates_with_fixes += 1
                report.flag_counts.update(flags)
            out_f.write(json.dumps(cleaned, ensure_ascii=False) + "\n")

    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return report
