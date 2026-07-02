"""Build canonical text and JD-driven structured features from candidate JSON."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from .constants import (
    EVAL_RANKING_TERMS,
    IT_SERVICES_COMPANIES,
    JD_CORRIDOR_CITIES,
    KNOWN_PRODUCT_COMPANIES,
    PROFICIENCY_WEIGHT,
    TIER1_RELOC_CITIES,
    TIER1_WELCOME_CITIES,
    WORK_MODE_VALUES,
)
from .jd_parser import JdContext


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _skill_matches_jd(skill_name: str, jd: JdContext) -> bool:
    name = _norm(skill_name)
    if not name:
        return False
    if name in jd.skill_terms:
        return True
    return any(term in name or name in term for term in jd.skill_terms if len(term) > 2)


def count_jd_skills(skills: list[dict[str, Any]], jd: JdContext) -> int:
    seen: set[str] = set()
    for sk in skills or []:
        name = _norm(sk.get("name", ""))
        if name and _skill_matches_jd(name, jd):
            seen.add(name)
    return len(seen)


def skill_fit_score(skills: list[dict[str, Any]], jd: JdContext) -> float:
    """Weighted skill depth on JD-relevant skills."""
    if not jd.skill_terms:
        return 0.4
    total = 0.0
    for sk in skills or []:
        name = _norm(sk.get("name", ""))
        if not _skill_matches_jd(name, jd):
            continue
        prof_key = _norm(sk.get("proficiency", "beginner"))
        prof = PROFICIENCY_WEIGHT.get(prof_key, 0.35)
        months = int(sk.get("duration_months") or 0)
        if prof_key in ("expert", "advanced") and months == 0:
            continue
        months_norm = min(months, 72) / 72.0
        endorse = min(int(sk.get("endorsements") or 0), 50) / 50.0
        total += prof * (0.55 + 0.3 * months_norm + 0.15 * endorse)
    return min(total / 8.0, 1.0)


def jd_assessment_fit_score(
    assessments: dict[str, Any] | None,
    jd: JdContext,
) -> tuple[float, tuple[str, float] | None]:
    """Avg score/100 for JD-matching platform assessments; neutral 0.5 if absent."""
    assess = assessments or {}
    if not assess or not jd.skill_terms:
        return 0.5, None
    matched: list[tuple[str, float]] = []
    for key, raw in assess.items():
        if not _skill_matches_jd(str(key), jd):
            continue
        try:
            score = float(raw)
        except (TypeError, ValueError):
            continue
        matched.append((str(key), min(max(score, 0.0), 100.0)))
    if not matched:
        return 0.5, None
    fit = sum(s / 100.0 for _, s in matched) / len(matched)
    best = max(matched, key=lambda x: x[1])
    return fit, best


def blended_skill_fit(
    skills: list[dict[str, Any]],
    assessments: dict[str, Any] | None,
    jd: JdContext,
    assessment_blend: float = 0.15,
) -> tuple[float, float, tuple[str, float] | None]:
    """Blend profile skill_fit with JD-matched platform assessments (max 15% assess weight)."""
    base = skill_fit_score(skills, jd)
    assess_fit, top = jd_assessment_fit_score(assessments, jd)
    blend = min(max(float(assessment_blend), 0.0), 0.15)
    blended = (1.0 - blend) * base + blend * assess_fit
    return blended, assess_fit, top


def role_fit_score(title: str, headline: str, summary: str, jd: JdContext) -> float:
    blob = _norm(f"{title} {headline} {summary}")
    pos = sum(1 for p in jd.role_terms if p in blob)
    neg = sum(1 for p in jd.negative_role_terms if p in blob)
    base = 0.35
    if jd.role_terms:
        base += min(pos * 0.18, 0.55)
    base -= min(neg * 0.22, 0.65)
    return max(0.0, min(base, 1.0))


def career_evidence_score(career_history: list[dict[str, Any]], summary: str, jd: JdContext) -> float:
    if not jd.evidence_terms:
        return 0.3
    summary_n = _norm(summary)
    weighted_hits = 0.0
    for term in jd.evidence_terms:
        best = 1.0 if term in summary_n else 0.0
        for job in career_history or []:
            block = _norm(f"{job.get('title', '')} {job.get('description', '')}")
            if term not in block:
                continue
            weight = 2.5 if job.get("is_current") else 1.0
            best = max(best, weight)
        weighted_hits += best
    base = min(weighted_hits / max(len(jd.evidence_terms) * 0.18, 4), 1.0)
    # JD must-have: ranking evaluation frameworks (NDCG, MRR, A/B, etc.)
    eval_blob = " ".join(
        _norm(f"{summary} {j.get('description', '')}") for j in (career_history or [])
    )
    eval_hits = sum(1 for t in EVAL_RANKING_TERMS if t in eval_blob)
    eval_bonus = min(0.12 * eval_hits, 0.28)
    # JD: shipper > researcher — production delivery in career
    prod_hits = sum(
        1
        for t in ("shipped", "deployed", "production", "serving", "million", "scale")
        if t in eval_blob
    )
    prod_bonus = min(0.04 * prod_hits, 0.16)
    return min(base + eval_bonus + prod_bonus, 1.0)


def evaluation_framework_score(
    career_history: list[dict[str, Any]],
    summary: str,
    skills: list[dict[str, Any]],
) -> float:
    """JD must-have: offline/online ranking evaluation experience."""
    blob = _norm(summary)
    blob += " " + " ".join(
        _norm(f"{j.get('title', '')} {j.get('description', '')}") for j in (career_history or [])
    )
    blob += " " + " ".join(_norm(s.get("name", "")) for s in (skills or []))
    hits = sum(1 for t in EVAL_RANKING_TERMS if t in blob)
    if hits == 0:
        return 0.42
    return min(0.50 + 0.10 * hits, 0.96)


def culture_engagement_score(
    profile: dict[str, Any],
    signals: dict[str, Any],
    career_history: list[dict[str, Any]],
) -> float:
    """JD vibe-check proxy: engaged, responsive, articulate profiles (async writing culture)."""
    score = 0.42
    summary = str(profile.get("summary") or "")
    if len(summary) >= 180:
        score += 0.14
    elif len(summary) >= 80:
        score += 0.07
    completeness = float(signals.get("profile_completeness_score") or 0)
    if completeness >= 80:
        score += 0.14
    elif completeness >= 60:
        score += 0.07
    rr = float(signals.get("recruiter_response_rate") or 0)
    if rr >= 0.65:
        score += 0.12
    elif rr >= 0.35:
        score += 0.06
    icr = float(signals.get("interview_completion_rate") or 0)
    if icr >= 0.7:
        score += 0.10
    careers = career_history or []
    if careers:
        avg_desc = sum(len(str(j.get("description") or "")) for j in careers) / len(careers)
        if avg_desc >= 200:
            score += 0.10
        elif avg_desc >= 100:
            score += 0.05
    if signals.get("open_to_work_flag"):
        score += 0.08
    views = int(signals.get("profile_views_received_30d") or 0)
    if views >= 100:
        score += 0.08
    elif views >= 40:
        score += 0.04
    search_app = int(signals.get("search_appearance_30d") or 0)
    if search_app >= 400:
        score += 0.06
    return min(score, 0.97)


def experience_fit_score(years: float, jd: JdContext) -> float:
    min_y, ideal_y, max_y = jd.min_years, jd.ideal_years, jd.max_years
    if min_y <= 0 and max_y <= 0:
        return 0.6
    if years < min_y - 1:
        return max(0.0, 0.4 + 0.15 * years)
    if years > max_y + 2:
        return max(0.5, 1.0 - 0.08 * (years - max_y))
    dist = abs(years - ideal_y)
    span = max(max_y - min_y, 1)
    return max(0.25, 1.0 - (dist / span) ** 1.4)


def _is_it_services_company(company: str) -> bool:
    norm = _norm(company)
    return any(s in norm for s in IT_SERVICES_COMPANIES)


def current_company_product_score(current_company: str, jd: JdContext) -> float:
    """Boost candidates at product/SaaS employers when JD emphasizes product experience."""
    if not jd.use_product_fit:
        return 0.5
    company = _norm(current_company)
    if not company:
        return 0.35
    if any(name in company for name in KNOWN_PRODUCT_COMPANIES):
        return 1.0
    if _is_it_services_company(company):
        return 0.12
    return 0.72


def product_company_score(
    career_history: list[dict[str, Any]],
    jd: JdContext,
    current_company: str = "",
) -> float:
    if not jd.use_product_fit:
        return 0.5
    if not career_history:
        career_score = 0.3
    else:
        product_hits = 0
        services_only = True
        for job in career_history:
            desc = _norm(job.get("description", ""))
            company = _norm(job.get("company", ""))
            if any(t in desc for t in ("product", "saas", "startup", "scale", "users")):
                product_hits += 1
            if not _is_it_services_company(company):
                services_only = False
        career_score = min(product_hits / max(len(career_history), 1), 1.0)
        if services_only:
            career_score *= 0.32
    company_score = current_company_product_score(current_company, jd)
    return 0.72 * career_score + 0.28 * company_score


_WORK_MODE_SCORES: dict[str, dict[str, float]] = {
    "hybrid": {"hybrid": 1.0, "onsite": 0.92, "flexible": 0.88, "remote": 0.52},
    "onsite": {"onsite": 1.0, "hybrid": 0.9, "flexible": 0.62, "remote": 0.35},
    "remote": {"remote": 1.0, "hybrid": 0.86, "flexible": 0.74, "onsite": 0.38},
    "flexible": {"flexible": 1.0, "hybrid": 0.9, "onsite": 0.82, "remote": 0.68},
}


def work_mode_fit_score(preferred_work_mode: str, jd: JdContext) -> float:
    """Match candidate preferred_work_mode to JD work arrangement (hybrid/onsite/remote)."""
    if not jd.preferred_work_modes:
        return 0.5
    candidate_mode = _norm(preferred_work_mode)
    if candidate_mode not in WORK_MODE_VALUES:
        return 0.45
    best = 0.0
    for jd_mode in jd.preferred_work_modes:
        table = _WORK_MODE_SCORES.get(jd_mode, {})
        best = max(best, table.get(candidate_mode, 0.45))
    if jd.founding_team and candidate_mode in ("hybrid", "onsite"):
        best = min(best + 0.04, 1.0)
    elif jd.founding_team and candidate_mode == "remote":
        best *= 0.88
    return best


_ENGLISH_PROFICIENCY_SCORE = {
    "native": 1.0,
    "bilingual": 1.0,
    "fluent": 0.95,
    "professional": 0.9,
    "full professional": 0.9,
    "conversational": 0.72,
    "intermediate": 0.55,
    "basic": 0.35,
    "elementary": 0.25,
}


def language_fit_score(languages: list[dict[str, Any]], jd: JdContext) -> float:
    """Score English only when the JD explicitly requires it; otherwise neutral."""
    if not jd.requires_english:
        return 0.5
    for entry in languages or []:
        if not isinstance(entry, dict):
            continue
        lang = _norm(entry.get("language", ""))
        if lang != "english":
            continue
        prof = _norm(entry.get("proficiency", ""))
        for key, score in _ENGLISH_PROFICIENCY_SCORE.items():
            if key in prof:
                return score
        return 0.75
    return 0.2


def offer_acceptance_signal(signals: dict[str, Any]) -> float:
    """Light touch — neutral when missing (-1); small boost when present."""
    raw = signals.get("offer_acceptance_rate")
    if raw is None:
        return 0.4
    val = float(raw)
    if val < 0:
        return 0.4
    return min(0.32 + 0.58 * val, 0.92)


def connection_count_signal(signals: dict[str, Any]) -> float:
    """Light platform network signal; neutral when missing or zero."""
    raw = signals.get("connection_count")
    if raw is None:
        return 0.4
    try:
        count = int(raw)
    except (TypeError, ValueError):
        return 0.4
    if count <= 0:
        return 0.4
    norm = min(count, 500) / 500.0
    return min(0.40 + 0.52 * norm, 0.92)


def marketplace_activity_signal(signals: dict[str, Any]) -> float:
    """Recruiter search visibility + candidate job-search activity (JD: active on platform)."""
    views = min(int(signals.get("profile_views_received_30d") or 0), 50) / 50.0
    apps = min(int(signals.get("applications_submitted_30d") or 0), 15) / 15.0
    search = min(int(signals.get("search_appearance_30d") or 0), 40) / 40.0
    if views + apps + search <= 0:
        return 0.42
    return min(0.38 + 0.55 * (0.35 * views + 0.25 * apps + 0.40 * search), 0.95)


def response_time_signal(signals: dict[str, Any]) -> float:
    """Faster recruiter response time is better; neutral when missing."""
    raw = signals.get("avg_response_time_hours")
    if raw is None:
        return 0.45
    try:
        hours = float(raw)
    except (TypeError, ValueError):
        return 0.45
    if hours <= 0:
        return 0.55
    if hours <= 24:
        return 0.92
    if hours <= 72:
        return 0.75
    if hours <= 168:
        return 0.55
    return 0.35


def platform_endorsements_signal(signals: dict[str, Any]) -> float:
    raw = signals.get("endorsements_received")
    if raw is None:
        return 0.42
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return 0.42
    return min(0.40 + 0.50 * (min(n, 80) / 80.0), 0.90)


_ML_EDU_FIELDS = frozenset(
    {
        "computer science",
        "machine learning",
        "artificial intelligence",
        "data science",
        "electrical engineering",
        "information technology",
        "software engineering",
        "statistics",
        "computational",
    }
)


def education_fit_score(education: list[dict[str, Any]]) -> float:
    """Light boost for strong technical education when present."""
    if not education:
        return 0.45
    best = 0.45
    for edu in education:
        field = _norm(edu.get("field_of_study", ""))
        degree = _norm(edu.get("degree", ""))
        tier = _norm(edu.get("tier", ""))
        blob = f"{field} {degree}"
        field_hit = any(term in blob for term in _ML_EDU_FIELDS)
        tier_boost = 1.0 if tier == "tier_1" else 0.85 if tier == "tier_2" else 0.7
        if field_hit:
            best = max(best, min(0.55 + 0.40 * tier_boost, 0.95))
    return best


def education_reasoning_snippet(education: list[dict[str, Any]]) -> str | None:
    """Short education line for Stage-4 reasoning when technical degree is strong."""
    if not education:
        return None
    best_score = 0.0
    best_label: str | None = None
    for edu in education:
        fit = education_fit_score([edu])
        if fit <= best_score:
            continue
        best_score = fit
        field = (edu.get("field_of_study") or "").strip()
        degree = (edu.get("degree") or "").strip()
        tier = _norm(edu.get("tier", ""))
        tier_lbl = "tier-1" if tier == "tier_1" else "tier-2" if tier == "tier_2" else ""
        core = field or degree
        if tier_lbl and core:
            best_label = f"{tier_lbl} {core}"
        elif core:
            best_label = core
        elif tier_lbl:
            best_label = tier_lbl
    return best_label if best_score >= 0.70 else None


def location_feasibility_score(
    profile: dict[str, Any],
    signals: dict[str, Any],
    jd: JdContext,
    cfg: dict[str, Any],
    languages: list[dict[str, Any]] | None = None,
) -> float:
    score = 0.40
    country = profile.get("country", "")
    loc = _norm(profile.get("location", ""))
    country_n = _norm(country)
    in_india = country_n == "india" or any(
        c in loc for c in ("india", *TIER1_RELOC_CITIES)
    )

    if jd.prefer_country and country == jd.prefer_country:
        score += 0.12
    if jd.prefer_locations and any(city in loc for city in jd.prefer_locations):
        score += 0.10

    corridor = tuple(cfg.get("jd_corridor_cities", JD_CORRIDOR_CITIES))
    corridor_boost = float(cfg.get("location_corridor_boost", 0.18))
    reloc_boost = float(cfg.get("location_relocation_boost", 0.14))
    tier1_welcome_boost = float(cfg.get("tier1_welcome_boost", 0.11))
    outside_india_penalty = float(cfg.get("outside_india_penalty", 0.20))
    relocate_intent_boost = float(cfg.get("relocate_intent_boost", 0.10))
    tier1 = tuple(cfg.get("tier1_reloc_cities", TIER1_RELOC_CITIES))
    welcome = tuple(cfg.get("tier1_welcome_cities", TIER1_WELCOME_CITIES))
    jd_corridor = [c for c in jd.prefer_locations if c in corridor] or list(corridor)

    in_corridor = any(c in loc for c in jd_corridor)
    in_tier1_welcome = any(c in loc for c in welcome)
    willing = bool(signals.get("willing_to_relocate"))

    if in_corridor:
        score += corridor_boost
    elif in_india and in_tier1_welcome:
        score += tier1_welcome_boost
    elif willing and in_india and any(t in loc for t in tier1):
        score += reloc_boost
    elif willing and in_india:
        score += reloc_boost * 0.78
    elif willing:
        score += reloc_boost * 0.42

    if willing:
        score += relocate_intent_boost

    if country_n and country_n not in ("india", "") and not in_india:
        score -= outside_india_penalty
    elif not in_india and not willing and not in_corridor:
        score -= outside_india_penalty * 0.65

    notice = int(signals.get("notice_period_days") or 90)
    prefer_notice = int(cfg.get("prefer_notice_days", 30))
    if notice <= prefer_notice:
        score += 0.18
    elif notice <= 60:
        score += 0.05
    elif notice <= jd.max_notice_days:
        score -= 0.08
    else:
        score -= 0.14

    sal = signals.get("expected_salary_range_inr_lpa") or {}
    smin, smax = float(sal.get("min", 0)), float(sal.get("max", 0))
    band_lo = float(cfg.get("salary_min_lpa", 0))
    band_hi = float(cfg.get("salary_max_lpa", 999))
    mid = (smin + smax) / 2 if smax else smin
    if mid > 0 and band_lo <= mid <= band_hi * 1.3:
        score += 0.08
    if signals.get("open_to_work_flag"):
        score += 0.07

    base = min(max(score, 0.0), 1.0)
    extras: list[float] = []
    if jd.preferred_work_modes:
        extras.append(work_mode_fit_score(signals.get("preferred_work_mode", ""), jd))
    if jd.requires_english:
        extras.append(language_fit_score(languages or [], jd))
    if not extras:
        return base
    return min(base * 0.84 + (sum(extras) / len(extras)) * 0.16, 1.0)


def activity_score(signals: dict[str, Any], ref_date: datetime | None = None) -> float:
    score = 0.5
    if signals.get("open_to_work_flag"):
        score += 0.25
    try:
        lad = signals.get("last_active_date", "")[:10]
        anchor = ref_date or datetime.now()
        days = (anchor - datetime.strptime(lad, "%Y-%m-%d")).days
        if days <= 60:
            score += 0.25
        elif days <= 180:
            score += 0.1
        else:
            score -= 0.25
    except (ValueError, TypeError):
        pass
    return max(0.0, min(score, 1.0))


def github_activity_signal(signals: dict[str, Any]) -> float:
    """Score GitHub activity: neutral when missing (-1), scaled boost when linked."""
    raw = signals.get("github_activity_score")
    if raw is None:
        return 0.40
    gh = float(raw)
    if gh < 0:
        return 0.40
    norm = min(max(gh, 0.0), 100.0) / 100.0
    # Present scores start above neutral; high activity (60+) gets a meaningful lift.
    return min(0.45 + 0.52 * (norm**0.75), 0.97)


def platform_signals_score(signals: dict[str, Any]) -> float:
    parts: list[float] = []
    parts.append(activity_score(signals))
    parts.append(min(float(signals.get("profile_completeness_score") or 0) / 100.0, 1.0))
    parts.append(min(float(signals.get("recruiter_response_rate") or 0), 1.0))
    parts.append(min(float(signals.get("interview_completion_rate") or 0), 1.0))
    parts.append(github_activity_signal(signals))
    parts.append(offer_acceptance_signal(signals))
    parts.append(connection_count_signal(signals))
    parts.append(marketplace_activity_signal(signals))
    parts.append(response_time_signal(signals))
    parts.append(platform_endorsements_signal(signals))
    saved_raw = int(signals.get("saved_by_recruiters_30d") or 0)
    saved = min(saved_raw, 40) / 40.0
    parts.append(saved)
    assess = signals.get("skill_assessment_scores") or {}
    if assess:
        parts.append(sum(assess.values()) / (100.0 * len(assess)))
    else:
        parts.append(0.4)
    verified = sum(
        1
        for k in ("verified_email", "verified_phone", "linkedin_connected")
        if signals.get(k)
    ) / 3.0
    parts.append(verified)
    return sum(parts) / len(parts)


def blended_platform_score(
    signals: dict[str, Any],
    profile: dict[str, Any],
    career_history: list[dict[str, Any]],
) -> float:
    """Platform signals + JD vibe-check engagement proxy."""
    base = platform_signals_score(signals)
    culture = culture_engagement_score(profile, signals, career_history)
    return min(0.78 * base + 0.22 * culture, 1.0)


def build_canonical_text(candidate: dict[str, Any]) -> str:
    """Single document for semantic matching."""
    p = candidate.get("profile") or {}
    chunks = [
        p.get("headline", ""),
        p.get("summary", ""),
        p.get("current_title", ""),
        p.get("current_industry", ""),
    ]
    for job in candidate.get("career_history") or []:
        chunks.extend([job.get("title", ""), job.get("description", ""), job.get("industry", "")])
    for sk in candidate.get("skills") or []:
        chunks.append(
            f"{sk.get('name')} {sk.get('proficiency')} {sk.get('duration_months')} months"
        )
    for edu in candidate.get("education") or []:
        chunks.append(
            f"{edu.get('degree')} {edu.get('field_of_study')} {edu.get('institution')} {edu.get('tier')}"
        )
    for cert in candidate.get("certifications") or []:
        chunks.append(f"{cert.get('name', '')} {cert.get('issuer', '')}".strip())
    company = (candidate.get("profile") or {}).get("current_company", "")
    if company:
        chunks.append(company)
    work_mode = (candidate.get("redrob_signals") or {}).get("preferred_work_mode", "")
    if work_mode:
        chunks.append(f"work mode {work_mode}")
    return "\n".join(c for c in chunks if c)


def build_sparse_retrieval_text(
    candidate: dict[str, Any],
    field_weights: dict[str, int] | None = None,
) -> str:
    """Field-weighted document for BM25 (title/skills/career emphasized)."""
    fw = field_weights or {
        "title": 3,
        "skills": 2,
        "career": 2,
        "summary": 1,
        "certs": 1,
    }
    p = candidate.get("profile") or {}

    def _repeat(text: str, times: int) -> str:
        text = (text or "").strip()
        if not text or times <= 0:
            return ""
        return " ".join([text] * times)

    title = f"{p.get('headline', '')} {p.get('current_title', '')}".strip()
    skill_text = " ".join(
        sk.get("name", "") for sk in (candidate.get("skills") or []) if sk.get("name")
    )
    career_parts: list[str] = []
    for job in candidate.get("career_history") or []:
        block = f"{job.get('title', '')} {job.get('description', '')}".strip()
        if block:
            career_parts.append(block)
    career = " ".join(career_parts)
    summary = p.get("summary", "")
    cert_text = " ".join(
        c.get("name", "") for c in (candidate.get("certifications") or []) if c.get("name")
    )
    company = p.get("current_company", "")
    chunks = [
        _repeat(title, int(fw.get("title", 3))),
        _repeat(skill_text, int(fw.get("skills", 2))),
        _repeat(career, int(fw.get("career", 2))),
        _repeat(summary, int(fw.get("summary", 1))),
        _repeat(cert_text, int(fw.get("certs", 1))),
        company,
    ]
    return "\n".join(c for c in chunks if c)


def top_jd_skills(skills: list[dict[str, Any]], jd: JdContext, n: int = 3) -> list[str]:
    ranked: list[tuple[int, str]] = []
    for sk in skills or []:
        name = sk.get("name", "")
        if not name or not _skill_matches_jd(name, jd):
            continue
        months = int(sk.get("duration_months") or 0)
        ranked.append((months, name))
    ranked.sort(reverse=True)
    if ranked:
        return [name for _, name in ranked[:n]]
    # Fallback: top skills by tenure when JD terms don't match names exactly
    fallback = []
    for sk in skills or []:
        name = sk.get("name", "")
        if name:
            fallback.append((int(sk.get("duration_months") or 0), name))
    fallback.sort(reverse=True)
    return [name for _, name in fallback[:n]]


_STRONG_CERT_PHRASES: tuple[str, ...] = (
    "machine learning specialty",
    "professional ml engineer",
    "deep learning specialization",
    "nlp specialization",
    "langchain for llm",
)

_GENERIC_CERT_PHRASES: tuple[str, ...] = (
    "six sigma",
    "scrum master",
    "scrum",
    "cloud practitioner",
    "pmp",
    "itil",
)

_ML_CERT_KEYWORDS: tuple[str, ...] = (
    "machine learning",
    "deep learning",
    "nlp",
    "natural language",
    "llm",
    "langchain",
    "tensorflow",
    "pytorch",
    "data science",
    "embedding",
    "retrieval",
    "ranking",
    "vector",
)


def _single_cert_score(name: str, issuer: str, jd: JdContext) -> float:
    blob = _norm(f"{name} {issuer}")
    if not blob:
        return 0.0
    if any(p in blob for p in _STRONG_CERT_PHRASES):
        return 1.0
    jd_hits = sum(1 for term in jd.skill_terms if len(term) > 3 and term in blob)
    if jd_hits >= 2:
        return 0.95
    if jd_hits == 1:
        return 0.78
    if any(kw in blob for kw in _ML_CERT_KEYWORDS):
        return 0.72
    if any(g in blob for g in _GENERIC_CERT_PHRASES):
        return 0.12
    return 0.2


def certification_fit_score(
    certifications: list[dict[str, Any]], jd: JdContext
) -> tuple[float, str | None]:
    """Score JD-relevant certifications; neutral baseline when none listed."""
    if not certifications:
        return 0.35, None
    scored: list[tuple[float, str]] = []
    for cert in certifications:
        name = cert.get("name", "")
        if not name:
            continue
        score = _single_cert_score(name, cert.get("issuer", ""), jd)
        scored.append((score, name))
    if not scored:
        return 0.35, None
    scored.sort(reverse=True)
    best_score, best_name = scored[0]
    strong_count = sum(1 for s, _ in scored if s >= 0.7)
    combined = best_score + min(max(strong_count - 1, 0), 2) * 0.06
    return min(max(combined, 0.35), 1.0), best_name if best_score >= 0.7 else None

