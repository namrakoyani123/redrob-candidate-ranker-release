"""Trap and honeypot detection — JD-aware keyword stuffing guards."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from .constants import (
    ACADEMIC_RESEARCH_TERMS,
    ARCHITECTURE_ONLY_TERMS,
    CAREER_AI_KEYWORD_TERMS,
    CV_DOMAIN_TERMS,
    EXTERNAL_VALIDATION_TERMS,
    IT_SERVICES_COMPANIES,
    JOB_HOP_MAX_AVG_MONTHS,
    JOB_HOP_MIN_JOBS,
    CAREER_MISMATCH_RULES,
    NLP_IR_DOMAIN_TERMS,
    NON_CODING_SENIOR_TITLE_TERMS,
    PRE_LLM_PRODUCTION_TERMS,
    RECENT_CODE_TERMS,
)
from .jd_parser import JdContext
from .profile import _skill_matches_jd, career_evidence_score, role_fit_score


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _matches_jd_role(blob: str, jd: JdContext) -> bool:
    b = _norm(blob)
    return any(p in b for p in jd.role_terms) if jd.role_terms else role_fit_score("", "", blob, jd) > 0.45


def job_title_desc_mismatch(title: str, description: str) -> bool:
    """True when job description clearly belongs to a different role than the title."""
    t = _norm(title)
    d = _norm(description)
    if not t or not d:
        return False
    for title_key, forbidden in CAREER_MISMATCH_RULES:
        if title_key in t:
            return any(phrase in d for phrase in forbidden)
    return False


def is_job_hopper(career_history: list[dict[str, Any]]) -> bool:
    """JD disqualifier: frequent short stints (avg < 18 months over 4+ roles)."""
    durations = [
        int(job.get("duration_months") or 0)
        for job in career_history or []
        if int(job.get("duration_months") or 0) > 0
    ]
    if len(durations) < JOB_HOP_MIN_JOBS:
        return False
    return sum(durations) / len(durations) < JOB_HOP_MAX_AVG_MONTHS


def neg_role_career_ai_keywords(
    current_title: str, career_history: list[dict[str, Any]], jd: JdContext
) -> bool:
    """Non-engineering title with AI/ML buzzwords only in career descriptions."""
    title = _norm(current_title)
    if not any(neg in title for neg in jd.negative_role_terms):
        return False
    career_text = " ".join(
        _norm(job.get("description", "")) for job in career_history or []
    )
    return any(term in career_text for term in CAREER_AI_KEYWORD_TERMS)


_PRODUCTION_ML_TERMS = (
    "production",
    "deployed",
    "shipped",
    "serving",
    "million",
    "pytorch",
    "tensorflow",
    "sklearn",
    "xgboost",
    "ranking pipeline",
    "recommendation system",
)

_RESEARCH_ONLY_TERMS = (
    "research scientist",
    "postdoctoral",
    "postdoc",
    "phd candidate",
    "doctoral",
    "academic lab",
    "university lab",
    "pure research",
)

_LANGCHAIN_RECENT_MONTHS = 12
_RECENT_ROLE_MONTHS = 18


def _recent_career_text(career: list[dict[str, Any]], max_months: int = _RECENT_ROLE_MONTHS) -> str:
    """Aggregate descriptions from current + recent roles (JD: last 18 months)."""
    chunks: list[str] = []
    months = 0
    for job in career or []:
        if job.get("is_current"):
            chunks.append(_norm(job.get("description", "")))
            months += int(job.get("duration_months") or 0)
        elif months < max_months:
            chunks.append(_norm(job.get("description", "")))
            months += int(job.get("duration_months") or 0)
        if months >= max_months:
            break
    return " ".join(chunks)


def has_pre_llm_production_evidence(career: list[dict[str, Any]], summary: str) -> bool:
    """JD: substantial ML production before recent LLM/LangChain work."""
    blob = " ".join(
        _norm(f"{j.get('title', '')} {j.get('description', '')}") for j in career or []
    )
    blob = f"{blob} {_norm(summary)}"
    return any(term in blob for term in PRE_LLM_PRODUCTION_TERMS) or any(
        term in blob for term in _PRODUCTION_ML_TERMS
    )


def is_langchain_only_profile(
    skills: list[dict[str, Any]],
    career: list[dict[str, Any]],
    summary: str,
) -> bool:
    """JD: recent (<12mo) LangChain/OpenAI work without pre-LLM production ML."""
    career_blob = " ".join(
        _norm(f"{j.get('title', '')} {j.get('description', '')}") for j in career or []
    )
    summary_n = _norm(summary)
    blob = f"{career_blob} {summary_n}"
    recent = _recent_career_text(career)
    has_lc = any(
        t in blob or t in recent
        for t in ("langchain", "llamaindex", "openai api", "chatgpt api")
    )
    if not has_lc:
        return False
    lc_skill_months = 0
    for sk in skills or []:
        name = _norm(sk.get("name", ""))
        if any(x in name for x in ("langchain", "llamaindex", "openai")):
            lc_skill_months += int(sk.get("duration_months") or 0)
    recent_lc_heavy = any(t in recent for t in ("langchain", "llamaindex", "openai"))
    short_lc_tenure = lc_skill_months <= _LANGCHAIN_RECENT_MONTHS or recent_lc_heavy
    if not short_lc_tenure:
        return False
    return not has_pre_llm_production_evidence(career, summary)


def is_pure_research_no_production(
    title: str,
    career: list[dict[str, Any]],
    summary: str,
) -> bool:
    """JD hard disqualifier: pure research / academic without production deployment."""
    title_blob = _norm(
        f"{title} {' '.join(_norm(j.get('title', '')) for j in career or [])} {summary}"
    )
    career_blob = " ".join(_norm(j.get("description", "")) for j in career or [])
    in_research_env = any(term in title_blob or term in career_blob for term in ACADEMIC_RESEARCH_TERMS)
    in_research_env = in_research_env or any(term in title_blob for term in _RESEARCH_ONLY_TERMS)
    if not in_research_env:
        if "research" in title_blob and "engineer" not in title_blob and "applied" not in title_blob:
            in_research_env = True
        else:
            return False
    has_prod = any(term in career_blob for term in _PRODUCTION_ML_TERMS) or any(
        term in career_blob for term in PRE_LLM_PRODUCTION_TERMS
    )
    return not has_prod


def is_research_only_profile(title: str, career: list[dict[str, Any]], summary: str) -> bool:
    """Alias for pure-research disqualifier (tests + traps)."""
    return is_pure_research_no_production(title, career, summary)


_TUTORIAL_FRAMEWORK_TERMS = (
    "how i used",
    "tutorial",
    "blog post",
    "demo project",
    "getting started with langchain",
    "follow along",
    "step-by-step guide",
)

_RANKING_PROD_TERMS = _PRODUCTION_ML_TERMS + (
    "ranking",
    "retrieval",
    "recommendation",
    "ndcg",
    "hybrid search",
    "vector database",
)


def is_framework_enthusiast_profile(
    skills: list[dict[str, Any]],
    career: list[dict[str, Any]],
    summary: str,
) -> bool:
    """JD: LangChain/tutorial-heavy profiles without ranking/search systems depth."""
    career_blob = " ".join(
        _norm(f"{j.get('title', '')} {j.get('description', '')}") for j in career or []
    )
    summary_n = _norm(summary)
    blob = f"{career_blob} {summary_n}"
    fw_hits = sum(1 for t in ("langchain", "llamaindex", "haystack") if t in blob)
    tutorial_hits = sum(1 for t in _TUTORIAL_FRAMEWORK_TERMS if t in blob)
    if fw_hits < 1 and tutorial_hits < 1:
        return False
    has_systems_depth = any(term in blob for term in _RANKING_PROD_TERMS)
    lc_months = sum(
        int(sk.get("duration_months") or 0)
        for sk in skills or []
        if any(x in _norm(sk.get("name", "")) for x in ("langchain", "llamaindex"))
    )
    return (fw_hits >= 2 or tutorial_hits >= 1) and not has_systems_depth and lc_months <= 24


def _is_it_services_company(company: str) -> bool:
    norm = _norm(company)
    return any(s in norm for s in IT_SERVICES_COMPANIES)


def is_consulting_only_career(career: list[dict[str, Any]]) -> bool:
    """JD: entire career at consulting/IT services with no product-company evidence."""
    if len(career or []) < 2:
        return False
    product_markers = ("product", "saas", "marketplace", "consumer", "startup", "scale")
    all_services = True
    has_product_evidence = False
    for job in career or []:
        company = _norm(job.get("company", ""))
        desc = _norm(job.get("description", ""))
        if any(m in desc for m in product_markers):
            has_product_evidence = True
        if not _is_it_services_company(company):
            all_services = False
    return all_services and not has_product_evidence


def is_cv_without_nlp_ir(
    title: str,
    career: list[dict[str, Any]],
    skills: list[dict[str, Any]],
    summary: str,
) -> bool:
    """JD: CV/speech/robotics expertise without meaningful NLP/IR exposure."""
    skill_blob = " ".join(_norm(s.get("name", "")) for s in skills or [])
    career_blob = " ".join(
        _norm(f"{j.get('title', '')} {j.get('description', '')}") for j in career or []
    )
    blob = _norm(f"{title} {summary} {skill_blob}")
    has_cv = any(term in blob or term in career_blob for term in CV_DOMAIN_TERMS)
    has_nlp_ir = any(
        term in blob or term in career_blob or term in skill_blob
        for term in NLP_IR_DOMAIN_TERMS
    )
    return has_cv and not has_nlp_ir


def is_architect_without_recent_code(
    title: str,
    career: list[dict[str, Any]],
    summary: str,
) -> bool:
    """JD: senior/architect/tech-lead without production code in last 18 months."""
    t = _norm(title)
    if not any(term in t for term in NON_CODING_SENIOR_TITLE_TERMS):
        return False
    recent_blob = _norm(summary)
    recent_blob = f"{recent_blob} {_recent_career_text(career)}"
    has_code = any(term in recent_blob for term in RECENT_CODE_TERMS)
    if has_code:
        return False
    arch_only = any(term in recent_blob for term in ARCHITECTURE_ONLY_TERMS)
    return arch_only or not has_code


def lacks_external_validation(
    signals: dict[str, Any],
    career: list[dict[str, Any]],
    yrs: float,
) -> bool:
    """JD: 5+ years proprietary-only with no GitHub/papers/OSS signal."""
    if yrs < 5:
        return False
    try:
        gh = float(signals.get("github_activity_score") or 0)
    except (TypeError, ValueError):
        gh = 0.0
    if gh >= 35:
        return False
    career_blob = " ".join(_norm(j.get("description", "")) for j in career or [])
    if any(term in career_blob for term in EXTERNAL_VALIDATION_TERMS):
        return False
    saved = int(signals.get("saved_by_recruiters_30d") or 0)
    endorse = int(signals.get("endorsements_received") or 0)
    return gh < 25 and saved < 5 and endorse < 12


def detect_traps(candidate: dict[str, Any], jd: JdContext) -> dict[str, Any]:
    """Return trap flags and a penalty multiplier in [0.05, 1.0]."""
    p = candidate.get("profile") or {}
    sig = candidate.get("redrob_signals") or {}
    skills = candidate.get("skills") or []
    career = candidate.get("career_history") or []

    flags: list[str] = []
    penalty = 1.0

    title = p.get("current_title", "")
    summary = p.get("summary", "")
    blob = f"{title} {p.get('headline', '')}"
    jd_skill_n = sum(1 for s in skills if _skill_matches_jd(s.get("name", ""), jd))

    if jd_skill_n >= 7 and not _matches_jd_role(blob, jd):
        if any(neg in _norm(title) for neg in jd.negative_role_terms):
            flags.append("keyword_stuffer")
            penalty *= 0.08

    expert_zero = sum(
        1
        for s in skills
        if _norm(s.get("proficiency", "")) in ("expert", "advanced")
        and int(s.get("duration_months") or 0) == 0
    )
    if expert_zero >= 3:
        flags.append("expert_skills_zero_months")
        penalty *= 0.05

    yrs = float(p.get("years_of_experience") or 0)
    for job in career:
        dm = int(job.get("duration_months") or 0)
        if dm > 0 and yrs > 0 and dm > yrs * 12 + 24:
            flags.append("timeline_impossible")
            penalty *= 0.05
            break

    current_mismatch = False
    any_mismatch = False
    for job in career:
        if job_title_desc_mismatch(job.get("title", ""), job.get("description", "")):
            any_mismatch = True
            if job.get("is_current"):
                current_mismatch = True
    if current_mismatch:
        flags.append("title_description_mismatch")
        penalty *= 0.10
    elif any_mismatch:
        flags.append("title_description_mismatch")
        penalty *= 0.25

    if is_job_hopper(career):
        durations = [
            int(job.get("duration_months") or 0)
            for job in career or []
            if int(job.get("duration_months") or 0) > 0
        ]
        avg_months = sum(durations) / len(durations) if durations else 0.0
        role_fit = role_fit_score(title, p.get("headline", ""), summary, jd)
        career_ev = career_evidence_score(career, summary, jd)
        if avg_months >= 16.0 and role_fit >= 0.85 and career_ev >= 0.85:
            flags.append("title_chaser")
        elif avg_months >= 16.0 and role_fit >= 0.85:
            flags.append("title_chaser")
            penalty *= 0.82
        else:
            flags.append("title_chaser")
            penalty *= 0.35

    if neg_role_career_ai_keywords(title, career, jd):
        flags.append("career_ai_keyword_stuffer")
        penalty *= 0.12

    if is_langchain_only_profile(skills, career, summary):
        flags.append("langchain_only")
        penalty *= 0.10

    if is_framework_enthusiast_profile(skills, career, summary):
        flags.append("framework_enthusiast")
        penalty *= 0.22

    if is_pure_research_no_production(title, career, summary):
        flags.append("research_only_no_prod")
        penalty *= 0.08

    if jd.use_product_fit and is_consulting_only_career(career):
        flags.append("consulting_only_career")
        penalty *= 0.22

    if is_cv_without_nlp_ir(title, career, skills, summary):
        flags.append("cv_without_nlp_ir")
        penalty *= 0.20

    if is_architect_without_recent_code(title, career, summary):
        flags.append("architect_no_recent_code")
        penalty *= 0.12

    if lacks_external_validation(sig, career, yrs):
        flags.append("no_external_validation")
        penalty *= 0.55

    response = float(sig.get("recruiter_response_rate") or 0)
    if not sig.get("open_to_work_flag") and response < 0.15:
        flags.append("inactive_low_response")
        penalty *= 0.35

    # Hackathon JD: down-weight stale profiles with very low recruiter response.
    try:
        lad = (sig.get("last_active_date") or "")[:10]
        if lad:
            days_inactive = (datetime.now() - datetime.strptime(lad, "%Y-%m-%d")).days
            if days_inactive > 180 and response < 0.05:
                flags.append("inactive_6m_low_response")
                penalty *= 0.20
            elif days_inactive > 180 and response < 0.15:
                flags.append("inactive_6m")
                penalty *= 0.45
    except (ValueError, TypeError):
        pass

    if response < 0.10:
        flags.append("very_low_response")
        penalty *= 0.6

    return {"flags": flags, "penalty": max(penalty, 0.05)}
