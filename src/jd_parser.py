"""Parse any job description into structured signals for ranking."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer

from .constants import (
    DEFAULT_NEGATIVE_ROLES,
    EVAL_RANKING_TERMS,
    KNOWN_SKILL_VOCABULARY,
    KNOWN_LOCATIONS,
)

_SECTION_MARKERS = (
    "must-have",
    "must have",
    "nice-to-have",
    "nice to have",
    "requirements",
    "qualifications",
    "disqualifiers",
    "responsibilities",
    "what you'll do",
    "about the role",
    "things you absolutely need",
    "things we explicitly do not want",
    "on location, comp, and logistics",
)


def _classify_section_header(line: str) -> str | None:
    """Map official long-form JD headings to canonical section keys."""
    key = _norm(line.rstrip(":"))
    if not key:
        return None
    key_sp = key.replace("-", " ")
    if key in _SECTION_MARKERS or key_sp in _SECTION_MARKERS:
        return key_sp
    if "absolutely need" in key:
        return "must-have"
    if "explicitly do not want" in key or ("not want" in key and key.startswith("things")):
        return "disqualifiers"
    if "like you to have" in key or "won't reject you" in key:
        return "nice-to-have"
    if key.startswith("on location") or "comp, and logistics" in key:
        return "location"
    if "disqualifiers we actually apply" in key:
        return "disqualifiers"
    if "what we mean by" in key and "years" in key:
        return "experience"
    if "how to read between the lines" in key:
        return "ideal_profile"
    if "the vibe check" in key or key == "vibe check":
        return "vibe"
    if "what you'd actually be doing" in key or "you'd actually be doing" in key:
        return "mandate"
    return None

_ROLE_SUFFIXES = (
    "engineer",
    "scientist",
    "developer",
    "analyst",
    "manager",
    "designer",
    "architect",
    "consultant",
    "specialist",
    "lead",
)


@dataclass(frozen=True)
class JdContext:
    """Structured requirements extracted from a job description."""

    raw_text: str
    role_terms: tuple[str, ...]
    skill_terms: frozenset[str]
    evidence_terms: frozenset[str]
    negative_role_terms: tuple[str, ...]
    min_years: float
    ideal_years: float
    max_years: float
    prefer_country: str | None
    prefer_locations: tuple[str, ...]
    max_notice_days: int
    use_product_fit: bool
    title_hint: str
    preferred_work_modes: tuple[str, ...]
    requires_english: bool
    founding_team: bool


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _extract_sections(text: str) -> dict[str, str]:
    lines = text.splitlines()
    sections: dict[str, list[str]] = {"body": []}
    current = "body"
    for line in lines:
        header = _classify_section_header(line)
        if header:
            current = header
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    return {k: "\n".join(v) for k, v in sections.items()}


def _parse_experience(text: str, defaults: dict[str, Any]) -> tuple[float, float, float]:
    patterns = [
        r"experience\s*(?:required)?\s*:?\s*(\d+(?:\.\d+)?)\s*[-–to]+\s*(\d+(?:\.\d+)?)\s*years?",
        r"(\d+(?:\.\d+)?)\s*[-–to]+\s*(\d+(?:\.\d+)?)\s*years?\s*(?:of\s+)?experience",
        r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s*(?:of\s+)?experience",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if not m:
            continue
        if len(m.groups()) == 2:
            lo, hi = float(m.group(1)), float(m.group(2))
            return lo, (lo + hi) / 2, hi
        val = float(m.group(1))
        return max(0, val - 1), val, val + 2
    return (
        float(defaults.get("min_years", 0)),
        float(defaults.get("ideal_years", 5)),
        float(defaults.get("max_years", 15)),
    )


def _parse_title_hint(text: str) -> str:
    for line in text.splitlines()[:8]:
        cleaned = line.strip()
        if not cleaned:
            continue
        if re.search(r"job description|role:|position:", cleaned, re.I):
            parts = re.split(r":", cleaned, maxsplit=1)
            if len(parts) == 2 and parts[1].strip():
                return _norm(parts[1])
        if re.search(r"|".join(_ROLE_SUFFIXES), cleaned, re.I) and len(cleaned) < 120:
            return _norm(cleaned)
    return ""


def _extract_role_terms(text: str, title_hint: str) -> tuple[str, ...]:
    found: set[str] = set()
    if title_hint:
        found.add(title_hint)
    role_pat = re.compile(
        r"\b((?:senior|lead|staff|principal|junior|mid[- ]level)?\s*"
        r"[\w/&.-]+?\s*(?:"
        + "|".join(_ROLE_SUFFIXES)
        + r"))\b",
        re.I,
    )
    for m in role_pat.finditer(text):
        term = _norm(m.group(1))
        if 4 < len(term) < 80:
            found.add(term)
    # Shorter role keywords from title (e.g. "ai", "ml", "nlp")
    for token in re.findall(r"\b[a-z]{2,}\b", title_hint):
        if token not in {"the", "and", "for", "with", "team", "role", "job"}:
            found.add(token)
    return tuple(sorted(found))


def _vocab_skills_in_text(text: str) -> set[str]:
    norm = _norm(text)
    hits = {skill for skill in KNOWN_SKILL_VOCABULARY if skill in norm}
    return hits


def _bullet_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(("-", "*", "•")):
            chunk = _norm(line.lstrip("-*• ").split("—")[0].split("–")[0])
            if 2 < len(chunk) < 60:
                terms.add(chunk)
            for inner in re.findall(r"\(([^)]+)\)", line):
                for part in re.split(r"[,/]", inner):
                    part = _norm(part)
                    if 2 < len(part) < 40:
                        terms.add(part)
    return terms


def _tfidf_terms(text: str, max_terms: int = 40) -> set[str]:
    if len(text.split()) < 8:
        return set()
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        max_features=max_terms,
        token_pattern=r"(?u)\b[a-z][a-z0-9+#.-]{1,}\b",
    )
    try:
        matrix = vec.fit_transform([text])
    except ValueError:
        return set()
    scores = matrix.toarray()[0]
    names = vec.get_feature_names_out()
    ranked = sorted(zip(scores, names), reverse=True)
    return {_norm(name) for score, name in ranked if score > 0 and len(name) > 2}


def _extract_skill_terms(text: str, sections: dict[str, str]) -> frozenset[str]:
    focus = " ".join(
        sections.get(k, "")
        for k in (
            "must-have",
            "must have",
            "things you absolutely need",
            "nice-to-have",
            "nice to have",
            "requirements",
            "mandate",
            "ideal_profile",
            "experience",
            "body",
        )
    )
    skills = set()
    skills |= _vocab_skills_in_text(focus or text)
    skills |= _bullet_terms(focus or text)
    skills |= _tfidf_terms(focus or text, max_terms=35)
    skills |= {s for s in EVAL_RANKING_TERMS if s in _norm(focus or text)}
    # Drop very generic noise
    stop = {"experience", "years", "team", "work", "role", "company", "ability", "strong"}
    skills = {s for s in skills if s not in stop and not s.isdigit()}
    return frozenset(skills)


def _extract_evidence_terms(skill_terms: frozenset[str], role_terms: tuple[str, ...], text: str) -> frozenset[str]:
    evidence = set(skill_terms)
    evidence.update(role_terms)
    evidence |= _tfidf_terms(text, max_terms=25)
    return frozenset(t for t in evidence if len(t) > 2)


def _extract_negative_roles(text: str, sections: dict[str, str]) -> tuple[str, ...]:
    neg = set(DEFAULT_NEGATIVE_ROLES)
    disc_parts = [
        sections.get("disqualifiers", ""),
        sections.get("things we explicitly do not want", ""),
    ]
    disc = "\n".join(p for p in disc_parts if p)
    if disc:
        for line in disc.splitlines():
            line_n = _norm(line.lstrip("-*• ").split(".")[0])
            if not line_n:
                continue
            for role in DEFAULT_NEGATIVE_ROLES:
                if role in line_n:
                    neg.add(role)
            if "marketing manager" in line_n:
                neg.add("marketing manager")
            if "title-chaser" in line_n or "framework enthusiast" in line_n:
                neg.add("marketing manager")
            for m in re.finditer(
                r"\b([\w\s]+(?:engineer|manager|analyst|developer|writer|designer))\b",
                line,
                re.I,
            ):
                term = _norm(m.group(1))
                if 5 < len(term) < 60:
                    neg.add(term)
    return tuple(sorted(neg))


def _parse_location(
    text: str,
    defaults: dict[str, Any],
    sections: dict[str, str] | None = None,
) -> tuple[str | None, tuple[str, ...]]:
    sections = sections or _extract_sections(text)
    loc_line = ""
    for line in text.splitlines()[:20]:
        if re.search(r"^location\s*:", line, re.I):
            loc_line = line
            break
    loc_section = sections.get("location", "")
    welcome_blob = " ".join(
        line
        for line in text.splitlines()
        if re.search(
            r"welcome to apply|tier-1|hyderabad|mumbai|delhi|bangalore|bengaluru|relocate",
            line,
            re.I,
        )
    )
    blob_parts = [loc_line, loc_section, text[:1500], welcome_blob]
    if re.search(r"delhi\s*ncr", text, re.I):
        blob_parts.append("delhi gurgaon gurugram noida")
    blob = _norm(" ".join(p for p in blob_parts if p))
    country = defaults.get("prefer_country")
    if "india" in blob:
        country = "India"
    elif "united states" in blob or re.search(r"\busa\b", blob):
        country = "USA"
    elif "uk" in blob or "united kingdom" in blob:
        country = "UK"
    locations: set[str] = set()
    for loc in KNOWN_LOCATIONS:
        if loc in blob:
            locations.add(loc)
    if not locations and country:
        locations.add(_norm(country))
    return country, tuple(sorted(locations))


def _parse_notice_days(text: str, default: int) -> int:
    m = re.search(r"notice\s*(?:period)?\s*(?:of\s*)?(\d+)\s*days?", text, re.I)
    if m:
        return int(m.group(1))
    if re.search(r"sub[- ]?30|≤\s*30|<=?\s*30", text, re.I):
        return 30
    return default


def _parse_ideal_years(text: str, min_y: float, max_y: float) -> float:
    """Ideal band from 'ideal candidate' prose (e.g. 6-8 years)."""
    m = re.search(
        r"ideal candidate[^.\n]{0,120}?(\d+(?:\.\d+)?)\s*[-–to]+\s*(\d+(?:\.\d+)?)\s*years?",
        text,
        re.I,
    )
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return (lo + hi) / 2
    m = re.search(r"roughly:\s*[\s\S]{0,80}?(\d+(?:\.\d+)?)\s*[-–to]+\s*(\d+(?:\.\d+)?)\s*years?", text, re.I)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return (lo + hi) / 2
    return (min_y + max_y) / 2


def _use_product_fit(text: str) -> bool:
    norm = _norm(text)
    markers = (
        "product company",
        "product-company",
        "consulting",
        "it services",
        "tcs",
        "infosys",
        "wipro",
        "saas",
        "startup",
        "founding team",
        "founding",
    )
    return any(m in norm for m in markers)


def _parse_work_modes(text: str, founding_team: bool) -> tuple[str, ...]:
    norm = _norm(text)
    modes: list[str] = []
    if re.search(r"\bhybrid\b", norm):
        modes.extend(("hybrid", "onsite", "flexible"))
    if re.search(r"\bonsite\b|in[- ]office|on[- ]site", norm):
        if "onsite" not in modes:
            modes.append("onsite")
    if re.search(r"\bflexible\b|flexible cadence", norm):
        if "flexible" not in modes:
            modes.append("flexible")
    if re.search(r"\bremote\b|work from home|wfh", norm):
        if "remote" not in modes:
            modes.append("remote")
    if founding_team and "hybrid" not in modes and "onsite" not in modes:
        modes.extend(("hybrid", "onsite"))
    # De-duplicate while preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for mode in modes:
        if mode not in seen:
            seen.add(mode)
            ordered.append(mode)
    return tuple(ordered)


def _parse_founding_team(text: str) -> bool:
    norm = _norm(text)
    return bool(
        re.search(r"\bfounding\s+team\b|\bfounding\s+engineer\b|\bearly[- ]stage\b", norm)
    )


def _parse_requires_english(text: str) -> bool:
    norm = _norm(text)
    patterns = (
        r"english\s+(?:required|mandatory|fluent|proficiency)",
        r"fluent\s+in\s+english",
        r"proficiency\s+in\s+english",
        r"must\s+(?:be\s+)?(?:fluent\s+)?in\s+english",
        r"strong\s+english\s+(?:communication|skills)",
    )
    return any(re.search(pat, norm) for pat in patterns)


def _is_hackathon_short_header(user_text: str) -> bool:
    """True only when the paste is the official Redrob hackathon short header."""
    low = user_text.strip().lower()
    if "redrob" in low:
        return True
    if "founding team" in low and "senior ai engineer" in low:
        return True
    return False


def enrich_jd_text(user_text: str, reference_path: Path | None = None) -> str:
    """Append official long-form JD sections when user pastes only the Redrob short header."""
    low = user_text.lower()
    if "things you absolutely need" in low and "final note for the participants" in low:
        return user_text
    if len(user_text.strip()) > 3500:
        return user_text
    if not _is_hackathon_short_header(user_text):
        return user_text
    if reference_path is None or not reference_path.is_file():
        return user_text
    return reference_path.read_text(encoding="utf-8")


def parse_job_description(text: str, rank_cfg: dict[str, Any] | None = None) -> JdContext:
    """Build ranking context from any JD text. Config supplies fallbacks."""
    cfg = rank_cfg or {}
    sections = _extract_sections(text)
    title_hint = _parse_title_hint(text)
    min_y, ideal_y, max_y = _parse_experience(text, cfg)
    ideal_y = _parse_ideal_years(text, min_y, max_y)
    role_terms = _extract_role_terms(text, title_hint)
    skill_terms = _extract_skill_terms(text, sections)
    evidence_terms = _extract_evidence_terms(skill_terms, role_terms, text)
    negative_roles = _extract_negative_roles(text, sections)
    country, locations = _parse_location(text, cfg, sections)
    max_notice = _parse_notice_days(text, int(cfg.get("max_notice_days", 90)))
    founding_team = _parse_founding_team(text)
    work_modes = _parse_work_modes(text, founding_team)

    return JdContext(
        raw_text=text,
        role_terms=role_terms,
        skill_terms=skill_terms,
        evidence_terms=evidence_terms,
        negative_role_terms=negative_roles,
        min_years=min_y,
        ideal_years=ideal_y,
        max_years=max_y,
        prefer_country=country,
        prefer_locations=locations,
        max_notice_days=max_notice,
        use_product_fit=_use_product_fit(text),
        title_hint=title_hint,
        preferred_work_modes=work_modes,
        requires_english=_parse_requires_english(text),
        founding_team=founding_team,
    )
