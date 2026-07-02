"""Tests for JD-aligned profile signals."""

from src.jd_parser import parse_job_description
from src.profile import (
    blended_skill_fit,
    connection_count_signal,
    current_company_product_score,
    education_fit_score,
    education_reasoning_snippet,
    jd_assessment_fit_score,
    language_fit_score,
    location_feasibility_score,
    marketplace_activity_signal,
    offer_acceptance_signal,
    skill_fit_score,
    work_mode_fit_score,
)


def _jd(text: str):
    return parse_job_description(text, {})


def test_work_mode_hybrid_jd_prefers_hybrid_candidate():
    jd = _jd("Location: Pune. Hybrid work. Founding team role.")
    assert jd.preferred_work_modes == ("hybrid", "onsite", "flexible")
    assert work_mode_fit_score("hybrid", jd) >= 0.99
    assert work_mode_fit_score("remote", jd) < 0.6


def test_english_only_when_jd_requires():
    jd = _jd("Must-have: Python. English required for client calls.")
    assert jd.requires_english is True
    langs = [{"language": "English", "proficiency": "professional"}]
    assert language_fit_score(langs, jd) >= 0.9
    assert language_fit_score([], jd) == 0.2

    neutral_jd = _jd("Must-have: Python.")
    assert neutral_jd.requires_english is False
    assert language_fit_score(langs, neutral_jd) == 0.5


def test_current_company_product_boost():
    jd = _jd("Product company experience required. No TCS-only careers.")
    assert current_company_product_score("Zomato", jd) == 1.0
    assert current_company_product_score("TCS", jd) < 0.2


def test_offer_acceptance_missing_is_neutral():
    assert offer_acceptance_signal({"offer_acceptance_rate": -1}) == 0.4
    assert offer_acceptance_signal({"offer_acceptance_rate": 0.7}) > 0.6


def test_connection_count_missing_is_neutral():
    assert connection_count_signal({}) == 0.4
    assert connection_count_signal({"connection_count": 0}) == 0.4
    assert connection_count_signal({"connection_count": 500}) > 0.85
    assert connection_count_signal({"connection_count": 1000}) == connection_count_signal(
        {"connection_count": 500}
    )


def test_marketplace_activity_signal():
    assert marketplace_activity_signal({}) == 0.42
    high = marketplace_activity_signal(
        {
            "profile_views_received_30d": 40,
            "search_appearance_30d": 30,
            "applications_submitted_30d": 10,
        }
    )
    assert high > 0.75


def test_education_fit_and_snippet():
    edu = [{"field_of_study": "Computer Science", "degree": "BTech", "tier": "tier_1"}]
    assert education_fit_score(edu) > 0.85
    assert education_reasoning_snippet(edu) == "tier-1 Computer Science"
    assert education_reasoning_snippet([]) is None


def test_jd_assessment_fit_matches_jd_skills_only():
    jd = _jd("Must-have: FAISS, Python, embeddings. 5+ years.")
    fit, top = jd_assessment_fit_score(
        {"FAISS": 82, "YOLO": 95, "Cooking": 99},
        jd,
    )
    assert abs(fit - 0.82) < 0.01
    assert top == ("FAISS", 82.0)


def test_jd_assessment_missing_is_neutral():
    jd = _jd("Must-have: Python.")
    assert jd_assessment_fit_score({}, jd) == (0.5, None)
    assert jd_assessment_fit_score({"YOLO": 90}, jd) == (0.5, None)


def test_blended_skill_fit_caps_assessment_weight():
    jd = _jd("Must-have: Python.")
    skills = [{"name": "Python", "proficiency": "expert", "duration_months": 48, "endorsements": 10}]
    base = skill_fit_score(skills, jd)
    blended, _, _ = blended_skill_fit(skills, {"Python": 80}, jd, assessment_blend=0.15)
    expected = 0.85 * base + 0.15 * 0.8
    assert abs(blended - expected) < 0.01
    capped, _, _ = blended_skill_fit(skills, {"Python": 80}, jd, assessment_blend=0.5)
    expected_capped = 0.85 * base + 0.15 * 0.8
    assert abs(capped - expected_capped) < 0.01


def test_relocate_and_corridor_boost_location_fit():
    jd = _jd(
        "Location: Pune/Noida, India. Experience: 5-9 years. "
        "Candidates in Hyderabad, Mumbai, Delhi welcome to apply."
    )
    cfg = {
        "location_corridor_boost": 0.18,
        "location_relocation_boost": 0.14,
        "tier1_welcome_boost": 0.11,
        "relocate_intent_boost": 0.10,
        "outside_india_penalty": 0.20,
        "prefer_notice_days": 30,
        "max_notice_days": 90,
    }
    pune = location_feasibility_score(
        {"location": "Pune, Maharashtra", "country": "India"},
        {"notice_period_days": 20, "willing_to_relocate": False},
        jd,
        cfg,
    )
    hyd_reloc = location_feasibility_score(
        {"location": "Hyderabad, Telangana", "country": "India"},
        {"notice_period_days": 20, "willing_to_relocate": True},
        jd,
        cfg,
    )
    hyd_static = location_feasibility_score(
        {"location": "Hyderabad, Telangana", "country": "India"},
        {"notice_period_days": 20, "willing_to_relocate": False},
        jd,
        cfg,
    )
    london = location_feasibility_score(
        {"location": "London", "country": "UK"},
        {"notice_period_days": 20, "willing_to_relocate": False},
        jd,
        cfg,
    )
    assert pune > hyd_static
    assert hyd_reloc > hyd_static
    assert hyd_static > london
