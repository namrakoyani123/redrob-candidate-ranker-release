"""JD enrichment for short header pastes."""

from pathlib import Path

from src.jd_parser import enrich_jd_text, parse_job_description

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "job_description.md"

SHORT = """Job Description: Senior AI Engineer — Founding Team
Company: Redrob AI
Location: Pune/Noida, India (Hybrid)
Experience Required: 5–9 years
"""


def test_short_jd_enriched_to_full_sections():
    enriched = enrich_jd_text(SHORT, FULL)
    full_text = FULL.read_text(encoding="utf-8")
    assert enriched.strip() == full_text.strip()
    low = enriched.lower()
    assert "things you absolutely need" in low
    assert "final note for the participants" in low


def test_short_enriched_parses_same_skills_as_full_file():
    cfg = {}
    full_text = FULL.read_text(encoding="utf-8")
    rich = parse_job_description(enrich_jd_text(SHORT, FULL), cfg)
    full = parse_job_description(full_text, cfg)
    assert len(rich.skill_terms) == len(full.skill_terms)


CUSTOM_SHORT = """Job Description: Senior NLP Engineer — Document Intelligence
Company: SpotDraft (Series B legal automation)
Location: Remote, India (IST overlap required)
Employment Type: Full-time
Experience Required: 4–7 years
"""


def test_custom_jd_short_header_not_enriched_with_redrob():
    enriched = enrich_jd_text(CUSTOM_SHORT, FULL)
    assert enriched.strip() == CUSTOM_SHORT.strip()
    assert "redrob" not in enriched.lower()
    assert "let's be honest about this role" not in enriched.lower()
