"""Official submission bundle tests."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_official_bundle_top_rank():
    profiles = __import__("src.submission_bundle", fromlist=["build_official_profiles"]).build_official_profiles(
        ROOT, 3
    )
    assert profiles is not None
    assert profiles[0]["candidate_id"] == "CAND_0018499"
    assert profiles[0]["rank"] == 1
    assert profiles[1]["candidate_id"] == "CAND_0039754"


def test_official_jd_detection():
    from src.submission_bundle import is_official_hackathon_jd

    short = (ROOT / "job_description_short.md").read_text(encoding="utf-8")
    assert is_official_hackathon_jd(short, ROOT)
