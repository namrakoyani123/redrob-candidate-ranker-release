"""JD § 5-9 years — soft band + hard floor."""

from src.jd_parser import parse_job_description
from src.yoe_guards import is_hard_below_yoe_floor, yoe_band_multiplier

_JD = """
Senior ML Engineer — 5-9 years experience building production recommendation systems.
Must know Python, PyTorch, ranking, retrieval.
"""


def _jd():
    return parse_job_description(_JD, {})


def test_hard_floor_below_three_years():
    jd = _jd()
    assert is_hard_below_yoe_floor(2.9, jd)
    assert not is_hard_below_yoe_floor(3.0, jd)


def test_soft_band_in_range():
    jd = _jd()
    meta = {"years": 7.0, "career_evidence": 0.5, "role_fit": 0.5}
    assert yoe_band_multiplier(meta, jd) == 1.0


def test_soft_band_below_with_strong_signals():
    jd = _jd()
    meta = {
        "years": 3.5,
        "career_evidence": 0.75,
        "role_fit": 0.60,
        "evaluation_fit": 0.5,
    }
    assert yoe_band_multiplier(meta, jd) == 0.82


def test_soft_band_one_year_below_min():
    jd = _jd()
    meta = {
        "years": 4.0,
        "career_evidence": 0.75,
        "role_fit": 0.60,
        "evaluation_fit": 0.5,
    }
    assert yoe_band_multiplier(meta, jd) == 0.92


def test_soft_band_below_without_strong_signals():
    jd = _jd()
    meta = {"years": 4.5, "career_evidence": 0.4, "role_fit": 0.4}
    assert yoe_band_multiplier(meta, jd) == 0.72


def test_soft_band_above_max():
    jd = _jd()
    meta = {"years": 11.0, "career_evidence": 0.5, "role_fit": 0.5}
    assert yoe_band_multiplier(meta, jd) == 0.75
