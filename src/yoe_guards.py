"""JD § 'What we mean by 5-9 years' — soft band + strong-signal salvage."""

from __future__ import annotations

from typing import Any

from .jd_parser import JdContext

# Below this YOE with weak signals → prescreen hard gate.
HARD_YOE_FLOOR_GAP = 2.0


def is_hard_below_yoe_floor(years: float, jd: JdContext) -> bool:
    """True only for far-below-band profiles (e.g. <3y when JD asks 5-9)."""
    if jd.min_years <= 0:
        return False
    return years < jd.min_years - HARD_YOE_FLOOR_GAP


def yoe_band_multiplier(meta: dict[str, Any], jd: JdContext | None = None) -> float:
    """
  5-9y is a preference band, not a hard filter (JD).
  Outside-band candidates with strong ML signals get a soft penalty only.
    """
    yrs = float(meta.get("years") or 0)
    if jd is None:
        min_y = float(meta.get("jd_min_years") or 0)
        max_y = float(meta.get("jd_max_years") or 0)
    else:
        min_y, max_y = jd.min_years, jd.max_years
    if min_y <= 0 and max_y <= 0:
        return 1.0
    if min_y <= yrs <= max_y:
        return 1.0

    career = float(meta.get("career_evidence") or 0)
    role = float(meta.get("role_fit") or 0)
    eval_fit = float(meta.get("evaluation_fit") or 0)
    sem = float(meta.get("semantic") or 0)
    strong = (
        (career >= 0.70 and role >= 0.55)
        or eval_fit >= 0.75
        or sem >= 0.72
        or float(meta.get("platform") or 0) >= 0.78
    )

    if yrs < min_y - 1.0:
        return 0.82 if strong else 0.45
    if yrs < min_y:
        return 0.92 if strong else 0.72
    if yrs > max_y + 2.0:
        return 0.78 if strong else 0.55
    if yrs > max_y:
        return 0.90 if strong else 0.75
    return 1.0
