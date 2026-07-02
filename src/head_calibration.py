"""Top-of-funnel calibration — boost consensus-strong profiles for NDCG@10."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _marketplace_signal(meta: dict[str, Any]) -> float:
    saved = int(meta.get("saved_by_recruiters_30d") or 0)
    rr = float(meta.get("response_rate") or 0)
    saved_n = min(saved / 20.0, 1.0)
    return 0.55 * rr + 0.45 * saved_n


def top50_rerank_score(meta: dict[str, Any], rank_cfg: dict[str, Any]) -> float:
    """
    NDCG@10-oriented second-pass score in [0, 1].
    Emphasizes eval frameworks, production career proof, marketplace traction.
    """
    if float(meta.get("trap_penalty") or 1.0) < 0.5:
        return 0.0
    if meta.get("below_jd_yoe_floor"):
        return 0.0

    cal = rank_cfg.get("head_calibration") or {}
    w = cal.get("top50_weights") or {}
    sem = float(meta.get("semantic") or 0)
    score = (
        float(w.get("semantic", 0.22)) * sem
        + float(w.get("evaluation_fit", 0.20)) * float(meta.get("evaluation_fit") or 0)
        + float(w.get("career_evidence", 0.16)) * float(meta.get("career_evidence") or 0)
        + float(w.get("role_fit", 0.12)) * float(meta.get("role_fit") or 0)
        + float(w.get("location_fit", 0.10)) * float(meta.get("location_fit") or 0)
        + float(w.get("platform", 0.10)) * float(meta.get("platform") or 0)
        + float(w.get("marketplace", 0.10)) * _marketplace_signal(meta)
    )

    notice = int(meta.get("notice_days") or 0)
    if notice <= 30:
        score += float(w.get("notice_bonus", 0.05))
    elif notice <= 60:
        score += float(w.get("notice_bonus", 0.05)) * 0.65
    elif notice <= 90 and float(meta.get("evaluation_fit") or 0) >= 0.80:
        score += float(w.get("notice_bonus", 0.05)) * 0.45

    yrs = float(meta.get("years") or 0)
    if 5.0 <= yrs <= 9.0:
        score += float(w.get("yoe_band_bonus", 0.04))

    title_n = str(meta.get("display_title") or meta.get("title") or "").lower()
    if "staff" in title_n:
        score += float(w.get("role_fit", 0.12)) * 0.35

    loc = str(meta.get("location") or "").lower()
    saved = int(meta.get("saved_by_recruiters_30d") or 0)
    if "pune" in loc or "noida" in loc:
        score += float(w.get("corridor_bonus", 0.05))
    elif willing := bool(meta.get("willing_to_relocate")):
        if float(meta.get("location_fit") or 0) >= 0.75:
            score += float(w.get("relocate_bonus", 0.06))
        else:
            score += float(w.get("relocate_bonus", 0.06)) * 0.55
    if saved >= 15:
        score += float(w.get("marketplace", 0.10)) * 0.5
    if saved >= 30:
        score += float(w.get("marketplace", 0.10)) * 0.35

    return min(score, 1.0)


def apply_top50_rerank(
    df: pd.DataFrame,
    rank_cfg: dict[str, Any],
    *,
    output_limit: int = 100,
) -> pd.DataFrame:
    """
    Re-order ranks 1–50 only (NDCG@10 focus). Ranks 51–100 stay in original pipeline order.
    Candidates may move only within the rerank_window (default top 80).
    """
    cal = rank_cfg.get("head_calibration") or {}
    output_head = int(cal.get("top50_output", 50))
    if not cal.get("top50_rerank_enabled", True) or len(df) < output_head:
        return df

    window = min(int(cal.get("rerank_window", 80)), len(df))
    adj_max = float(cal.get("rerank_adj_max", 0.12))

    original = df.reset_index(drop=True)
    pool = original.head(window).copy()

    pool["_rerank"] = pool.apply(lambda r: top50_rerank_score(r.to_dict(), rank_cfg), axis=1)
    lo, hi = float(pool["_rerank"].min()), float(pool["_rerank"].max())
    if hi > lo:
        pool["_rerank_adj"] = (pool["_rerank"] - lo) / (hi - lo) * adj_max
    else:
        pool["_rerank_adj"] = 0.0

    pool["score"] = pool["score"] + pool["_rerank_adj"]
    pool = pool.sort_values(
        ["score", "response_rate", "candidate_id"],
        ascending=[False, False, True],
    ).drop(columns=["_rerank", "_rerank_adj"])

    new_head = pool.head(output_head)
    used_ids = set(new_head["candidate_id"].astype(str))

    output_limit = max(int(output_limit), output_head)
    tail_rows: list[dict[str, Any]] = []
    for _, row in original.iterrows():
        cid = str(row["candidate_id"])
        if cid in used_ids:
            continue
        tail_rows.append(row.to_dict())
        used_ids.add(cid)
        if len(new_head) + len(tail_rows) >= output_limit:
            break

    out = pd.concat([new_head, pd.DataFrame(tail_rows)], ignore_index=True)
    moved = sum(
        1
        for i in range(min(output_head, len(out), len(original)))
        if out.iloc[i]["candidate_id"] != original.iloc[i]["candidate_id"]
    )
    if moved:
        print(f"  Top-{output_head} rerank (window {window}): {moved} positions changed")
    return out


def head_calibration_bonus(meta: dict[str, Any], rank_cfg: dict[str, Any]) -> float:
    """
    Small additive bonus (0–0.06) for elite marketplace + eval signals.
    Applied before final sort; intended to lift strong peers without overriding traps.
    """
    if float(meta.get("trap_penalty") or 1.0) < 0.5:
        return 0.0
    if meta.get("below_jd_yoe_floor"):
        return 0.0

    cal = rank_cfg.get("head_calibration") or {}
    if not cal.get("enabled", True):
        return 0.0

    bonus = 0.0
    cap = float(cal.get("max_bonus", 0.06))

    sem = float(meta.get("semantic") or 0)
    eval_fit = float(meta.get("evaluation_fit") or 0)
    career = float(meta.get("career_evidence") or 0)
    role = float(meta.get("role_fit") or 0)
    rr = float(meta.get("response_rate") or 0)
    notice = int(meta.get("notice_days") or 0)
    saved = int(meta.get("saved_by_recruiters_30d") or 0)

    if sem >= 0.80:
        bonus += 0.012
    elif sem >= 0.70:
        bonus += 0.006

    if eval_fit >= 0.88:
        bonus += 0.014
    elif eval_fit >= 0.75:
        bonus += 0.007

    if career >= 0.72 and role >= 0.62:
        bonus += 0.010

    if rr >= 0.78:
        bonus += 0.008
    if saved >= 12:
        bonus += 0.010
    elif saved >= 6:
        bonus += 0.005
    if saved >= 30:
        bonus += 0.012

    if meta.get("open_to_work"):
        bonus += 0.004
    if meta.get("willing_to_relocate"):
        bonus += 0.012

    # JD: 30+ notice still in scope — do not over-penalize strong profiles at 61–120d.
    if 61 <= notice <= 120 and eval_fit >= 0.75 and role >= 0.58:
        bonus += 0.012
    elif 31 <= notice <= 60 and eval_fit >= 0.70:
        bonus += 0.006

    loc_fit = float(meta.get("location_fit") or 0)
    if loc_fit >= 0.92:
        bonus += 0.016
    elif loc_fit >= 0.85:
        bonus += 0.010
    elif loc_fit >= 0.78:
        bonus += 0.006

    return min(bonus, cap)


def notice_guard_multiplier(meta: dict[str, Any], rank_cfg: dict[str, Any]) -> float:
    """Softer notice decay when marketplace + eval signals are strong."""
    notice = int(meta.get("notice_days") or 0)
    if notice <= 30:
        mult = 1.0
    else:
        eval_fit = float(meta.get("evaluation_fit") or 0)
        rr = float(meta.get("response_rate") or 0)
        saved = int(meta.get("saved_by_recruiters_30d") or 0)
        strong = eval_fit >= 0.78 or rr >= 0.75 or saved >= 10

        if notice <= 60:
            mult = 0.97 if strong else 0.95
        elif notice <= 90:
            mult = 0.94 if strong else 0.90
        elif notice <= 120:
            mult = 0.90 if strong else 0.86
        else:
            mult = 0.82 if strong else 0.78

    loc = str(meta.get("location") or "").lower()
    in_corridor = "pune" in loc or "noida" in loc
    if notice > 30 and loc and not in_corridor and not meta.get("willing_to_relocate"):
        mult *= 0.96
    return mult
