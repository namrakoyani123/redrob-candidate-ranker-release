"""Head calibration bonuses for consensus-strong profiles."""

from src.head_calibration import apply_top50_rerank, head_calibration_bonus, notice_guard_multiplier


def test_strong_profile_gets_notice_relief():
    meta = {
        "trap_penalty": 1.0,
        "below_jd_yoe_floor": False,
        "notice_days": 90,
        "evaluation_fit": 0.90,
        "response_rate": 0.83,
        "saved_by_recruiters_30d": 19,
    }
    assert notice_guard_multiplier(meta, {}) >= 0.94


def test_head_bonus_for_elite_signals():
    meta = {
        "trap_penalty": 1.0,
        "below_jd_yoe_floor": False,
        "semantic": 0.83,
        "evaluation_fit": 0.90,
        "career_evidence": 0.75,
        "role_fit": 0.70,
        "response_rate": 0.83,
        "saved_by_recruiters_30d": 19,
        "open_to_work": True,
        "notice_days": 90,
        "location_fit": 0.70,
    }
    bonus = head_calibration_bonus(meta, {"head_calibration": {"enabled": True, "max_bonus": 0.06}})
    assert bonus >= 0.04


def test_top50_rerank_promotes_high_eval_candidate():
    import pandas as pd

    cfg = {
        "head_calibration": {
            "top50_rerank_enabled": True,
            "rerank_window": 5,
            "rerank_adj_max": 0.20,
            "top50_output": 2,
        }
    }
    df = pd.DataFrame(
        [
            {"candidate_id": "A", "score": 1.02, "semantic": 0.5, "evaluation_fit": 0.5, "career_evidence": 0.5,
             "role_fit": 0.5, "location_fit": 0.5, "platform": 0.5, "trap_penalty": 1.0, "response_rate": 0.3,
             "saved_by_recruiters_30d": 0, "notice_days": 30, "years": 6},
            {"candidate_id": "B", "score": 0.99, "semantic": 0.85, "evaluation_fit": 0.95, "career_evidence": 0.85,
             "role_fit": 0.75, "location_fit": 0.7, "platform": 0.8, "trap_penalty": 1.0, "response_rate": 0.85,
             "saved_by_recruiters_30d": 18, "notice_days": 90, "years": 7},
            {"candidate_id": "C", "score": 0.5, "semantic": 0.2, "evaluation_fit": 0.4, "career_evidence": 0.4,
             "role_fit": 0.4, "location_fit": 0.4, "platform": 0.4, "trap_penalty": 1.0, "response_rate": 0.2,
             "saved_by_recruiters_30d": 0, "notice_days": 120, "years": 6},
        ]
    )
    out = apply_top50_rerank(df, cfg)
    assert out.iloc[0]["candidate_id"] == "B"
    assert out.iloc[2]["candidate_id"] == "C"
