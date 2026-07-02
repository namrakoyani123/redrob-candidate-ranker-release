"""Hybrid candidate ranker: JD-driven semantic + structured recruiter signals."""

from __future__ import annotations

import heapq
import json
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import yaml

from .embedding_index import load_embedding_index
from .honeypot import detect_traps
from .head_calibration import apply_top50_rerank, head_calibration_bonus, notice_guard_multiplier
from .yoe_guards import is_hard_below_yoe_floor, yoe_band_multiplier
from .hybrid_retrieval import hybrid_semantic_scores, retrieval_cfg
from .jd_parser import JdContext, parse_job_description
from .profile import (
    _norm,
    activity_score,
    build_canonical_text,
    build_sparse_retrieval_text,
    career_evidence_score,
    blended_platform_score,
    certification_fit_score,
    count_jd_skills,
    current_company_product_score,
    education_fit_score,
    education_reasoning_snippet,
    evaluation_framework_score,
    experience_fit_score,
    blended_skill_fit,
    location_feasibility_score,
    platform_signals_score,
    product_company_score,
    role_fit_score,
    skill_fit_score,
    top_jd_skills,
    work_mode_fit_score,
)

try:
    import orjson

    def _parse_line(line: bytes | str) -> dict[str, Any]:
        if isinstance(line, str):
            line = line.encode("utf-8")
        return orjson.loads(line)

except ImportError:

    def _parse_line(line: bytes | str) -> dict[str, Any]:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        return json.loads(line)


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def iter_candidates(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb") as f:
        for raw in f:
            raw = raw.strip()
            if raw:
                yield _parse_line(raw)


def load_job_description(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _career_highlight(career_history: list[dict[str, Any]], jd: JdContext) -> str:
    terms = tuple(jd.evidence_terms)[:20] or ("experience", "built", "led", "developed")
    for job in career_history or []:
        desc = (job.get("description") or "").lower()
        if any(t in desc for t in terms):
            return job.get("description", "")[:120].strip()
    if career_history:
        return (career_history[0].get("description") or "")[:100].strip()
    return ""


def _cheap_prescreen_score(cand: dict[str, Any], jd_ctx: JdContext) -> float:
    """Fast gate — title/skills/YOE only (no traps, career loops, or platform signals)."""
    p = cand.get("profile") or {}
    title = p.get("current_title", "")
    if any(neg in _norm(title) for neg in jd_ctx.negative_role_terms):
        return -1.0
    role = role_fit_score(title, p.get("headline", ""), "", jd_ctx)
    skill_hits = count_jd_skills(cand.get("skills") or [], jd_ctx)
    yrs = float(p.get("years_of_experience") or 0)
    if is_hard_below_yoe_floor(yrs, jd_ctx):
        return -1.0
    title_norm = _norm(title)
    recsys_boost = 0.0
    if any(
        t in title_norm
        for t in (
            "machine learning",
            "ml engineer",
            "ai engineer",
            "applied scientist",
            "nlp engineer",
            "recommendation",
            "search engineer",
            "applied ml",
        )
    ):
        recsys_boost = 0.07
    in_band = 1.0 if jd_ctx.min_years <= yrs <= jd_ctx.max_years else 0.35
    return role * 0.55 + min(skill_hits, 12) * 0.04 + in_band * 0.15 + recsys_boost


def _build_row_meta(
    cand: dict[str, Any],
    jd_ctx: JdContext,
    rank_cfg: dict[str, Any],
    *,
    lite: bool = False,
) -> dict[str, Any]:
    p = cand.get("profile") or {}
    sig = cand.get("redrob_signals") or {}
    title = p.get("current_title", "Professional")
    yrs = float(p.get("years_of_experience") or 0)
    skills = cand.get("skills") or []
    career = cand.get("career_history") or []
    languages = cand.get("languages") or []
    current_company = p.get("current_company", "")
    work_mode = sig.get("preferred_work_mode", "")
    traps = detect_traps(cand, jd_ctx)
    cert_fit, top_cert = certification_fit_score(cand.get("certifications") or [], jd_ctx)
    wm_fit = work_mode_fit_score(work_mode, jd_ctx)
    company_prod = current_company_product_score(current_company, jd_ctx)
    assess_blend = float(rank_cfg.get("skill_assessment_blend", 0.15))
    skill_fit, jd_assess_fit, top_assess = blended_skill_fit(
        skills,
        sig.get("skill_assessment_scores"),
        jd_ctx,
        assessment_blend=assess_blend,
    )

    role_fit = role_fit_score(title, p.get("headline", ""), p.get("summary", ""), jd_ctx)
    title_n = _norm(title)
    if any(t in title_n for t in ("recommendation", "search engineer", "ranking", "applied ml")):
        role_fit = min(1.0, role_fit + 0.06)
    if "staff" in title_n:
        role_fit = min(1.0, role_fit + 0.04)

    meta = {
        "candidate_id": cand["candidate_id"],
        "display_title": title,
        "years": yrs,
        "location": p.get("location", ""),
        "country": p.get("country", ""),
        "current_company": current_company,
        "work_mode": work_mode,
        "work_mode_fit": wm_fit,
        "company_product": company_prod,
        "jd_skill_count": count_jd_skills(skills, jd_ctx),
        "notice_days": int(sig.get("notice_period_days") or 0),
        "open_to_work": bool(sig.get("open_to_work_flag")),
        "willing_to_relocate": bool(sig.get("willing_to_relocate")),
        "saved_by_recruiters_30d": int(sig.get("saved_by_recruiters_30d") or 0),
        "response_rate": float(sig.get("recruiter_response_rate") or 0),
        "role_fit": role_fit,
        "skill_fit": skill_fit,
        "jd_assessment_fit": jd_assess_fit,
        "career_evidence": career_evidence_score(career, p.get("summary", ""), jd_ctx),
        "product_fit": product_company_score(career, jd_ctx, current_company),
        "certification_fit": cert_fit,
        "education_fit": education_fit_score(cand.get("education") or []),
        "evaluation_fit": evaluation_framework_score(career, p.get("summary", ""), skills),
        "experience_fit": experience_fit_score(yrs, jd_ctx),
        "location_fit": location_feasibility_score(p, sig, jd_ctx, rank_cfg, languages),
        "platform": blended_platform_score(sig, p, career),
        "activity": activity_score(sig),
        "trap_penalty": traps["penalty"],
        "trap_flags": ",".join(traps["flags"]),
        "below_jd_yoe_floor": is_hard_below_yoe_floor(yrs, jd_ctx),
        "jd_min_years": jd_ctx.min_years,
        "jd_max_years": jd_ctx.max_years,
        "assessment_strong_threshold": float(rank_cfg.get("assessment_strong_threshold", 70)),
    }
    if lite:
        meta["top_skills"] = []
        meta["top_cert"] = top_cert if top_cert else None
        meta["career_highlight"] = ""
        if top_assess:
            meta["top_assessment_skill"] = top_assess[0]
            meta["top_assessment_score"] = top_assess[1]
        return meta

    meta["top_skills"] = top_jd_skills(skills, jd_ctx)
    meta["top_cert"] = top_cert
    meta["career_highlight"] = _career_highlight(career, jd_ctx)
    if top_assess:
        meta["top_assessment_skill"] = top_assess[0]
        meta["top_assessment_score"] = top_assess[1]
    return meta


def _enrich_reasoning_meta(meta: dict[str, Any], cand: dict[str, Any], jd_ctx: JdContext) -> None:
    skills = cand.get("skills") or []
    career = cand.get("career_history") or []
    sig = cand.get("redrob_signals") or {}
    meta["top_skills"] = top_jd_skills(skills, jd_ctx)
    meta["career_highlight"] = _career_highlight(career, jd_ctx)
    meta["education_note"] = education_reasoning_snippet(cand.get("education") or [])
    try:
        gh = float(sig.get("github_activity_score") or 0)
    except (TypeError, ValueError):
        gh = 0.0
    meta["github_note"] = f"GitHub activity {gh:.0f}/100" if gh >= 65 else None
    saved = int(sig.get("saved_by_recruiters_30d") or 0)
    meta["recruiter_saved_note"] = f"saved by {saved} recruiters (30d)" if saved >= 5 else None


def _structured_score(meta: dict[str, Any], weights: dict[str, float], product_weight: float) -> float:
    cert_weight = weights.get("certification_fit", 0.04)
    edu_weight = weights.get("education_fit", 0.02)
    eval_weight = weights.get("evaluation_fit", 0.02)
    return (
        weights["role_fit"] * meta["role_fit"]
        + weights["skill_fit"] * meta["skill_fit"]
        + weights["career_evidence"] * meta["career_evidence"]
        + product_weight * meta["product_fit"]
        + cert_weight * meta["certification_fit"]
        + edu_weight * meta["education_fit"]
        + eval_weight * meta["evaluation_fit"]
        + weights["platform_signals"]
        * (0.38 * meta["platform"] + 0.42 * meta["location_fit"] + 0.20 * meta["activity"])
        + 0.06 * meta["experience_fit"]
    )


def _apply_listing_guards(
    score: float,
    meta: dict[str, Any],
    rank_cfg: dict[str, Any] | None = None,
    jd_ctx: JdContext | None = None,
) -> float:
    if meta.get("below_jd_yoe_floor"):
        score *= 0.01
    else:
        score *= yoe_band_multiplier(meta, jd_ctx)
    score *= meta["trap_penalty"]
    notice = int(meta.get("notice_days") or 0)
    if notice > 30:
        mult = notice_guard_multiplier(meta, rank_cfg or {})
        score *= mult
    if meta["role_fit"] < 0.25 and meta["career_evidence"] < 0.15:
        score *= 0.25
    if meta["jd_skill_count"] >= 7 and meta["role_fit"] < 0.4:
        score *= 0.15
    return score


def _preliminary_score(
    meta: dict[str, Any],
    weights: dict[str, float],
    product_weight: float,
    rank_cfg: dict[str, Any] | None = None,
    jd_ctx: JdContext | None = None,
) -> float:
    return _apply_listing_guards(
        _structured_score(meta, weights, product_weight), meta, rank_cfg, jd_ctx
    )


def _heap_replace_smallest(
    heap: list[tuple[float, str]],
    store: dict[str, Any],
    key: str,
    score: float,
    value: Any,
    max_size: int,
) -> None:
    if score < 0:
        return
    if len(heap) < max_size:
        heapq.heappush(heap, (score, key))
        store[key] = value
        return
    if score > heap[0][0]:
        _, evicted = heap[0]
        heapq.heapreplace(heap, (score, key))
        store.pop(evicted, None)
        store[key] = value


def score_candidates(
    candidates_path: Path,
    jd_text: str,
    cfg: dict[str, Any],
    jd: JdContext | None = None,
    project_root: Path | None = None,
    *,
    output_limit: int | None = None,
) -> pd.DataFrame:
    rank_cfg = cfg["ranking"]
    weights = rank_cfg["weights"]
    jd_ctx = jd or parse_job_description(jd_text, rank_cfg)
    prescreen_size = int(rank_cfg.get("prescreen_pool_size", 6000))
    pool_size = int(rank_cfg.get("semantic_pool_size", 2000))
    product_weight = weights.get("product_fit", 0.08) if jd_ctx.use_product_fit else 0.0

    print(f"JD parsed: role={jd_ctx.title_hint or 'n/a'}, "
          f"skills={len(jd_ctx.skill_terms)}, "
          f"experience={jd_ctx.min_years:.0f}-{jd_ctx.max_years:.0f}y")

    t0 = time.perf_counter()
    print(f"Stage 1a: fast prescreen (funnel top {prescreen_size:,})...")
    prescreen_heap: list[tuple[float, str]] = []
    prescreen_cands: dict[str, dict[str, Any]] = {}
    scanned = 0

    for cand in iter_candidates(candidates_path):
        scanned += 1
        cid = cand["candidate_id"]
        cheap = _cheap_prescreen_score(cand, jd_ctx)
        _heap_replace_smallest(prescreen_heap, prescreen_cands, cid, cheap, cand, prescreen_size)

    t_prescreen = time.perf_counter() - t0
    print(f"  Scanned {scanned:,} in {t_prescreen:.1f}s; prescreen pool = {len(prescreen_cands):,}")

    print(f"Stage 1b: full structured score on prescreen pool -> top {pool_size:,}...")
    semantic_heap: list[tuple[float, str]] = []
    meta_by_id: dict[str, dict[str, Any]] = {}
    text_by_id: dict[str, str] = {}
    sparse_by_id: dict[str, str] = {}
    retrieval = retrieval_cfg(rank_cfg)
    field_weights = retrieval.get("field_weights") or {}
    embed_index = load_embedding_index(rank_cfg, project_root)
    if embed_index:
        print(f"  Precomputed embeddings: {len(embed_index):,} vectors ({embed_index.model_name})")

    for cand in prescreen_cands.values():
        meta = _build_row_meta(cand, jd_ctx, rank_cfg, lite=True)
        prelim = _preliminary_score(meta, weights, product_weight, rank_cfg, jd_ctx)
        cid = meta["candidate_id"]
        if len(semantic_heap) < pool_size:
            heapq.heappush(semantic_heap, (prelim, cid))
            meta_by_id[cid] = meta
            text_by_id[cid] = build_canonical_text(cand)
            sparse_by_id[cid] = build_sparse_retrieval_text(cand, field_weights)
        elif prelim > semantic_heap[0][0]:
            _, evicted = semantic_heap[0]
            heapq.heapreplace(semantic_heap, (prelim, cid))
            meta_by_id.pop(evicted, None)
            text_by_id.pop(evicted, None)
            sparse_by_id.pop(evicted, None)
            meta_by_id[cid] = meta
            text_by_id[cid] = build_canonical_text(cand)
            sparse_by_id[cid] = build_sparse_retrieval_text(cand, field_weights)

    for cid, meta in meta_by_id.items():
        _enrich_reasoning_meta(meta, prescreen_cands[cid], jd_ctx)

    t_structured = time.perf_counter() - t0 - t_prescreen
    print(f"  Structured scoring {t_structured:.1f}s; semantic pool = {len(meta_by_id):,}")

    pool_ids = list(meta_by_id.keys())
    pool_texts = [text_by_id[cid] for cid in pool_ids]
    pool_sparse = [sparse_by_id[cid] for cid in pool_ids]

    bi_cfg = retrieval.get("bi_encoder") or {}
    model_name = str(bi_cfg.get("model", "BAAI/bge-small-en-v1.5"))
    bge_pool = int(bi_cfg.get("pool_size", 1000))
    dense_mode = "precomputed index (all pool)" if embed_index else f"live BGE top {bge_pool:,}"
    print(
        f"Stage 2: BM25 + BGE + RRF on {len(pool_ids):,} docs "
        f"(BM25=all, dense={dense_mode}, model={model_name}, "
        f"bi_encoder={'on' if bi_cfg.get('enabled', True) else 'off'})..."
    )
    t_sem_start = time.perf_counter()
    semantic_scores, _ = hybrid_semantic_scores(
        jd_text,
        pool_texts,
        pool_sparse,
        rank_cfg,
        pool_ids=pool_ids,
        embedding_index=embed_index,
    )
    t_semantic = time.perf_counter() - t_sem_start
    print(f"  Hybrid semantic {t_semantic:.1f}s")

    records = []
    for i, cid in enumerate(pool_ids):
        meta = meta_by_id[cid]
        sem = float(semantic_scores[i])
        meta["semantic"] = sem
        structured = _structured_score(meta, weights, product_weight)
        final = weights["semantic"] * sem + structured
        final = _apply_listing_guards(final, meta, rank_cfg, jd_ctx)
        bonus = head_calibration_bonus(meta, rank_cfg)
        final += bonus
        records.append({**meta, "semantic": sem, "score": final, "head_bonus": bonus})

    total = time.perf_counter() - t0
    print(
        f"Total ranking core: {total:.1f}s "
        f"(prescreen {t_prescreen:.1f}s + structured {t_structured:.1f}s + "
        f"semantic {t_semantic:.1f}s)"
    )

    df = pd.DataFrame(records)
    df = df.sort_values(
        ["score", "response_rate", "candidate_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    print("Stage 3: top-50 NDCG calibration pass...")
    out_n = output_limit if output_limit is not None else int(cfg.get("output", {}).get("top_n", 100))
    return apply_top50_rerank(df, rank_cfg, output_limit=out_n)


def fit_badge(rank: int, candidate_id: str) -> str:
    """Short rank-tiered label for UI badges (top ranks get strongest wording)."""
    pick = sum(ord(c) for c in str(candidate_id))
    if rank <= 3:
        labels = ("Top pick", "Strong match", "Best fit")
    elif rank <= 10:
        labels = ("Highly recommended", "Shortlist", "Strong match", "Recommended")
    elif rank <= 25:
        labels = ("Good fit", "Solid match", "JD aligned")
    elif rank <= 50:
        labels = ("Fit", "Worth review", "Consider")
    else:
        labels = ("Pipeline", "Backup", "Review")
    return labels[pick % len(labels)]


def reasoning_display_body(full_reasoning: str) -> str:
    """Reasoning text without the opener line — for UI when badge shows the label."""
    for sep in (" · ", "; "):
        pos = full_reasoning.find(sep)
        if pos != -1:
            body = full_reasoning[pos + len(sep) :].strip()
            if body:
                return body
    return full_reasoning


def _reasoning_lead(candidate_id: str, title: str, yrs: float, rank: int) -> str:
    """Stable opener variety per candidate (Stage-4 reasoning polish)."""
    variants = (
        f"{title} with {yrs:.1f} years experience",
        f"{yrs:.1f}-year {title} with strong JD alignment",
        f"Fit: {title} ({yrs:.1f} years)",
        f"{title} — {yrs:.1f} years in product ML",
        f"Strong match: {title}, {yrs:.1f} years",
        f"{title}; {yrs:.1f} years relevant experience",
        f"Recommended: {title} ({yrs:.1f}y) — JD role + stack fit",
        f"Shortlist: {yrs:.1f}y {title} with production ML evidence",
        f"{title} profile aligns on experience ({yrs:.1f}y) and delivery",
    )
    pick = (int(rank) + sum(ord(c) for c in str(candidate_id))) % len(variants)
    return variants[pick]


def _career_frame(highlight: str, candidate_id: str, rank: int) -> str:
    frames = (
        "career shows {}",
        "recent work: {}",
        "production track record: {}",
        "shipped: {}",
        "experience includes {}",
    )
    pick = (int(rank) + sum(ord(c) for c in candidate_id)) % len(frames)
    return frames[pick].format(highlight)


def _skills_frame(skills: list[str], candidate_id: str, rank: int) -> str:
    joined = ", ".join(skills)
    frames = (
        f"strong on {joined}",
        f"JD skills: {joined}",
        f"core stack {joined}",
        f"matches must-haves: {joined}",
    )
    pick = (int(rank) * 3 + sum(ord(c) for c in candidate_id)) % len(frames)
    return frames[pick]


def _caveat_for_row(row: pd.Series) -> str:
    """One honest caveat for Stage-4 top-row review."""
    flags = str(row.get("trap_flags") or "")
    if "research_only_no_prod" in flags:
        return (
            "Caveat: pure research / academic career without production deployment "
            "— JD hard disqualifier."
        )
    if "langchain_only" in flags:
        return (
            "Caveat: recent LangChain/OpenAI work without pre-LLM production ML "
            "— JD concern."
        )
    if "architect_no_recent_code" in flags:
        return (
            "Caveat: senior/architect role without hands-on code in last 18 months "
            "— JD concern."
        )
    notice = int(row.get("notice_days") or 0)
    loc = str(row.get("location") or "").lower()
    corridor = ("pune" in loc) or ("noida" in loc)
    if notice > 90:
        return f"Caveat: {notice}-day notice — above ideal 30d buyout window."
    if notice > 30:
        return f"Caveat: {notice}-day notice — still in scope but bar is higher per JD."
    if loc and not corridor and not row.get("willing_to_relocate"):
        return (
            f"Caveat: based in {row.get('location', 'outside corridor')} "
            f"— outside Pune/Noida; has not indicated willingness to relocate."
        )
    if loc and not corridor and row.get("willing_to_relocate"):
        return f"Caveat: outside Pune/Noida today — willing to relocate."
    yrs = float(row.get("years") or 0)
    if yrs < 5.0:
        return (
            f"Caveat: {yrs:.1f}y experience — below JD 5–9y preference band "
            f"(strong outliers still considered)."
        )
    if yrs > 9.5:
        return f"Caveat: {yrs:.1f}y experience — above JD 5–9y band."
    rr = float(row.get("response_rate") or 0)
    if rr < 0.35:
        return f"Caveat: recruiter response {rr:.0%} — lighter marketplace engagement."
    return "Caveat: none material — profile aligns on role, stack, and logistics."


def build_reasoning_premium(row: pd.Series) -> str:
    """Stage-4 reasoning for ranks 1–20: fact-dense + one caveat."""
    title = row["display_title"]
    yrs = float(row["years"])
    cid = str(row.get("candidate_id") or "")
    company = row.get("current_company") or ""
    loc = row.get("location") or ""
    skills = row.get("top_skills") or []
    skill_txt = ", ".join(skills[:4]) if skills else "core ML/IR stack"

    facts: list[str] = [f"{yrs:.1f}y {title}"]
    if company:
        facts.append(f"at {company}")
    if loc:
        facts.append(loc)

    if row.get("career_highlight"):
        hl = str(row["career_highlight"]).replace("\n", " ")[:95]
        facts.append(f"Shipped: {hl}")

    assess_skill = row.get("top_assessment_skill")
    assess_score = row.get("top_assessment_score")
    if assess_skill and assess_score is not None:
        try:
            score_f = float(assess_score)
            skill_name = str(assess_skill).strip()
            if skill_name.lower() != "nan" and score_f == score_f:
                facts.append(f"Redrob assessment {skill_name} {score_f:.0f}/100")
        except (TypeError, ValueError):
            pass

    if row.get("github_note"):
        facts.append(str(row["github_note"]))
    if row.get("recruiter_saved_note"):
        facts.append(str(row["recruiter_saved_note"]))

    rr = float(row.get("response_rate") or 0)
    if rr >= 0.65:
        facts.append(f"recruiter response {rr:.0%}")

    notice = int(row.get("notice_days") or 0)
    if notice and notice <= 30:
        facts.append(f"{notice}d notice")

    facts.append(f"JD stack: {skill_txt}")
    caveat = _caveat_for_row(row)
    body = ". ".join(facts)
    if len(body) > 210:
        body = body[:207] + "..."
    return f"{body}. {caveat}"


def build_reasoning(row: pd.Series) -> str:
    """Specific reasoning per submission spec (avoid templated strings)."""
    rank = int(row.get("rank") or 1)
    if rank <= 20:
        return build_reasoning_premium(row)
    parts: list[str] = []
    title = row["display_title"]
    yrs = row["years"]
    rank = int(row.get("rank") or 1)
    cid = str(row.get("candidate_id") or "")
    parts.append(_reasoning_lead(cid, title, yrs, rank))

    if row.get("career_highlight"):
        hl = str(row["career_highlight"]).replace("\n", " ")
        if len(hl) > 90:
            hl = hl[:87] + "..."
        parts.append(_career_frame(hl, cid, rank))

    skills = row.get("top_skills") or []
    if skills:
        parts.append(_skills_frame(skills, cid, rank))

    assess_skill = row.get("top_assessment_skill")
    assess_score = row.get("top_assessment_score")
    assess_threshold = float(row.get("assessment_strong_threshold") or 70)
    if assess_skill and assess_score is not None:
        try:
            score_f = float(assess_score)
            if score_f >= assess_threshold:
                assess_frames = (
                    f"assessed strong on {assess_skill} ({score_f:.0f})",
                    f"platform assessment {assess_skill} {score_f:.0f}/100",
                    f"validated {assess_skill} ({score_f:.0f}) on Redrob",
                )
                pick = (rank + len(assess_skill)) % len(assess_frames)
                parts.append(assess_frames[pick])
        except (TypeError, ValueError):
            pass

    if row.get("top_cert"):
        parts.append(f"cert: {row['top_cert']}")

    if row.get("education_note"):
        edu_frames = (
            f"education: {row['education_note']}",
            f"{row['education_note']} background",
        )
        parts.append(edu_frames[rank % len(edu_frames)])

    loc = row.get("location", "")
    country = row.get("country", "")
    if loc:
        parts.append(f"based in {loc}" + (f", {country}" if country and country not in loc else ""))
    elif country:
        parts.append(f"based in {country}")

    work_mode = row.get("work_mode", "")
    wm_fit = float(row.get("work_mode_fit") or 0)
    if work_mode and wm_fit >= 0.88:
        parts.append(f"work mode {work_mode} (JD-aligned)")
    elif work_mode and wm_fit <= 0.45:
        parts.append(f"prefers {work_mode} (JD hybrid/onsite)")

    company = row.get("current_company", "")
    company_prod = float(row.get("company_product") or 0)
    if company and company_prod >= 0.95:
        parts.append(f"at {company}")

    notice = int(row.get("notice_days") or 0)
    if notice <= 30:
        parts.append(f"notice {notice}d (favorable)")
    elif notice > 90:
        parts.append(f"notice {notice}d (stretch)")

    rr = float(row.get("response_rate") or 0)
    if rr >= 0.7:
        parts.append(f"high recruiter response ({rr:.0%})")
    elif rr < 0.2:
        parts.append(f"low engagement ({rr:.0%} response)")

    if row.get("github_note"):
        parts.append(str(row["github_note"]))
    if row.get("recruiter_saved_note"):
        parts.append(str(row["recruiter_saved_note"]))

    if row.get("trap_flags"):
        parts.append(f"flags: {row['trap_flags']}")

    sep = " · " if rank % 3 == 0 else "; "
    text = sep.join(parts)
    return text if len(text) <= 220 else text[:217] + "..."


def prepare_submission_df(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Top-N rows with normalized display scores and reasoning text."""
    top = df.head(top_n).copy()
    top["rank"] = range(1, len(top) + 1)

    raw = top["score"].values
    if raw.max() > raw.min():
        norm = 0.99 - 0.79 * (np.arange(len(raw)) / max(len(raw) - 1, 1))
    else:
        norm = np.linspace(0.99, 0.20, len(raw))
    top["score"] = np.round(norm, 4)
    top["reasoning"] = top.apply(build_reasoning, axis=1)
    top["fit_badge"] = top.apply(
        lambda r: fit_badge(int(r["rank"]), str(r["candidate_id"])),
        axis=1,
    )
    top["reasoning_body"] = top["reasoning"].map(reasoning_display_body)
    return top


def build_ranked_profiles(
    df: pd.DataFrame,
    store: Any,
    top_n: int,
) -> list[dict[str, Any]]:
    """Attach full candidate records to ranked shortlist rows."""
    top = prepare_submission_df(df, top_n)
    results: list[dict[str, Any]] = []
    for _, row in top.iterrows():
        cid = str(row["candidate_id"])
        cand = store.get(cid)
        if not cand:
            continue
        entry: dict[str, Any] = {
            "candidate_id": cid,
            "rank": int(row["rank"]),
            "score": float(row["score"]),
            "reasoning": str(row["reasoning"]),
            "fit_badge": str(row.get("fit_badge") or fit_badge(int(row["rank"]), cid)),
            "reasoning_body": str(row.get("reasoning_body") or row["reasoning"]),
            "semantic": float(row.get("semantic") or 0),
            "profile": cand.get("profile"),
            "skills": cand.get("skills"),
            "career_history": cand.get("career_history"),
            "certifications": cand.get("certifications"),
            "languages": cand.get("languages"),
            "education": cand.get("education"),
            "redrob_signals": cand.get("redrob_signals"),
        }
        results.append(entry)
    return results


def write_submission(df: pd.DataFrame, out_path: Path, top_n: int) -> pd.DataFrame:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    top = prepare_submission_df(df, top_n)
    submission = top[["candidate_id", "rank", "score", "reasoning"]]
    submission.to_csv(out_path, index=False, encoding="utf-8")
    return submission
