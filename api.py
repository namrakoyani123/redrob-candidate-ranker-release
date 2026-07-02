#!/usr/bin/env python3
"""HTTP API: paste JD text → ranked top-N candidates with full profiles."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.candidate_store import CandidateStore
from src.embedding_index import load_embedding_index
from src.hybrid_retrieval import retrieval_cfg
from src.jd_parser import JdContext, enrich_jd_text, parse_job_description
from src.ranker import build_ranked_profiles, load_config, score_candidates
from src.semantic_rerank import _get_encoder

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"


class RankRequest(BaseModel):
    jd_text: str = Field(..., min_length=20, description="Job description plain text")
    top_n: int = Field(100, ge=1, le=500)


class AppState:
    cfg: dict[str, Any]
    store: CandidateStore
    candidates_path: Path
    project_root: Path


state = AppState()


def _jd_summary(jd: JdContext) -> dict[str, Any]:
    return {
        "title_hint": jd.title_hint,
        "min_years": jd.min_years,
        "max_years": jd.max_years,
        "ideal_years": jd.ideal_years,
        "skill_count": len(jd.skill_terms),
        "prefer_country": jd.prefer_country,
        "prefer_locations": list(jd.prefer_locations),
        "preferred_work_modes": list(jd.preferred_work_modes),
        "founding_team": jd.founding_team,
        "requires_english": jd.requires_english,
    }


@asynccontextmanager
async def lifespan(_app: FastAPI):
    config_path = ROOT / "config.yaml"
    state.cfg = load_config(config_path)
    state.project_root = ROOT
    data_cfg = state.cfg["data"]
    state.candidates_path = (ROOT / data_cfg["candidates_path"]).resolve()
    if not state.candidates_path.exists():
        raise RuntimeError(f"Candidates file not found: {state.candidates_path}")
    print(f"Loading candidates into memory: {state.candidates_path}")
    t0 = time.perf_counter()
    state.store = CandidateStore.load(state.candidates_path)
    print(f"  {len(state.store):,} candidates in {time.perf_counter() - t0:.1f}s")

    rank_cfg = state.cfg.get("ranking", {})
    retrieval = retrieval_cfg(rank_cfg)
    bi = retrieval.get("bi_encoder") or {}
    model_name = str(bi.get("model", "BAAI/bge-small-en-v1.5"))
    print(f"Warming bi-encoder: {model_name}")
    _get_encoder(model_name)

    embed_index = load_embedding_index(rank_cfg, state.project_root)
    if embed_index:
        print(f"  Precomputed embeddings: {len(embed_index):,} vectors")
    else:
        print("  No precomputed index (ranking will be slower)")

    yield


app = FastAPI(
    title="Redrob Candidate Ranker API",
    description="Rank 100K candidates against a pasted job description.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/api/sample-jd")
def sample_jd(full: bool = False) -> dict[str, Any]:
    """Return hackathon short JD header by default; ?full=true for complete text."""
    if full:
        jd_path = _resolve_jd_path()
        mode = "full"
    else:
        for candidate in (
            WEB_DIR / "official_jd_short.md",
            ROOT / "job_description_short.md",
        ):
            if candidate.is_file():
                jd_path = candidate
                break
        else:
            jd_path = _resolve_jd_path()
        mode = "short"
    return {
        "jd_text": jd_path.read_text(encoding="utf-8"),
        "mode": mode,
        "note": "Short header is enough — full JD sections are merged automatically when you rank.",
    }


def _resolve_jd_path() -> Path:
    rel = state.cfg["data"].get("job_description_path", "job_description.md")
    for candidate in (
        ROOT / rel,
        ROOT / "job_description.md",
        WEB_DIR / "job_description.md",
    ):
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    raise HTTPException(status_code=404, detail="job_description.md not found")


@app.get("/health")
def health() -> dict[str, Any]:
    jd_ok = False
    if state.cfg:
        try:
            jd_ok = _resolve_jd_path().is_file()
        except HTTPException:
            jd_ok = False
    return {
        "status": "ok",
        "ranker_version": "2026-06-24",
        "candidates_loaded": len(state.store),
        "candidates_path": str(state.candidates_path),
        "job_description_ok": jd_ok,
    }


@app.get("/")
def index() -> FileResponse:
    page = WEB_DIR / "index.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="web/index.html not found")
    return FileResponse(page)


@app.post("/api/rank")
def rank_candidates(body: RankRequest) -> dict[str, Any]:
    jd_text = body.jd_text.strip()
    if len(jd_text) < 20:
        raise HTTPException(status_code=400, detail="Job description too short")

    raw_jd = jd_text
    jd_text = enrich_jd_text(jd_text, ROOT / "job_description.md")
    rank_cfg = state.cfg.get("ranking", {})
    jd_ctx = parse_job_description(jd_text, rank_cfg)

    t0 = time.perf_counter()
    df = score_candidates(
        state.candidates_path,
        jd_text,
        state.cfg,
        jd=jd_ctx,
        project_root=state.project_root,
        output_limit=body.top_n,
    )
    profiles = build_ranked_profiles(df, state.store, body.top_n)
    elapsed = time.perf_counter() - t0

    return {
        "runtime_seconds": round(elapsed, 2),
        "top_n": len(profiles),
        "jd_parsed": _jd_summary(jd_ctx),
        "jd_enriched": len(jd_text) > len(raw_jd) + 200,
        "jd_input_chars": len(raw_jd),
        "jd_rank_chars": len(jd_text),
        "candidates": profiles,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
