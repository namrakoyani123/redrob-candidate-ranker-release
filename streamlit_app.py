"""Streamlit sandbox — rank ≤500 candidates + pasted JD (hackathon Stage 1 demo)."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SAMPLE_PATH = ROOT / "sandbox" / "sample_candidates.json"
FALLBACK_SAMPLE = (
    ROOT.parent
    / "[PUB] India_runs_data_and_ai_challenge"
    / "India_runs_data_and_ai_challenge"
    / "sample_candidates.json"
)
DEFAULT_JD = ROOT / "job_description_short.md"
FULL_JD = ROOT / "job_description.md"

from src.candidate_store import CandidateStore
from src.jd_parser import enrich_jd_text, parse_job_description
from src.ranker import build_ranked_profiles, load_config, score_candidates


def _resolve_sample_path() -> Path | None:
    for path in (SAMPLE_PATH, FALLBACK_SAMPLE):
        if path.is_file():
            return path
    return None


@st.cache_resource
def _config():
    return load_config(ROOT / "config.yaml")


def _json_array_to_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


st.set_page_config(page_title="Redrob Ranker Sandbox", layout="wide")
st.title("Redrob Candidate Ranker — Sandbox")
st.caption("Paste the short JD header — full sections are merged automatically when you rank.")

cfg = _config()
rank_cfg = cfg.get("ranking", {})

_default_jd = ""
if DEFAULT_JD.exists():
    _default_jd = DEFAULT_JD.read_text(encoding="utf-8")

col1, col2 = st.columns(2)
with col1:
    sample_path = _resolve_sample_path()
    if sample_path:
        if st.button("Load sample candidates + short JD"):
            st.session_state["jd_text"] = _default_jd
            st.session_state["sample_rows"] = json.loads(sample_path.read_text(encoding="utf-8"))
    jd_text = st.text_area(
        "Job description (short header is enough)",
        value=st.session_state.get("jd_text", _default_jd),
        height=200,
    )
with col2:
    uploaded = st.file_uploader("Candidates JSON array or JSONL", type=["json", "jsonl"])
    top_n = st.number_input("Top N", min_value=1, max_value=500, value=min(100, 100))

if st.button("Rank candidates", type="primary"):
    if len(jd_text.strip()) < 20:
        st.error("Paste a longer job description.")
        st.stop()

    rows: list[dict] = []
    if uploaded:
        raw = uploaded.read().decode("utf-8")
        if uploaded.name.endswith(".jsonl"):
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        else:
            rows = json.loads(raw)
    elif "sample_rows" in st.session_state:
        rows = st.session_state["sample_rows"]
    else:
        st.error("Upload candidates or load the official sample.")
        st.stop()

    with tempfile.TemporaryDirectory() as tmp:
        jsonl_path = Path(tmp) / "candidates.jsonl"
        _json_array_to_jsonl(rows, jsonl_path)
        store = CandidateStore.load(jsonl_path)
        jd_ctx = parse_job_description(
            enrich_jd_text(jd_text, FULL_JD),
            rank_cfg,
        )

        t0 = time.perf_counter()
        with st.spinner(f"Ranking {len(rows)} candidates…"):
            df = score_candidates(
                jsonl_path,
                jd_text,
                cfg,
                jd=jd_ctx,
                project_root=ROOT,
            )
            profiles = build_ranked_profiles(df, store, int(top_n))
        elapsed = time.perf_counter() - t0

    st.success(f"Done in {elapsed:.1f}s — {len(profiles)} results")
    st.json(
        {
            "title_hint": jd_ctx.title_hint,
            "experience": f"{jd_ctx.min_years:.0f}-{jd_ctx.max_years:.0f}y",
            "ideal_years": jd_ctx.ideal_years,
            "locations": list(jd_ctx.prefer_locations),
            "skills_tracked": len(jd_ctx.skill_terms),
        }
    )
    st.dataframe(
        [
            {
                "rank": p["rank"],
                "candidate_id": p["candidate_id"],
                "score": p["score"],
                "title": (p.get("profile") or {}).get("current_title"),
                "reasoning": p["reasoning"],
            }
            for p in profiles
        ],
        use_container_width=True,
    )

    csv_lines = ["candidate_id,rank,score,reasoning"]
    for p in profiles:
        reasoning = str(p["reasoning"]).replace('"', '""')
        csv_lines.append(f'{p["candidate_id"]},{p["rank"]},{p["score"]},"{reasoning}"')
    st.download_button(
        "Download submission CSV",
        "\n".join(csv_lines) + "\n",
        file_name="submission.csv",
        mime="text/csv",
    )
