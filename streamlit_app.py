"""Streamlit sandbox — official portal CSV (100K pipeline) or small custom demo."""

from __future__ import annotations

import csv
import io
import json
import tempfile
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SAMPLE_PATH = ROOT / "sandbox" / "sample_candidates.json"
DEFAULT_JD = ROOT / "job_description_short.md"
FULL_JD = ROOT / "job_description.md"
SUBMISSION_CSV = ROOT / "team_redrob_candidate_ranker.csv"

from src.candidate_store import CandidateStore
from src.jd_parser import enrich_jd_text, parse_job_description
from src.ranker import build_ranked_profiles, load_config, score_candidates
from src.submission_bundle import (
    build_official_profiles,
    is_official_hackathon_jd,
    load_bundle,
)


def _json_array_to_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _profiles_to_dataframe(profiles: list[dict]) -> list[dict]:
    return [
        {
            "rank": p["rank"],
            "candidate_id": p["candidate_id"],
            "score": p["score"],
            "title": (p.get("profile") or {}).get("current_title"),
            "reasoning": p["reasoning"],
        }
        for p in profiles
    ]


def _csv_download(profiles: list[dict], filename: str = "team_redrob_candidate_ranker.csv") -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    for p in profiles:
        writer.writerow([p["candidate_id"], p["rank"], p["score"], p["reasoning"]])
    return buf.getvalue()


@st.cache_resource
def _config():
    return load_config(ROOT / "config.yaml")


st.set_page_config(page_title="Redrob Ranker Sandbox", layout="wide")
st.title("Redrob Candidate Ranker — Sandbox")

bundle = load_bundle(ROOT)
has_official = bundle is not None and SUBMISSION_CSV.is_file()

mode = st.radio(
    "Mode",
    (
        "Official portal submission (matches team_redrob_candidate_ranker.csv)"
        if has_official
        else "Official submission (bundle missing — run export script)",
        "Custom demo (small candidate pool only)",
    ),
    index=0 if has_official else 1,
)

cfg = _config()
rank_cfg = cfg.get("ranking", {})
_default_jd = DEFAULT_JD.read_text(encoding="utf-8") if DEFAULT_JD.exists() else ""

if mode.startswith("Official"):
    st.info(
        "Shows the **exact** top 100 from the full **100K** pipeline (`run.py` + `job_description.md`). "
        "Same ranks, scores, and reasoning as `team_redrob_candidate_ranker.csv`."
    )
    if FULL_JD.exists():
        st.session_state.setdefault("jd_text", FULL_JD.read_text(encoding="utf-8"))
    jd_text = st.text_area("Job description (official full JD)", value=st.session_state.get("jd_text", _default_jd), height=160)
    top_n = st.number_input("Top N", min_value=1, max_value=100, value=100)

    if st.button("Show official submission", type="primary"):
        if not is_official_hackathon_jd(jd_text, ROOT):
            st.warning(
                "JD is not the official Redrob hackathon role — results may differ from the portal CSV. "
                "Use the bundled `job_description.md` or the short Redrob header for an exact match."
            )
        profiles = build_official_profiles(ROOT, int(top_n))
        if not profiles:
            st.error("Missing sandbox/submission_bundle.json — run scripts/export_submission_bundle.py")
            st.stop()
        jd_ctx = parse_job_description(enrich_jd_text(jd_text, FULL_JD), rank_cfg)
        st.success(f"Loaded {len(profiles)} rows from portal submission (100K pipeline output)")
        st.json(
            {
                "source": SUBMISSION_CSV.name,
                "pipeline": bundle.get("pipeline_note") if bundle else "",
                "title_hint": jd_ctx.title_hint,
                "experience": f"{jd_ctx.min_years:.0f}-{jd_ctx.max_years:.0f}y",
                "skills_tracked": len(jd_ctx.skill_terms),
            }
        )
        st.dataframe(_profiles_to_dataframe(profiles), use_container_width=True)
        st.download_button(
            "Download submission CSV",
            _csv_download(profiles),
            file_name=SUBMISSION_CSV.name,
            mime="text/csv",
        )

else:
    st.warning(
        "Custom demo ranks only the **uploaded or sample pool** (≤500). "
        "Output will **not** match `team_redrob_candidate_ranker.csv`."
    )
    col1, col2 = st.columns(2)
    with col1:
        if SAMPLE_PATH.is_file() and st.button("Load hackathon sample (50 candidates)"):
            st.session_state["jd_text"] = _default_jd
            st.session_state["sample_rows"] = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
        jd_text = st.text_area("Job description", value=st.session_state.get("jd_text", _default_jd), height=160)
    with col2:
        uploaded = st.file_uploader("Optional: small JSON / JSONL pool", type=["json", "jsonl"])
        top_n = st.number_input("Top N", min_value=1, max_value=500, value=100)

    if st.button("Rank custom pool", type="primary"):
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
            st.error("Upload a small pool or load the hackathon sample.")
            st.stop()

        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = Path(tmp) / "candidates.jsonl"
            _json_array_to_jsonl(rows, jsonl_path)
            store = CandidateStore.load(jsonl_path)
            enriched = enrich_jd_text(jd_text, FULL_JD)
            jd_ctx = parse_job_description(enriched, rank_cfg)
            t0 = time.perf_counter()
            with st.spinner(f"Ranking {len(rows)} candidates…"):
                df = score_candidates(jsonl_path, enriched, cfg, jd=jd_ctx, project_root=ROOT)
                profiles = build_ranked_profiles(df, store, int(top_n))
            elapsed = time.perf_counter() - t0

        st.success(f"Done in {elapsed:.1f}s — {len(profiles)} results (custom pool only)")
        st.dataframe(_profiles_to_dataframe(profiles), use_container_width=True)
        st.download_button(
            "Download CSV",
            _csv_download(profiles, "custom_rank.csv"),
            file_name="custom_rank.csv",
            mime="text/csv",
        )
