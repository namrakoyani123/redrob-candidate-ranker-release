# Release package manifest

This folder contains **only files required** to run, reproduce, and submit the ranker at full quality.

## Included

| Path | Purpose |
|------|---------|
| `src/` | Ranking pipeline (BM25 + BGE + traps + head calibration) |
| `config.yaml` | Weights, pools, location boosts |
| `run.py` / `rank.py` | CLI rank + official reproduce |
| `api.py` + `web/` | Paste-JD website demo |
| `streamlit_app.py` + `sandbox/` | Streamlit sandbox sample |
| `data/candidates_cleaned.jsonl` | Cleaned 100K profiles (skip re-clean) |
| `data/embeddings/` | Precomputed BGE vectors (fast rank ~130â€“210s) |
| `job_description.md` | Official full JD for best submission |
| `team_redrob_candidate_ranker.csv` | Portal submission CSV |
| `scripts/validate_submission.py` | Format checker |
| `scripts/clean_data.py` | Re-clean from raw (optional) |
| `scripts/precompute_embeddings.py` | Rebuild index if needed |
| `tests/` | Unit tests (39 tests) |
| `Dockerfile` | Stage 3 CPU reproduce |
| `submission_metadata.yaml` | Portal metadata template |

## Excluded (not needed to run)

- `.venv/`, `__pycache__/`, `.pytest_cache/`
- `data/embeddings/checkpoints/` (~duplicate of final index)
- Debug scripts: `compare_ranks.py`, `profile_blind_spots.py`, `generate_analysis_report.py`
- EDA notebook artifacts, internal prompts, deck builders
- Idea submission PDF/PPTX

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/validate_submission.py team_redrob_candidate_ranker.csv
python run.py --skip-clean --jd job_description.md -o team_redrob_candidate_ranker.csv
python api.py   # http://127.0.0.1:8000
```

## Official raw data path

`rank.py --official` expects raw `candidates.jsonl` at the path in `config.yaml`
(typically the hackathon bundle). With this package, use `--skip-clean` and
`data/candidates_cleaned.jsonl` for fastest reproduce.

