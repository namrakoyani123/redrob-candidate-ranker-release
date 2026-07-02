# Redrob Intelligent Candidate Ranker

Rank candidates from a JSONL pool against **any job description** and produce a trusted top-100 shortlist. Hybrid scoring combines semantic retrieval (BM25 + BGE) with recruiter-style structured signals and explicit honeypot guards — not keyword filters alone.

## How it works

1. **Clean the data** — normalize text, dedupe skills, fix inverted salaries, reconcile YOE, downgrade fake expert skills.
2. **Parse your JD** — role titles, skills, experience band, locations, work mode, founding-team cues, disqualifiers.
3. **Funnel 100K** — cheap prescreen (6K) → structured scoring + traps (2K): role, skills, career, product company, platform signals.
4. **Hybrid semantic retrieval** — field-weighted **BM25** on 2K + **BGE-small** dense scores, fused with **RRF** (reciprocal rank fusion).
5. **Guards** — trap penalties, YOE floor, role/skill listing guards.
6. **Output** — CSV with `candidate_id`, `rank`, `score`, `reasoning`; auto-copy to `comparison/csvs/ours.csv` for regression.

### Architecture

```
100K JSONL → clean → prescreen 6K → structured + traps 2K → BM25 + BGE + RRF → blend → top 100 CSV
```

**With precomputed embeddings** (`data/embeddings/`): BGE looks up vectors for the full 2K semantic pool (~**74–83s** on 100K CPU).

**Without index** (live BGE on BM25 top-500): still under the 300s spec (~**298s**).

One-time precompute: `python scripts/precompute_embeddings.py --resume` (~3–6 hours CPU for 100K; checkpoint-safe).

## Quick start

```bash
cd redrob-candidate-ranker
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt

# Step 1: Clean raw candidate data
python scripts/clean_data.py
# or: python run.py --clean-first

# Step 2a (one-time): precompute BGE embeddings — use --resume if interrupted
python scripts/precompute_embeddings.py --resume

# Step 2b: Rank against JD → top 100 CSV
python run.py --skip-clean
# Spec alias: python rank.py --skip-clean -o outputs/submission.csv

# Official portal reproduce (raw 100K + clean + rank)
python rank.py --official --jd job_description.md -o team_YOUR_ID.csv

# BM25-only (no BGE — faster, lower semantic quality)
python run.py --skip-clean --no-semantic-rerank
```

`run.py` auto-cleans on first run if `data/candidates_cleaned.jsonl` is missing.

Default paths: `job_description.md` → `outputs/submission.csv`.

## Use any job description

```bash
python run.py --jd path/to/your_job_description.md

python run.py \
  --jd jobs/senior_backend_engineer.md \
  --candidates data/candidates_cleaned.jsonl \
  --output outputs/backend_shortlist.csv \
  --top-n 100
```

Supported formats: `.md`, `.txt` with Must-have, Nice-to-have, Disqualifiers, Location, Experience sections. No code changes needed per JD.

### Example JD structure

```markdown
Job Description: Senior Backend Engineer

Location: Bangalore, India
Experience Required: 4–8 years

Must-have:
- Python, FastAPI, PostgreSQL
- Microservices and REST API design

Disqualifiers:
- Pure frontend roles without backend experience
```

## Validate output

```bash
python scripts/validate_submission.py outputs/submission.csv
python -m pytest tests/ -q
```

`validate_submission.py` checks columns, ranks 1–100, monotonic scores, unique score spread, trap-like titles in top-20, and IDs against official `candidates.jsonl`.

## Configuration

`config.yaml` sets paths, funnel sizes, score weights, and signal tuning. Override via CLI; the JD drives role/skill/location/work-mode signals.

Notable options:

| Key | Purpose |
|-----|---------|
| `semantic_pool_size` | Structured shortlist before BM25+BGE (default 2000) |
| `precomputed.path` | BGE index directory (`data/embeddings`) |
| `location_corridor_boost` | Extra boost for Pune/Noida when JD emphasizes corridor |
| `skill_assessment_blend` | Blend JD-matched platform assessments into skill_fit (max 15%) |
| `comparison_csv_path` | Regression export after each rank run |

## Scoring weights (default)

| Signal | Weight | JD-driven? |
|--------|--------|------------|
| Semantic (BM25 + BGE + RRF) | 16% | Yes — hybrid match vs JD text |
| Role fit | 30% | Yes — roles from JD |
| Skill fit | 18% | Yes — skills from JD + light platform assessment blend |
| Career evidence | 22% | Yes — JD terms in career/summary |
| Product fit | 8% | Career + `current_company` when JD cares about product |
| Certification fit | 4% | ML-relevant certs when present |
| Education fit | 2% | tier-1/2 CS/ML/EE field boost |
| Platform signals | 10% | Full `redrob_signals` engagement bundle |

### `redrob_signals` usage (22/23 fields)

| Field | Used? | Where |
|-------|-------|-------|
| `notice_period_days` | Yes | Location feasibility + reasoning |
| `preferred_work_mode` | Yes | Work-mode fit (hybrid/onsite/remote); founding-team boost |
| `skill_assessment_scores` | Yes | JD-matched blend into skill_fit; reasoning when ≥70 |
| `recruiter_response_rate` | Yes | Platform score + reasoning |
| `offer_acceptance_rate` | Yes | Neutral when `-1`; light platform boost |
| `connection_count` | Yes | Light platform network signal |
| `github_activity_score` | Yes | Platform score + reasoning when ≥65 |
| `saved_by_recruiters_30d` | Yes | Platform score + reasoning when ≥5 |
| `profile_completeness_score` | Yes | Platform score |
| `profile_views_received_30d` | Yes | Marketplace activity signal |
| `applications_submitted_30d` | Yes | Marketplace activity signal |
| `search_appearance_30d` | Yes | Marketplace activity signal |
| `avg_response_time_hours` | Yes | Response-time signal |
| `endorsements_received` | Yes | Platform endorsements signal |
| `interview_completion_rate` | Yes | Platform score |
| `last_active_date` | Yes | Activity score |
| `open_to_work_flag` | Yes | Activity + location |
| `willing_to_relocate` | Yes | Location feasibility |
| `expected_salary_range_inr_lpa` | Yes | Salary band check |
| `verified_email/phone/linkedin_connected` | Yes | Platform score |
| `signup_date` | No | Not predictive for rank |
| Skill-level `endorsements` on `skills[]` | Yes | Skill fit |

### Guards and reasoning

- **YOE floor** — score ×0.01 when years &lt; JD `min_years`
- **Traps** — keyword stuffing, title/career mismatch, job hopper, inactive profiles, LangChain-only, research-only-no-prod
- **Reasoning** — 9 opener variants, varied career/skill phrasing, education/GitHub/recruiter-saved notes, `;` vs `·` separators (Stage 4)

## Docker (Stage 3 reproduce)

```bash
docker build -t redrob-ranker .
docker run --rm -v "%cd%/data:/app/data" redrob-ranker python rank.py --skip-clean -o outputs/submission.csv
```

Mount `data/embeddings/` for sub-90s rank on 100K. See `sandbox/README.md` for Streamlit Cloud deploy.

## Streamlit sandbox (Stage 1)

```bash
streamlit run streamlit_app.py
```

Loads official `sample_candidates.json` when available; paste any JD and export ranked CSV with full profiles.

## Project layout

```
redrob-candidate-ranker/
  run.py                      # CLI entry point
  rank.py                     # Spec alias (portal reproduce)
  api.py                      # FastAPI + web UI (paste JD → profiles)
  streamlit_app.py            # Hackathon sandbox demo
  Dockerfile
  config.yaml
  submission_metadata.yaml    # Reproduce commands, runtime, stack
  job_description.md
  src/
    cleaner.py
    jd_parser.py
    ranker.py
    hybrid_retrieval.py       # BM25 + BGE + RRF
    semantic_rerank.py        # BGE bi-encoder
    embedding_index.py        # Precomputed vector lookup
    candidate_store.py        # In-memory id → full candidate (API)
    profile.py                # Structured + platform signals
    honeypot.py
    constants.py
  scripts/
    clean_data.py
    precompute_embeddings.py  # One-time BGE index (--resume)
    validate_submission.py
  data/
    candidates_cleaned.jsonl
    embeddings/               # vectors.npy + candidate_ids.json (after precompute)
  web/
    index.html                # Paste-JD ranking UI
  outputs/
    submission.csv
  docs/
    reference/
    eda/
```

## Runtime (measured, Windows CPU, 100K)

| Mode | Time |
|------|------|
| Precomputed index + BM25 + BGE + RRF | **~74–83s** |
| Live BGE (BM25 top-500) + RRF | **~298s** |
| BM25-only (`--no-semantic-rerank`) | **~200s** |

See `submission_metadata.yaml` for reproduce commands and spec compliance notes.

## Web API (paste JD → top-100 profiles)

Start the API server (loads candidates + warms BGE on startup):

```bash
pip install -r requirements.txt
python api.py
# or: uvicorn api:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** — paste a job description and click **Rank candidates**.

### HTTP endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/health` | Liveness + candidate count |
| `POST` | `/api/rank` | Rank against pasted JD text |

**Request** (`POST /api/rank`):

```json
{
  "jd_text": "Senior AI Engineer… Must-have: Python, FAISS…",
  "top_n": 100
}
```

**Response** includes `runtime_seconds`, `jd_parsed`, and `candidates[]` with full `profile`, `skills`, `career_history`, `certifications`, `redrob_signals`, plus `rank`, `score`, `reasoning`.

Requires `data/candidates_cleaned.jsonl` and (recommended) `data/embeddings/` for fast ranking.

## Production roadmap

- **FAISS** — ANN on precomputed vectors for multi-JD / dense-first prescreen
- **Cross-encoder** — optional rerank on final top 50–100

## Data

Expects one JSON object per line (`candidate_id`, `profile`, `career_history`, `skills`, `certifications`, `languages`, `redrob_signals`, …). Paths in `config.yaml` or `--candidates`.
