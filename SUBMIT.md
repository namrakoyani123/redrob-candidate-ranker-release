# Portal upload checklist

## Files ready

| Artifact | Path |
|----------|------|
| **Submission CSV** | `team_redrob_candidate_ranker.csv` |
| **Metadata** | `submission_metadata.yaml` |
| **Reproduce** | `python rank.py --official --jd job_description.md -o team_redrob_candidate_ranker.csv` |

Validate before upload:

```bash
python scripts/validate_submission.py team_redrob_candidate_ranker.csv
python -m pytest tests/ -q
```

## GitHub (public repo)

```bash
cd redrob-candidate-ranker
git init
git lfs install
git lfs track "data/embeddings/vectors.npy"
git add .
git commit -m "Redrob hackathon submission: hybrid BM25+BGE ranker"
gh repo create redrob-candidate-ranker --public --source=. --push
```

Replace `YOUR_GITHUB_USERNAME` in `submission_metadata.yaml` with your GitHub username.

**Note:** `vectors.npy` (~147 MB) requires **Git LFS**. Without LFS, reviewers run:

```bash
python scripts/precompute_embeddings.py --resume
```

## Streamlit sandbox (Stage 1)

1. Push repo to GitHub.
2. https://share.streamlit.io → New app → `streamlit_app.py`.
3. Paste URL into portal as `sandbox_url` (update `submission_metadata.yaml`).

Bundled sample: `sandbox/sample_candidates.json` (official hackathon sample).

## Portal upload

1. Upload `team_redrob_candidate_ranker.csv` (rename if portal assigned a different team ID).
2. Paste GitHub URL and Streamlit sandbox URL.
3. Declare AI tool use (Cursor) per form.
4. **Max 3 submissions** — this should be your final or near-final entry.

## Blind-spot calibration (this build)

| Candidate | Before | After |
|-----------|--------|-------|
| CAND_0088025 | #33 | **#18** |
| CAND_0009691 | outside top 100 | **#70** |
| CAND_0006557 | outside top 100 | **#44** |
