# Hugging Face Spaces / Streamlit Cloud sandbox

Deploy `streamlit_app.py` for hackathon **Stage 1 sandbox** requirement.

## Official submission mode (default)

The sandbox shows the **exact** portal output from the full 100K pipeline:

- `team_redrob_candidate_ranker.csv`
- `sandbox/submission_bundle.json` (top-100 profiles for display)

Regenerate after re-ranking:

```bash
python scripts/export_submission_bundle.py
```

In the UI: **Official portal submission** → **Show official submission**.

Top rank should be `CAND_0018499` (matches CSV).

## Custom demo mode

**Custom demo** ranks a small uploaded/sample pool only — output will **not** match the portal CSV.

## Streamlit Cloud

1. Push repo to GitHub (include `sandbox/submission_bundle.json`).
2. https://share.streamlit.io → New app → `streamlit_app.py`.
3. Paste sandbox URL into `submission_metadata.yaml`.

## Local

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Full 100K reproduce (Stage 3)

```bash
git lfs pull
python run.py --skip-clean --jd job_description.md -o team_redrob_candidate_ranker.csv
```
