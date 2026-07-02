# Hugging Face Spaces / Streamlit Cloud sandbox

Deploy `streamlit_app.py` for hackathon **Stage 1 sandbox** requirement.

## Streamlit Cloud (free)

1. Push `redrob-candidate-ranker` to GitHub (public).
2. https://streamlit.io/cloud → New app → repo root `streamlit_app.py`.
3. Add `sample_candidates.json` path or rely on upload in UI.
4. Sandbox URL → paste into portal metadata.

## Local

```bash
pip install streamlit
streamlit run streamlit_app.py
```

## Docker (full rank, CPU)

```bash
docker build -t redrob-ranker .
docker run --rm -v "%cd%/data:/app/data" redrob-ranker python rank.py --skip-clean -o outputs/submission.csv
```

Mount `data/embeddings/` for sub-90s rank on 100K.

## Official reproduce (portal / Stage 3)

```bash
python rank.py --official --out team_YOUR_ID.csv
```

Cleans raw `candidates.jsonl` then ranks with `job_description.md`.
