# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Precomputed embeddings optional — mount data/embeddings/ at run time for fast rank.
# One-time: python scripts/precompute_embeddings.py --resume

ENV PYTHONUNBUFFERED=1

# Official reproduce (rank step only; assumes cleaned data or runs clean-first):
# docker run --rm -v "$(pwd)/data:/app/data" redrob-ranker \
#   python rank.py --official --out /app/outputs/submission.csv
CMD ["python", "rank.py", "--skip-clean", "-o", "outputs/submission.csv"]
