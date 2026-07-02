#!/usr/bin/env python3
"""One-time: embed all candidates with BGE-small for fast rank-time lookup."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.embedding_index import EmbeddingIndex
from src.profile import build_canonical_text
from src.ranker import iter_candidates, load_config
from src.semantic_rerank import encode_document_vectors


def _log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute BGE embeddings for all candidates.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--candidates", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--resume", action="store_true", help="Skip checkpoint chunks already on disk.")
    args = parser.parse_args()

    root = args.config.resolve().parent
    cfg = load_config(args.config)
    rank_cfg = cfg["ranking"]
    retrieval = rank_cfg.get("semantic_retrieval") or {}
    bi = retrieval.get("bi_encoder") or {}
    model_name = str(bi.get("model", "BAAI/bge-small-en-v1.5"))
    max_chars = int(bi.get("max_chars", 2500))
    batch_size = int(bi.get("batch_size", 64))

    pre = retrieval.get("precomputed") or {}
    out_dir = args.out or root / pre.get("path", "data/embeddings")
    out_dir = out_dir.resolve()
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cand_path = args.candidates or root / cfg["data"]["candidates_path"]
    cand_path = cand_path.resolve()
    if not cand_path.exists():
        raise FileNotFoundError(cand_path)

    ids_path = ckpt_dir / "candidate_ids.json"
    if args.resume and ids_path.exists():
        ids = json.loads(ids_path.read_text(encoding="utf-8"))
        _log(f"Resume: reusing {len(ids):,} candidate ids from checkpoint")
        texts = []
        id_set = set(ids)
        for cand in iter_candidates(cand_path):
            if cand["candidate_id"] in id_set:
                texts.append(build_canonical_text(cand))
        if len(texts) != len(ids):
            _log("Warning: id/text count mismatch — rebuilding id list from scratch")
            ids, texts = [], []
            for cand in iter_candidates(cand_path):
                ids.append(cand["candidate_id"])
                texts.append(build_canonical_text(cand))
            ids_path.write_text(json.dumps(ids), encoding="utf-8")
    else:
        ids, texts = [], []
        t_load = time.perf_counter()
        for cand in iter_candidates(cand_path):
            ids.append(cand["candidate_id"])
            texts.append(build_canonical_text(cand))
        ids_path.write_text(json.dumps(ids), encoding="utf-8")
        _log(f"Loaded {len(ids):,} profiles in {time.perf_counter() - t_load:.1f}s")

    _log(f"Candidates: {cand_path}")
    _log(f"Model: {model_name} (max_chars={max_chars}, batch={batch_size})")
    _log(f"Output: {out_dir}")

    chunk = max(500, int(args.chunk_size))
    n_chunks = (len(texts) + chunk - 1) // chunk
    t1 = time.perf_counter()
    parts: list[np.ndarray] = []

    for ci in range(n_chunks):
        start = ci * chunk
        end = min(start + chunk, len(texts))
        ckpt_file = ckpt_dir / f"chunk_{ci:04d}.npy"
        if args.resume and ckpt_file.exists():
            parts.append(np.load(ckpt_file))
            _log(f"  Chunk {ci + 1}/{n_chunks}: loaded checkpoint ({end:,}/{len(texts):,})")
            continue
        part = encode_document_vectors(
            texts[start:end],
            model_name=model_name,
            batch_size=batch_size,
            max_chars=max_chars,
        )
        np.save(ckpt_file, part)
        parts.append(part)
        _log(f"  Chunk {ci + 1}/{n_chunks}: encoded {end:,}/{len(texts):,} ({time.perf_counter() - t1:.0f}s)")

    vectors = np.vstack(parts)
    _log(f"Stacked {len(ids):,} vectors in {time.perf_counter() - t1:.1f}s")

    index = EmbeddingIndex(ids, vectors, model_name=model_name, max_chars=max_chars)
    index.save(out_dir)
    _log(f"Saved index to {out_dir} ({vectors.shape[1]}-dim)")
    _log(f"Done. Re-run ranking with: python run.py --skip-clean")


if __name__ == "__main__":
    main()
