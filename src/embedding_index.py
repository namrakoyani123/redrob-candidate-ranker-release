"""Cosine scores via precomputed vectors (encode JD only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .semantic_rerank import encode_query_vector


class EmbeddingIndex:
    """Normalized bi-encoder vectors keyed by candidate_id."""

    def __init__(
        self,
        candidate_ids: list[str],
        vectors: np.ndarray,
        model_name: str,
        max_chars: int,
    ) -> None:
        if len(candidate_ids) != len(vectors):
            raise ValueError("candidate_ids and vectors length mismatch")
        self.candidate_ids = candidate_ids
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.model_name = model_name
        self.max_chars = max_chars
        self._id_to_row = {cid: i for i, cid in enumerate(candidate_ids)}

    def __len__(self) -> int:
        return len(self.candidate_ids)

    @classmethod
    def load(cls, index_dir: Path) -> EmbeddingIndex:
        meta_path = index_dir / "meta.json"
        meta: dict[str, Any] = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        ids = json.loads((index_dir / "candidate_ids.json").read_text(encoding="utf-8"))
        vectors = np.load(index_dir / "vectors.npy", mmap_mode="r")
        return cls(
            candidate_ids=ids,
            vectors=vectors,
            model_name=str(meta.get("model", "BAAI/bge-small-en-v1.5")),
            max_chars=int(meta.get("max_chars", 2500)),
        )

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "vectors.npy", self.vectors)
        (out_dir / "candidate_ids.json").write_text(
            json.dumps(self.candidate_ids), encoding="utf-8"
        )
        meta = {
            "model": self.model_name,
            "max_chars": self.max_chars,
            "count": len(self.candidate_ids),
            "dim": int(self.vectors.shape[1]) if len(self.vectors) else 0,
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def matches_config(self, model_name: str, max_chars: int) -> bool:
        return self.model_name == model_name and self.max_chars == max_chars

    def dense_scores_for_ids(
        self,
        jd_text: str,
        pool_ids: list[str],
        model_name: str,
        max_chars: int,
    ) -> np.ndarray:
        n = len(pool_ids)
        scores = np.zeros(n, dtype=np.float32)
        if n == 0:
            return scores

        rows: list[int] = []
        valid_idx: list[int] = []
        for i, cid in enumerate(pool_ids):
            row = self._id_to_row.get(cid)
            if row is not None:
                rows.append(row)
                valid_idx.append(i)

        if not rows:
            return scores

        jd_vec = encode_query_vector(jd_text, model_name=model_name, max_chars=max_chars)
        gathered = np.asarray(self.vectors[rows], dtype=np.float32)
        scores[valid_idx] = (gathered @ jd_vec).ravel()
        return scores


def load_embedding_index(rank_cfg: dict[str, Any], root: Path | None = None) -> EmbeddingIndex | None:
    pre = (rank_cfg.get("semantic_retrieval") or {}).get("precomputed") or {}
    if not pre.get("enabled", True):
        return None
    index_dir = Path(pre.get("path", "data/embeddings"))
    if root:
        index_dir = (root / index_dir).resolve()
    vectors = index_dir / "vectors.npy"
    ids = index_dir / "candidate_ids.json"
    if not vectors.exists() or not ids.exists():
        return None
    return EmbeddingIndex.load(index_dir)
