"""BM25 + bi-encoder retrieval fused with reciprocal rank fusion (RRF)."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from .embedding_index import EmbeddingIndex
from .semantic_rerank import embedding_similarity

_TOKEN_RE = re.compile(r"[a-z0-9+#./-]+", re.I)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def bm25_scores(jd_text: str, doc_texts: list[str]) -> np.ndarray:
    """Okapi BM25 scores of each document against the JD query."""
    if not doc_texts:
        return np.array([], dtype=np.float32)
    from rank_bm25 import BM25Okapi

    corpus = [_tokenize(t) for t in doc_texts]
    query = _tokenize(jd_text)
    if not query:
        return np.zeros(len(doc_texts), dtype=np.float32)
    bm25 = BM25Okapi(corpus)
    return np.asarray(bm25.get_scores(query), dtype=np.float32)


def reciprocal_rank_fusion(
    score_vectors: list[np.ndarray],
    k: int = 60,
) -> np.ndarray:
    """Merge ranked lists via RRF; higher score = better match."""
    if not score_vectors:
        return np.array([], dtype=np.float32)
    n = len(score_vectors[0])
    fused = np.zeros(n, dtype=np.float64)
    for scores in score_vectors:
        order = np.argsort(-scores, kind="stable")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(1, n + 1, dtype=np.float64)
        fused += 1.0 / (float(k) + ranks)
    return fused.astype(np.float32)


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    lo, hi = float(scores.min()), float(scores.max())
    if hi <= lo:
        return np.full_like(scores, 0.5, dtype=np.float32)
    return ((scores - lo) / (hi - lo)).astype(np.float32)


def retrieval_cfg(rank_cfg: dict[str, Any]) -> dict[str, Any]:
    raw = rank_cfg.get("semantic_retrieval") or {}
    legacy = rank_cfg.get("semantic_rerank") or {}
    defaults: dict[str, Any] = {
        "method": "hybrid",
        "rrf_k": 60,
        "field_weights": {
            "title": 3,
            "skills": 2,
            "career": 2,
            "summary": 1,
            "certs": 1,
        },
        "bm25": {"enabled": True},
        "bi_encoder": {
            "enabled": legacy.get("enabled", True),
            "model": "BAAI/bge-small-en-v1.5",
            "pool_size": int(legacy.get("pool_size", 1000)),
            "batch_size": 64,
            "max_chars": 2500,
        },
        "precomputed": {
            "enabled": True,
            "path": "data/embeddings",
        },
    }
    merged = {**defaults, **raw}
    bi = {**defaults["bi_encoder"], **(raw.get("bi_encoder") or {})}
    if legacy.get("model") and not raw.get("bi_encoder"):
        # Backward compat: old MiniLM config only if no new bi_encoder block.
        if "bge" not in str(legacy.get("model", "")).lower():
            bi["model"] = legacy["model"]
    merged["bi_encoder"] = bi
    merged["bm25"] = {**defaults["bm25"], **(raw.get("bm25") or {})}
    merged["precomputed"] = {**defaults["precomputed"], **(raw.get("precomputed") or {})}
    return merged


def hybrid_semantic_scores(
    jd_text: str,
    canonical_texts: list[str],
    sparse_texts: list[str],
    rank_cfg: dict[str, Any],
    pool_ids: list[str] | None = None,
    embedding_index: EmbeddingIndex | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """BM25 on field-weighted text + BGE dense scores merged with RRF."""
    cfg = retrieval_cfg(rank_cfg)
    n = len(canonical_texts)
    if n == 0:
        return np.array([], dtype=np.float32), {}

    sparse = sparse_texts if cfg.get("bm25", {}).get("enabled", True) else canonical_texts
    bm25 = bm25_scores(jd_text, sparse)
    sources: list[np.ndarray] = [bm25]

    bi_cfg = cfg.get("bi_encoder") or {}
    if bi_cfg.get("enabled", True):
        model_name = str(bi_cfg.get("model", "BAAI/bge-small-en-v1.5"))
        max_chars = int(bi_cfg.get("max_chars", 2500))
        batch_size = int(bi_cfg.get("batch_size", 64))
        dense_full = np.zeros(n, dtype=np.float32)

        use_index = (
            embedding_index is not None
            and pool_ids is not None
            and embedding_index.matches_config(model_name, max_chars)
        )

        if use_index:
            dense_full = embedding_index.dense_scores_for_ids(
                jd_text, pool_ids, model_name=model_name, max_chars=max_chars
            )
        else:
            pool_n = min(int(bi_cfg.get("pool_size", 1000)), n)
            top_idx = np.argsort(-bm25, kind="stable")[:pool_n]
            embed_subset = [canonical_texts[i] for i in top_idx]
            embed = embedding_similarity(
                jd_text,
                embed_subset,
                model_name=model_name,
                batch_size=batch_size,
                max_chars=max_chars,
                query_prefix=str(bi_cfg.get("query_prefix", "")),
                doc_prefix=str(bi_cfg.get("doc_prefix", "")),
            )
            dense_full[top_idx] = embed

        sources.append(dense_full)

    if len(sources) == 1:
        return _normalize_scores(sources[0]), {}

    fused = reciprocal_rank_fusion(sources, k=int(cfg.get("rrf_k", 60)))
    return _normalize_scores(fused), {}
