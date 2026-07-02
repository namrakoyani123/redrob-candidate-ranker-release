"""Bi-encoder dense retrieval (BGE / MiniLM) for hybrid semantic ranking."""

from __future__ import annotations

from typing import Any

import numpy as np

_ENCODER = None
_ENCODER_NAME: str | None = None

# BGE v1.5 retrieval instructions (empty for generic MiniLM models).
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_BGE_DOC_PREFIX = ""


def _model_prefixes(model_name: str, query_prefix: str, doc_prefix: str) -> tuple[str, str]:
    if query_prefix or doc_prefix:
        return query_prefix, doc_prefix
    if "bge" in model_name.lower():
        return _BGE_QUERY_PREFIX, _BGE_DOC_PREFIX
    return "", ""


def _get_encoder(model_name: str):
    global _ENCODER, _ENCODER_NAME
    if _ENCODER is None or _ENCODER_NAME != model_name:
        from sentence_transformers import SentenceTransformer

        _ENCODER = SentenceTransformer(model_name)
        _ENCODER_NAME = model_name
    return _ENCODER


def encode_query_vector(
    jd_text: str,
    model_name: str = "BAAI/bge-small-en-v1.5",
    max_chars: int = 4000,
    query_prefix: str = "",
) -> np.ndarray:
    """Single normalized JD/query embedding."""
    q_pre, _ = _model_prefixes(model_name, query_prefix, "")
    model = _get_encoder(model_name)
    jd_trim = jd_text[:max_chars] if jd_text else ""
    vec = model.encode(
        [f"{q_pre}{jd_trim}"],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vec, dtype=np.float32).reshape(-1, 1)


def encode_document_vectors(
    doc_texts: list[str],
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 64,
    max_chars: int = 4000,
    doc_prefix: str = "",
) -> np.ndarray:
    """Batch normalized document embeddings."""
    if not doc_texts:
        return np.zeros((0, 0), dtype=np.float32)
    _, d_pre = _model_prefixes(model_name, "", doc_prefix)
    model = _get_encoder(model_name)
    trimmed = [f"{d_pre}{(t[:max_chars] if t else '')}" for t in doc_texts]
    return np.asarray(
        model.encode(
            trimmed,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )


def embedding_similarity(
    jd_text: str,
    doc_texts: list[str],
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 64,
    max_chars: int = 4000,
    query_prefix: str = "",
    doc_prefix: str = "",
) -> np.ndarray:
    """Cosine similarity between JD and each document (normalized embeddings)."""
    if not doc_texts:
        return np.array([], dtype=np.float32)
    q_pre, d_pre = _model_prefixes(model_name, query_prefix, doc_prefix)
    model = _get_encoder(model_name)
    jd_trim = jd_text[:max_chars] if jd_text else ""
    trimmed = [t[:max_chars] if t else "" for t in doc_texts]
    jd_vec = model.encode(
        [f"{q_pre}{jd_trim}"],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    doc_vecs = model.encode(
        [f"{d_pre}{t}" for t in trimmed],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(doc_vecs @ jd_vec.T, dtype=np.float32).ravel()


def blend_semantic_scores(
    sparse_scores: np.ndarray,
    embed_scores: np.ndarray,
    embed_weight: float,
) -> np.ndarray:
    """Legacy weighted blend (superseded by RRF in hybrid_retrieval)."""
    w = min(max(float(embed_weight), 0.0), 1.0)
    return w * embed_scores + (1.0 - w) * sparse_scores


def rerank_cfg(rank_cfg: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible config accessor."""
    raw = rank_cfg.get("semantic_retrieval") or {}
    legacy = rank_cfg.get("semantic_rerank") or {}
    bi = {**(raw.get("bi_encoder") or {}), **legacy}
    return {
        "enabled": bi.get("enabled", True),
        "model": bi.get("model", "BAAI/bge-small-en-v1.5"),
        "pool_size": int(rank_cfg.get("semantic_pool_size", 2000)),
        "embed_weight": 0.65,
        "batch_size": bi.get("batch_size", 64),
        "max_chars": bi.get("max_chars", 4000),
    }
