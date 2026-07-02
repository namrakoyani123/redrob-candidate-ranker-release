"""Tests for BM25 + RRF hybrid retrieval."""

import numpy as np

from src.hybrid_retrieval import bm25_scores, reciprocal_rank_fusion


def test_bm25_prefers_keyword_overlap():
    docs = [
        "python pytorch ranking embeddings faiss",
        "java spring accounting ledger",
        "marketing seo content strategy",
    ]
    scores = bm25_scores("python ranking embeddings", docs)
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]


def test_rrf_merges_two_rankings():
    bm25 = np.array([0.9, 0.1, 0.5], dtype=np.float32)
    dense = np.array([0.2, 0.95, 0.4], dtype=np.float32)
    fused = reciprocal_rank_fusion([bm25, dense], k=60)
    assert fused.argmax() in (0, 1)
